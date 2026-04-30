"""
Training entry point for the CVDMS EuroSAT single-label classifier.

This script wires together:

    config.yaml
    CVDMS metadata/manifests from S3
    cvdms_training_common DataLoaders
    project-specific ResNet18 model utilities
    project-specific staged fine-tuning loop

Run from the project root:

    python source/train.py --config config.yaml
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import torch
import torch.nn as nn
import yaml

from cvdms_training_common.dataloaders.single_label import build_single_label_data_bundle

from models import build_model
from staged_training import run_staged_training

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a staged ResNet classifier from CVDMS dataset artifacts."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to project config YAML file.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    training_config = require_dict(config.get("training"), "training")
    data_config = require_dict(config.get("data"), "data")
    model_config = require_dict(config.get("model"), "model")
    logging_config = require_dict(config.get("logging"), "logging")
    aws_config = config.get("aws") or {}

    seed = int(training_config.get("seed", 42))
    set_seed(seed)

    metadata_uri = require_nonempty_string(config.get("metadata_uri"), "metadata_uri")

    s3_client = make_s3_client(
        profile_name=aws_config.get("profile_name"),
        region_name=aws_config.get("region_name"),
    )

    data_bundle = build_single_label_data_bundle(
        metadata_uri=metadata_uri,
        batch_size=require_positive_int(data_config.get("batch_size"), "data.batch_size"),
        num_workers=require_nonnegative_int(data_config.get("num_workers", 0), "data.num_workers"),
        image_size=require_positive_int(data_config.get("image_size"), "data.image_size"),
        s3_client=s3_client,
        pin_memory=data_config.get("pin_memory"),
        persistent_workers=data_config.get("persistent_workers"),
        prefetch_factor=data_config.get("prefetch_factor"),
        drop_last_train=bool(data_config.get("drop_last_train", False)),
        distributed=False,
    )

    model = build_model(
        model_name=require_nonempty_string(model_config.get("name"), "model.name"),
        num_classes=data_bundle.num_classes,
        pretrained=bool(model_config.get("pretrained", True)),
    )

    loss_fn = nn.CrossEntropyLoss()

    phases = require_phase_list(training_config.get("phases"), "training.phases")
    output_dir = require_nonempty_string(logging_config.get("output_dir"), "logging.output_dir")
    tensorboard_dir = logging_config.get("tensorboard_dir")

    result = run_staged_training(
        model=model,
        train_loader=data_bundle.train_loader,
        val_loader=data_bundle.val_loader,
        test_loader=data_bundle.test_loader,
        loss_fn=loss_fn,
        cvdms_metadata=data_bundle.metadata,
        model_name=require_nonempty_string(model_config.get("name"), "model.name"),
        phases=phases,
        output_dir=output_dir,
        tensorboard_dir=tensorboard_dir,
        batch_size=require_positive_int(data_config.get("batch_size"), "data.batch_size"),
        channels=require_positive_int(data_config.get("channels", 3), "data.channels"),
        image_size=require_positive_int(data_config.get("image_size"), "data.image_size"),
        metadata_uri=metadata_uri,
        device=training_config.get("device"),
        best_metric_name=str(training_config.get("best_metric_name", "val_f1_macro")),
        best_metric_mode=str(training_config.get("best_metric_mode", "max")),
        hyperparameters=build_hyperparameter_summary(
            config=config,
            seed=seed,
            num_classes=data_bundle.num_classes,
            split_counts=data_bundle.split_counts,
        ),
        extra_checkpoint_metadata={
            "project": "cvdms_eurosat_single_label_classifier",
            "dataset_id": data_bundle.metadata.dataset_id,
            "dataset_version": data_bundle.metadata.version,
            "label_type": data_bundle.metadata.label_type,
        },
        log_train_confusion_matrix=bool(logging_config.get("log_train_confusion_matrix", False)),
        log_val_confusion_matrix=bool(logging_config.get("log_val_confusion_matrix", True)),
        log_val_precision_recall_curve=bool(
            logging_config.get("log_val_precision_recall_curve", True)
        ),
        log_test_figures=bool(logging_config.get("log_test_figures", True)),
        train_sampler=data_bundle.train_sampler,
        is_main_process=True,
        print_fn=print if bool(logging_config.get("print_every_epoch", True)) else None,
    )

    print("")
    print("Training complete.")
    print(f"Output directory: {Path(output_dir).resolve()}")
    if tensorboard_dir:
        print(f"TensorBoard directory: {Path(str(tensorboard_dir)).resolve()}")
    print(f"Best checkpoint: {result.get('best_checkpoint', {}).get('best_path')}")

def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)

    if not isinstance(payload, dict):
        raise TypeError(f"Config must parse to a dictionary, got {type(payload).__name__}")

    return payload

def make_s3_client(
    *,
    profile_name: str | None,
    region_name: str | None,
):
    """
    Build a boto3 S3 client from optional local AWS profile settings.

    If profile_name is None, boto3 uses the normal default credential chain.
    """
    if profile_name:
        session = boto3.Session(
            profile_name=profile_name,
            region_name=region_name,
        )
    else:
        session = boto3.Session(region_name=region_name)

    return session.client("s3")

def set_seed(seed: int) -> None:
    """
    Set common random seeds for reproducible-ish local experiments.

    This does not guarantee perfect determinism across all CUDA operations, but
    it is a good baseline for repeatable small experiments.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_hyperparameter_summary(
    *,
    config: dict[str, Any],
    seed: int,
    num_classes: int,
    split_counts: dict[str, int],
) -> dict[str, Any]:
    return {
        "seed": seed,
        "num_classes": num_classes,
        "split_counts": dict(split_counts),
        "data": dict(config.get("data") or {}),
        "model": dict(config.get("model") or {}),
        "training": dict(config.get("training") or {}),
    }

def require_dict(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a dictionary, got {type(value).__name__}")

    return value

def require_phase_list(value: Any, field_name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list, got {type(value).__name__}")

    if not value:
        raise ValueError(f"{field_name} cannot be empty")

    for idx, item in enumerate(value):
        if not isinstance(item, dict):
            raise TypeError(
                f"{field_name}[{idx}] must be a dictionary, got {type(item).__name__}"
            )

    return value

def require_nonempty_string(value: Any, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} cannot be null")

    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} cannot be empty")

    return text

def require_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int, got {value!r}")

    if value < 1:
        raise ValueError(f"{field_name} must be >= 1, got {value}")

    return value

def require_nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int, got {value!r}")

    if value < 0:
        raise ValueError(f"{field_name} must be >= 0, got {value}")

    return value

if __name__ == "__main__":
    main()