"""
Training entry point for the CVDMS BigEarthNet v2 multi-label classifier.

This script wires together:

    config.yaml
    CVDMS metadata/manifests from S3
    cvdms_training_common DataLoaders
    project-specific ResNet18 model utilities
    project-specific staged fine-tuning loop

Run from the project root:

    python source/train.py --config config.yaml

Metric note
-----------
For this project, mAP means macro-averaged Average Precision. In other words,
`macro_average_precision` is calculated by computing Average Precision (AP)
separately for each class and then averaging those per-class AP scores.

`micro_average_precision` is different: it flattens all class decisions across
all examples before computing AP, so user-facing outputs should call it
micro-AP rather than mAP.

Project 2 thresholding note
---------------------------
The normal/global evaluation threshold is controlled by `training.threshold`.
For Project 2, the staged training workflow can also run a separate
validation-derived per-class threshold evaluation after training:

    logging.evaluate_per_class_thresholds: true

Those per-class thresholds are derived from the validation split using the
best validation checkpoint, then frozen before evaluating the test split. This
is intentionally reported separately from the global-threshold result.
"""

import argparse
import random
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import torch
import torch.nn as nn
import yaml

from cvdms_training_common.dataloaders.multi_label import build_multi_label_data_bundle

from helpers import (
    build_project_image_loader,
    require_dict,
    require_nonempty_string,
    require_nonnegative_int,
    require_phase_list,
    require_positive_float,
    require_positive_int,
    require_probability_threshold,
    require_threshold_sweep_values,
)
from models import build_model
from staged_training import run_staged_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a staged ResNet multi-label classifier from CVDMS dataset artifacts."
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
    loss_config = require_dict(config.get("loss"), "loss")
    aws_config = config.get("aws") or {}

    if not isinstance(aws_config, dict):
        raise TypeError(f"aws must be a dictionary when provided, got {type(aws_config).__name__}")

    seed = int(training_config.get("seed", 42))
    set_seed(seed)

    device = resolve_device(training_config.get("device"))
    metadata_uri = require_nonempty_string(config.get("metadata_uri"), "metadata_uri")

    s3_client = make_s3_client(
        profile_name=aws_config.get("profile_name"),
        region_name=aws_config.get("region_name"),
    )

    image_loader = build_project_image_loader(
        data_config=data_config,
        s3_client=s3_client,
    )

    batch_size = require_positive_int(data_config.get("batch_size"), "data.batch_size")
    image_size = require_positive_int(data_config.get("image_size"), "data.image_size")

    data_bundle = build_multi_label_data_bundle(
        metadata_uri=metadata_uri,
        batch_size=batch_size,
        num_workers=require_nonnegative_int(data_config.get("num_workers", 0), "data.num_workers"),
        image_size=image_size,
        image_loader=image_loader,
        s3_client=s3_client,
        pin_memory=data_config.get("pin_memory"),
        persistent_workers=data_config.get("persistent_workers"),
        prefetch_factor=data_config.get("prefetch_factor"),
        drop_last_train=read_bool(
            data_config.get("drop_last_train", False),
            "data.drop_last_train",
        ),
        distributed=False,
    )

    if data_bundle.metadata.label_type != "multi-label":
        raise ValueError(
            f"Expected metadata.label_type='multi-label', got {data_bundle.metadata.label_type!r}"
        )

    model_name = require_nonempty_string(model_config.get("name"), "model.name")
    model = build_model(
        model_name=model_name,
        num_classes=data_bundle.num_classes,
        pretrained=read_bool(model_config.get("pretrained", True), "model.pretrained"),
    )

    loss_fn, loss_summary = build_loss_fn(
        loss_config=loss_config,
        metadata=data_bundle.metadata,
        device=device,
    )

    phases = require_phase_list(training_config.get("phases"), "training.phases")
    output_dir = require_nonempty_string(logging_config.get("output_dir"), "logging.output_dir")
    tensorboard_dir = logging_config.get("tensorboard_dir")
    threshold = require_probability_threshold(
        training_config.get("threshold", 0.5),
        "training.threshold",
    )

    result = run_staged_training(
        model=model,
        train_loader=data_bundle.train_loader,
        val_loader=data_bundle.val_loader,
        test_loader=data_bundle.test_loader,
        loss_fn=loss_fn,
        cvdms_metadata=data_bundle.metadata,
        model_name=model_name,
        phases=phases,
        output_dir=output_dir,
        tensorboard_dir=tensorboard_dir,
        batch_size=batch_size,
        channels=require_positive_int(data_config.get("channels", 3), "data.channels"),
        image_size=image_size,
        metadata_uri=metadata_uri,
        device=device,
        threshold=threshold,
        best_metric_name=str(training_config.get("best_metric_name", "val_f1_macro")),
        best_metric_mode=str(training_config.get("best_metric_mode", "max")),
        hyperparameters=build_hyperparameter_summary(
            config=config,
            seed=seed,
            num_classes=data_bundle.num_classes,
            split_counts=data_bundle.split_counts,
            loss_summary=loss_summary,
        ),
        extra_checkpoint_metadata={
            "project": "cvdms_bigearthnetv2_multi_label_classifier",
            "dataset_id": data_bundle.metadata.dataset_id,
            "dataset_version": data_bundle.metadata.version,
            "label_type": data_bundle.metadata.label_type,
            "threshold": threshold,
            "loss": loss_summary,
            "metric_notes": metric_notes(),
        },
        log_val_precision_recall_curve=read_bool(
            logging_config.get("log_val_precision_recall_curve", True),
            "logging.log_val_precision_recall_curve",
        ),
        log_test_figures=read_bool(
            logging_config.get("log_test_figures", True),
            "logging.log_test_figures",
        ),
        log_threshold_sweep=read_bool(
            logging_config.get("log_threshold_sweep", True),
            "logging.log_threshold_sweep",
        ),
        threshold_sweep_values=require_threshold_sweep_values(
            logging_config.get("threshold_sweep_values"),
            "logging.threshold_sweep_values",
        ),
        save_best_validation_diagnostics=read_bool(
            logging_config.get("save_best_validation_diagnostics", True),
            "logging.save_best_validation_diagnostics",
        ),
        save_test_diagnostics=read_bool(
            logging_config.get("save_test_diagnostics", True),
            "logging.save_test_diagnostics",
        ),
        evaluate_per_class_thresholds=read_bool(
            logging_config.get("evaluate_per_class_thresholds", True),
            "logging.evaluate_per_class_thresholds",
        ),
        train_sampler=data_bundle.train_sampler,
        is_main_process=True,
        print_fn=print
        if read_bool(logging_config.get("print_every_epoch", True), "logging.print_every_epoch")
        else None,
    )

    print_training_complete_summary(
        result=result,
        output_dir=output_dir,
        tensorboard_dir=tensorboard_dir,
    )


def build_loss_fn(
    *,
    loss_config: dict[str, Any],
    metadata,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    loss_name = require_nonempty_string(
        loss_config.get("name", "bce_with_logits"),
        "loss.name",
    )

    if loss_name != "bce_with_logits":
        raise ValueError(
            f"Unsupported loss.name={loss_name!r}; expected 'bce_with_logits'"
        )

    use_pos_weight = read_bool(
        loss_config.get("use_pos_weight", False),
        "loss.use_pos_weight",
    )
    pos_weight_source = str(loss_config.get("pos_weight_source", "train_class_counts")).strip()
    max_pos_weight = loss_config.get("max_pos_weight")

    if max_pos_weight is not None:
        max_pos_weight = require_positive_float(max_pos_weight, "loss.max_pos_weight")

    pos_weight = None
    pos_weight_values: list[float] | None = None

    if use_pos_weight:
        if pos_weight_source != "train_class_counts":
            raise ValueError(
                "loss.pos_weight_source must be 'train_class_counts' when "
                f"use_pos_weight is true, got {pos_weight_source!r}"
            )

        pos_weight = build_pos_weight_from_metadata(
            metadata=metadata,
            device=device,
            max_pos_weight=max_pos_weight,
        )
        pos_weight_values = [float(value) for value in pos_weight.detach().cpu().tolist()]

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    return loss_fn, {
        "name": loss_name,
        "use_pos_weight": use_pos_weight,
        "pos_weight_source": pos_weight_source if use_pos_weight else None,
        "max_pos_weight": max_pos_weight,
        "pos_weight": pos_weight_values,
    }


def build_pos_weight_from_metadata(
    *,
    metadata,
    device: torch.device,
    max_pos_weight: float | None,
) -> torch.Tensor:
    """
    Build BCEWithLogitsLoss pos_weight from CVDMS train class counts.

    For class c:

        pos_weight[c] = negative_train_count[c] / positive_train_count[c]

    Larger values make mistakes on positive examples of rare classes more
    expensive. This is useful for BigEarthNet v2 because the class distribution
    is intentionally imbalanced.
    """
    raw = getattr(metadata, "raw", None)
    if not isinstance(raw, dict):
        raise TypeError("metadata.raw must be a dictionary to build pos_weight")

    class_counts_by_split = raw.get("class_counts_by_split")
    if not isinstance(class_counts_by_split, dict):
        raise ValueError("metadata.raw['class_counts_by_split'] is required for pos_weight")

    train_class_counts = class_counts_by_split.get("train")
    if not isinstance(train_class_counts, dict):
        raise ValueError(
            "metadata.raw['class_counts_by_split']['train'] must be a dictionary"
        )

    split_counts = raw.get("split_counts")
    if not isinstance(split_counts, dict):
        raise ValueError("metadata.raw['split_counts'] is required for pos_weight")

    total_train = int(split_counts.get("train", 0))
    if total_train < 1:
        raise ValueError(f"metadata train split count must be >= 1, got {total_train}")

    weights: list[float] = []

    for class_name, class_idx in sorted(metadata.class_to_idx.items(), key=lambda item: item[1]):
        positive_count = int(train_class_counts.get(class_name, 0))

        if positive_count < 1:
            raise ValueError(
                f"Cannot build pos_weight because class {class_name!r} has "
                f"{positive_count} positive train examples"
            )

        negative_count = total_train - positive_count

        if negative_count < 0:
            raise ValueError(
                f"Class {class_name!r} has positive_count={positive_count}, which is "
                f"greater than total_train={total_train}"
            )

        weight = negative_count / positive_count

        if max_pos_weight is not None:
            weight = min(weight, max_pos_weight)

        weights.append(float(weight))

    return torch.tensor(weights, dtype=torch.float32, device=device)


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


def resolve_device(value: Any) -> torch.device:
    """
    Resolve training device from config.

    If value is None, choose CUDA when available, otherwise CPU.
    """
    if value is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(str(value))

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requested CUDA, but torch.cuda.is_available() is false")

    return device


def build_hyperparameter_summary(
    *,
    config: dict[str, Any],
    seed: int,
    num_classes: int,
    split_counts: dict[str, int],
    loss_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "seed": seed,
        "num_classes": num_classes,
        "split_counts": dict(split_counts),
        "data": dict(config.get("data") or {}),
        "model": dict(config.get("model") or {}),
        "loss": dict(loss_summary),
        "training": dict(config.get("training") or {}),
        "logging": dict(config.get("logging") or {}),
        "metric_notes": metric_notes(),
    }


def metric_notes() -> dict[str, str]:
    return {
        "map": (
            "mAP means macro-averaged Average Precision: compute AP separately "
            "for each class, then average those per-class AP scores."
        ),
        "micro_ap": (
            "micro-AP flattens all example/class decisions before computing AP. "
            "It is useful, but it is not the same as mAP."
        ),
        "thresholded_metrics": (
            "Precision, recall, F1, hamming accuracy, hamming loss, and subset "
            "accuracy use the configured probability threshold to convert sigmoid "
            "probabilities into binary predictions."
        ),
        "ap_metrics": (
            "AP, mAP, and micro-AP are threshold-free ranking diagnostics based "
            "on the model's predicted probabilities."
        ),
        "global_threshold": (
            "The global-threshold evaluation uses the same scalar threshold for "
            "every class, usually training.threshold = 0.5."
        ),
        "validation_derived_per_class_thresholds": (
            "Project 2 optionally derives one threshold per class from the validation "
            "split using the best validation checkpoint. Those thresholds are frozen "
            "before test evaluation and reported separately from the global-threshold result."
        ),
    }


def read_bool(value: Any, field_name: str) -> bool:
    """
    Read a YAML boolean config value.

    This intentionally rejects strings such as "false" because bool("false")
    evaluates to True in Python. Use real YAML booleans instead:

        good:  false
        bad:   "false"
    """
    if isinstance(value, bool):
        return value

    raise TypeError(
        f"{field_name} must be a YAML boolean true/false, got {value!r} "
        f"({type(value).__name__})"
    )


def print_training_complete_summary(
    *,
    result: dict[str, Any],
    output_dir: str | Path,
    tensorboard_dir: str | Path | None,
) -> None:
    best_global = (
        result.get("test_metrics_best_checkpoint_global_threshold")
        or result.get("test_metrics_best_checkpoint")
        or result.get("test_metrics")
        or {}
    )
    best_per_class = result.get("test_metrics_best_checkpoint_per_class_thresholds") or {}
    final_global = (
        result.get("test_metrics_final_model_global_threshold")
        or result.get("test_metrics_final_model")
        or {}
    )
    best_checkpoint = result.get("best_checkpoint") or {}
    threshold_strategies = result.get("threshold_strategies") or {}

    print("")
    print("Training complete.")
    print(f"Output directory: {Path(output_dir).resolve()}")
    if tensorboard_dir:
        print(f"TensorBoard directory: {Path(str(tensorboard_dir)).resolve()}")
    print(f"Best checkpoint: {best_checkpoint.get('best_path')}")

    if best_global:
        print(
            "Primary test result from best checkpoint, global threshold: "
            f"loss={_format_optional_metric(best_global.get('loss'))}, "
            f"f1_macro={_format_optional_metric(best_global.get('f1_macro'))}, "
            f"mAP={_format_optional_metric(best_global.get('map') or best_global.get('macro_average_precision'))}, "
            f"micro-AP={_format_optional_metric(best_global.get('micro_ap') or best_global.get('micro_average_precision'))}"
        )

    if best_per_class:
        strategy = threshold_strategies.get("per_class_validation_f1") or {}
        threshold_count = len(strategy.get("thresholds_by_class") or {})
        print(
            "Evaluation-only result from best checkpoint, validation-derived per-class thresholds: "
            f"loss={_format_optional_metric(best_per_class.get('loss'))}, "
            f"f1_macro={_format_optional_metric(best_per_class.get('f1_macro'))}, "
            f"precision_macro={_format_optional_metric(best_per_class.get('precision_macro'))}, "
            f"recall_macro={_format_optional_metric(best_per_class.get('recall_macro'))}, "
            f"threshold_count={threshold_count}"
        )

    if final_global:
        print(
            "Reference test result from final model state, global threshold: "
            f"loss={_format_optional_metric(final_global.get('loss'))}, "
            f"f1_macro={_format_optional_metric(final_global.get('f1_macro'))}, "
            f"mAP={_format_optional_metric(final_global.get('map') or final_global.get('macro_average_precision'))}"
        )


def _format_optional_metric(value: Any) -> str:
    if value is None:
        return "n/a"

    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
