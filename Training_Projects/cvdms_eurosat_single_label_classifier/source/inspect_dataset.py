"""
Dataset inspection smoke test for the CVDMS EuroSAT single-label classifier.

Run from the project root:

    python source/inspect_dataset.py --config config.yaml

This script does not train a model. It only verifies that the CVDMS metadata,
manifests, S3 image loading, transforms, and DataLoaders are wired correctly.
"""

import argparse
from pathlib import Path
from typing import Any

import boto3
import yaml

from cvdms_training_common.dataloaders.single_label import build_single_label_data_bundle

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a CVDMS single-label dataset before training."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to project config YAML file.",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=1,
        help="Number of train batches to inspect.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    data_config = require_dict(config.get("data"), "data")
    aws_config = config.get("aws") or {}
    metadata_uri = require_nonempty_string(config.get("metadata_uri"), "metadata_uri")

    s3_client = make_s3_client(
        profile_name=aws_config.get("profile_name"),
        region_name=aws_config.get("region_name"),
    )

    bundle = build_single_label_data_bundle(
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

    print("")
    print("CVDMS dataset loaded")
    print("=" * 80)
    print(f"dataset_id:       {bundle.metadata.dataset_id}")
    print(f"version:          {bundle.metadata.version}")
    print(f"label_type:       {bundle.metadata.label_type}")
    print(f"num_classes:      {bundle.metadata.num_classes}")
    print(f"effective_classes:{bundle.metadata.effective_classes}")
    print(f"split_counts:     {bundle.split_counts}")

    print("")
    print("Class maps")
    print("=" * 80)
    print(f"class_to_idx: {bundle.metadata.class_to_idx}")
    print(f"idx_to_class: {bundle.metadata.idx_to_class}")

    print("")
    print("Dataset lengths")
    print("=" * 80)
    print(f"train_dataset: {len(bundle.train_dataset)}")
    print(f"val_dataset:   {len(bundle.val_dataset)}")
    print(f"test_dataset:  {len(bundle.test_dataset)}")

    print("")
    print("Sample metadata")
    print("=" * 80)
    train_sample = bundle.train_dataset.get_sample_metadata(0)
    print(f"image_id:   {train_sample.image_id}")
    print(f"source_ref: {train_sample.source_ref}")
    print(f"split:      {train_sample.split}")
    print(f"label_name: {train_sample.label_name}")
    print(f"label_idx:  {train_sample.label_idx}")

    print("")
    print("Train batch inspection")
    print("=" * 80)

    if args.num_batches < 1:
        raise ValueError("--num-batches must be >= 1")

    for batch_idx, (images, targets) in enumerate(bundle.train_loader, start=1):
        print(f"batch {batch_idx}")
        print(f"  images.shape: {tuple(images.shape)}")
        print(f"  images.dtype: {images.dtype}")
        print(f"  images.min:   {float(images.min().item()):.4f}")
        print(f"  images.max:   {float(images.max().item()):.4f}")
        print(f"  targets.shape:{tuple(targets.shape)}")
        print(f"  targets.dtype:{targets.dtype}")
        print(f"  targets:      {targets.tolist()}")

        decoded_labels = [
            bundle.metadata.idx_to_class[int(target)]
            for target in targets.tolist()
        ]
        print(f"  labels:       {decoded_labels}")

        if batch_idx >= args.num_batches:
            break

    print("")
    print("Dataset inspection completed successfully.")

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
    if profile_name:
        session = boto3.Session(
            profile_name=profile_name,
            region_name=region_name,
        )
    else:
        session = boto3.Session(region_name=region_name)

    return session.client("s3")

def require_dict(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a dictionary, got {type(value).__name__}")

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