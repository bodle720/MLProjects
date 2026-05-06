"""
Dataset inspection smoke test for the CVDMS BigEarthNet v2 multi-label classifier.

Run from the project root:

    python source/inspect_dataset.py --config config.yaml

This script does not train a model. It only verifies that the CVDMS metadata,
manifests, image loading, transforms, multi-hot targets, and DataLoaders are
wired correctly.
"""

import argparse
from pathlib import Path
from typing import Any

import boto3
import torch
import yaml

from cvdms_training_common.dataloaders.multi_label import build_multi_label_data_bundle
from helpers import (
    build_project_image_loader,
    require_dict,
    require_nonempty_string,
    require_nonnegative_int,
    require_positive_int,
)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a CVDMS multi-label dataset before training."
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

    image_loader = build_project_image_loader(
        data_config=data_config,
        s3_client=s3_client,
    )

    bundle = build_multi_label_data_bundle(
        metadata_uri=metadata_uri,
        batch_size=require_positive_int(data_config.get("batch_size"), "data.batch_size"),
        num_workers=require_nonnegative_int(data_config.get("num_workers", 0), "data.num_workers"),
        image_size=require_positive_int(data_config.get("image_size"), "data.image_size"),
        image_loader=image_loader,
        s3_client=s3_client,
        pin_memory=data_config.get("pin_memory"),
        persistent_workers=data_config.get("persistent_workers"),
        prefetch_factor=data_config.get("prefetch_factor"),
        drop_last_train=bool(data_config.get("drop_last_train", False)),
        distributed=False,
    )

    if bundle.metadata.label_type != "multi-label":
        raise ValueError(
            f"Expected metadata.label_type='multi-label', got {bundle.metadata.label_type!r}"
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
    print(f"image_id:       {train_sample.image_id}")
    print(f"source_ref:     {train_sample.source_ref}")
    print(f"split:          {train_sample.split}")
    print(f"label_names:    {train_sample.label_names}")
    print(f"label_indices:  {train_sample.label_indices}")

    print("")
    print("Train batch inspection")
    print("=" * 80)

    if args.num_batches < 1:
        raise ValueError("--num-batches must be >= 1")

    for batch_idx, (images, targets) in enumerate(bundle.train_loader, start=1):
        validate_batch_targets(
            targets=targets,
            num_classes=bundle.metadata.num_classes,
            batch_idx=batch_idx,
        )

        cardinalities = targets.sum(dim=1)
        positive_counts = targets.sum(dim=0)
        decoded_labels = decode_multihot_batch(
            targets=targets,
            idx_to_class=bundle.metadata.idx_to_class,
        )

        print(f"batch {batch_idx}")
        print(f"  images.shape:        {tuple(images.shape)}")
        print(f"  images.dtype:        {images.dtype}")
        print(f"  images.min:          {float(images.min().item()):.4f}")
        print(f"  images.max:          {float(images.max().item()):.4f}")
        print(f"  targets.shape:       {tuple(targets.shape)}")
        print(f"  targets.dtype:       {targets.dtype}")
        print(f"  target values:       {sorted(float(x) for x in targets.unique().tolist())}")
        print(f"  labels per image:    {cardinalities.tolist()}")
        print(f"  min labels/image:    {float(cardinalities.min().item()):.0f}")
        print(f"  max labels/image:    {float(cardinalities.max().item()):.0f}")
        print(f"  mean labels/image:   {float(cardinalities.float().mean().item()):.4f}")
        print(f"  class positives:     {positive_counts.to(torch.int64).tolist()}")
        print(f"  decoded labels:      {decoded_labels}")

        if batch_idx >= args.num_batches:
            break

    print("")
    print("Dataset inspection completed successfully.")

def decode_multihot_batch(
    *,
    targets: torch.Tensor,
    idx_to_class: dict[int, str],
) -> list[list[str]]:
    decoded: list[list[str]] = []

    for row in targets:
        class_indices = torch.nonzero(row > 0.5, as_tuple=False).flatten().tolist()
        decoded.append([idx_to_class[int(class_idx)] for class_idx in class_indices])

    return decoded

def validate_batch_targets(
    *,
    targets: torch.Tensor,
    num_classes: int,
    batch_idx: int,
) -> None:
    if targets.ndim != 2:
        raise ValueError(
            f"Batch {batch_idx} targets must have shape [batch_size, num_classes], "
            f"got {tuple(targets.shape)}"
        )

    if targets.shape[1] != num_classes:
        raise ValueError(
            f"Batch {batch_idx} targets second dimension must equal num_classes={num_classes}, "
            f"got shape {tuple(targets.shape)}"
        )

    if not targets.is_floating_point():
        raise TypeError(
            f"Batch {batch_idx} targets must be floating point for BCEWithLogitsLoss, "
            f"got dtype {targets.dtype}"
        )

    if not torch.all((targets == 0) | (targets == 1)):
        unique_values = sorted(float(x) for x in targets.unique().tolist())
        raise ValueError(
            f"Batch {batch_idx} targets must be multi-hot 0/1 values, "
            f"got unique values {unique_values}"
        )

    label_counts = targets.sum(dim=1)
    if torch.any(label_counts < 1):
        raise ValueError(
            f"Batch {batch_idx} contains at least one sample with no positive labels."
        )

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

if __name__ == "__main__":
    main()