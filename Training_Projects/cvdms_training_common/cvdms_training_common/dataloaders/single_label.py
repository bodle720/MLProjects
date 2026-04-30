"""
DataLoader builders for CVDMS training projects.

This module provides convenience helpers for turning CVDMS dataset-version
metadata into PyTorch DataLoaders.

The functions here are intentionally thin. They should handle common CVDMS
loading patterns, while individual training projects remain free to define their
own transforms, samplers, collate functions, and training loops.
"""

from dataclasses import dataclass
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from cvdms_training_common.datasets.single_label import (
    CvdmsSingleLabelDataset,
    build_default_single_label_transforms,
    load_single_label_dataset_for_split,
)
from cvdms_training_common.image_loading import ImageLoader, S3ImageLoader
from cvdms_training_common.metadata import CvdmsDatasetMetadata, load_cvdms_metadata
from cvdms_training_common.dataloaders.common_utils import build_dataloader, validate_positive_int, \
                                                           validate_nonnegative_int

@dataclass(frozen=True)
class SingleLabelDataBundle:
    """
    Container for single-label datasets and DataLoaders.

    `train_sampler` is included because distributed training requires calling
    `train_sampler.set_epoch(epoch)` once per epoch.
    """

    metadata: CvdmsDatasetMetadata

    train_dataset: CvdmsSingleLabelDataset
    val_dataset: CvdmsSingleLabelDataset
    test_dataset: CvdmsSingleLabelDataset

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader

    train_sampler: DistributedSampler | None = None
    val_sampler: DistributedSampler | None = None
    test_sampler: DistributedSampler | None = None

    @property
    def num_classes(self) -> int:
        return self.metadata.num_classes

    @property
    def split_counts(self) -> dict[str, int]:
        return {
            "train": len(self.train_dataset),
            "val": len(self.val_dataset),
            "test": len(self.test_dataset),
        }

def build_single_label_data_bundle(
    *,
    metadata_uri: str,
    batch_size: int,
    num_workers: int = 0,
    image_size: int = 224,
    train_transform: Callable[[Image.Image], torch.Tensor] | None = None,
    eval_transform: Callable[[Image.Image], torch.Tensor] | None = None,
    image_loader: ImageLoader | None = None,
    s3_client=None,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
    drop_last_train: bool = False,
    distributed: bool = False,
    distributed_shuffle_train: bool = True,
    validate_unique_ids: bool = True,
) -> SingleLabelDataBundle:
    """
    Build train/val/test datasets and DataLoaders for single-label classification.

    Args:
        metadata_uri:
            S3 URI to CVDMS metadata/metadata.json.
        batch_size:
            DataLoader batch size.
        num_workers:
            DataLoader worker count. Start with 0 for local/S3 debugging.
        image_size:
            Used only when default transforms are built.
        train_transform:
            Optional project-specific train transform.
        eval_transform:
            Optional project-specific validation/test transform.
        image_loader:
            Optional image loader. Defaults to S3ImageLoader.
        s3_client:
            Optional boto3 S3 client used to read metadata/manifests and, if
            image_loader is not provided, images.
        pin_memory:
            If None, defaults to torch.cuda.is_available().
        persistent_workers:
            If None, enabled only when num_workers > 0.
        prefetch_factor:
            DataLoader prefetch factor. Only passed when num_workers > 0.
        drop_last_train:
            Whether to drop the last partial training batch.
        distributed:
            If True, use DistributedSampler for train/val/test.
        distributed_shuffle_train:
            Shuffle training data inside DistributedSampler.
        validate_unique_ids:
            Validate no duplicate image_id values within each split.

    Returns:
        SingleLabelDataBundle.
    """
    validate_positive_int(batch_size, "batch_size")
    validate_nonnegative_int(num_workers, "num_workers")
    validate_positive_int(image_size, "image_size")

    metadata = load_cvdms_metadata(
        metadata_uri,
        s3_client=s3_client,
        min_classes=2,
    )

    if metadata.label_type != "single-label":
        raise ValueError(
            "build_single_label_data_bundle requires metadata.label_type='single-label', "
            f"got {metadata.label_type!r}"
        )

    resolved_train_transform = train_transform or build_default_single_label_transforms(
        image_size=image_size,
        train=True,
    )
    resolved_eval_transform = eval_transform or build_default_single_label_transforms(
        image_size=image_size,
        train=False,
    )

    resolved_image_loader = image_loader or S3ImageLoader(s3_client=s3_client)

    train_dataset = load_single_label_dataset_for_split(
        metadata=metadata,
        split="train",
        transform=resolved_train_transform,
        image_loader=resolved_image_loader,
        s3_client=s3_client,
        validate_unique_ids=validate_unique_ids,
    )
    val_dataset = load_single_label_dataset_for_split(
        metadata=metadata,
        split="val",
        transform=resolved_eval_transform,
        image_loader=resolved_image_loader,
        s3_client=s3_client,
        validate_unique_ids=validate_unique_ids,
    )
    test_dataset = load_single_label_dataset_for_split(
        metadata=metadata,
        split="test",
        transform=resolved_eval_transform,
        image_loader=resolved_image_loader,
        s3_client=s3_client,
        validate_unique_ids=validate_unique_ids,
    )

    train_sampler = None
    val_sampler = None
    test_sampler = None

    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            shuffle=distributed_shuffle_train,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            shuffle=False,
        )
        test_sampler = DistributedSampler(
            test_dataset,
            shuffle=False,
        )

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    train_loader = build_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(not distributed),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=drop_last_train,
    )

    val_loader = build_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=False,
    )

    test_loader = build_dataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=False,
    )

    return SingleLabelDataBundle(
        metadata=metadata,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        test_sampler=test_sampler,
    )