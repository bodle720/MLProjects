"""
PyTorch Dataset support for CVDMS multi-label classification manifests.

Expected CVDMS multi-label manifest row shape:

    {
        "image_id": "...",
        "source_ref": "s3://...",
        "split": "train" | "val" | "test",
        "label_type": "multi-label",
        "labels": ["forest", "water", "..."]
    }

The string labels are mapped to a multi-hot float target vector using the
version-specific class_to_idx map from CVDMS metadata.json.
"""

from dataclasses import dataclass
from typing import Any, Callable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from cvdms_training_common.image_loading import ImageLoader, S3ImageLoader
from cvdms_training_common.manifests import (
    CvdmsManifestRow,
    SplitName,
    load_split_manifest_rows,
    validate_unique_image_ids,
)
from cvdms_training_common.metadata import CvdmsDatasetMetadata

@dataclass(frozen=True)
class MultiLabelSample:
    """
    Metadata for one multi-label sample.

    The Dataset returns tensors for training, but this sample object is useful
    for debugging, inspection, and future explainability/reporting.
    """

    image_id: str
    source_ref: str
    split: SplitName
    label_names: tuple[str, ...]
    label_indices: tuple[int, ...]
    raw: dict[str, Any]

class CvdmsMultiLabelDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    PyTorch Dataset for CVDMS multi-label classification.

    Args:
        rows:
            Normalized CVDMS manifest rows for one split.
        metadata:
            Version-scoped CVDMS metadata. Must have label_type='multi-label'.
        transform:
            Optional image transform. If omitted, images are converted with
            torchvision.transforms.ToTensor().
        image_loader:
            Optional image loader. Defaults to S3ImageLoader.
        validate_unique_ids:
            If True, verify there are no duplicate image_id values in this split.
        allow_empty_labels:
            If True, allow samples with zero positive labels. For the current
            BigEarthNet v2 workflow this should normally remain False.
    """

    def __init__(
        self,
        *,
        rows: list[CvdmsManifestRow] | tuple[CvdmsManifestRow, ...],
        metadata: CvdmsDatasetMetadata,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        image_loader: ImageLoader | None = None,
        validate_unique_ids: bool = True,
        allow_empty_labels: bool = False,
    ) -> None:
        if metadata.label_type != "multi-label":
            raise ValueError(
                "CvdmsMultiLabelDataset requires metadata.label_type='multi-label', "
                f"got {metadata.label_type!r}"
            )

        if not rows:
            raise ValueError("CvdmsMultiLabelDataset requires at least one manifest row")

        self.metadata = metadata
        self.rows = tuple(rows)
        self.transform = transform or transforms.ToTensor()
        self.image_loader = image_loader or S3ImageLoader()
        self.allow_empty_labels = allow_empty_labels

        if validate_unique_ids:
            validate_unique_image_ids(self.rows, context="multi-label dataset rows")

        self.samples = tuple(
            _build_multi_label_sample(
                row=row,
                metadata=metadata,
                row_idx=i,
                allow_empty_labels=allow_empty_labels,
            )
            for i, row in enumerate(self.rows)
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]

        image = self.image_loader(sample.source_ref)
        image_tensor = self.transform(image)
        target = _build_multi_hot_target(
            label_indices=sample.label_indices,
            num_classes=self.num_classes,
        )

        return image_tensor, target

    def get_sample_metadata(self, idx: int) -> MultiLabelSample:
        """
        Return non-tensor sample metadata for debugging/inspection.
        """
        return self.samples[idx]

    @property
    def class_to_idx(self) -> dict[str, int]:
        return self.metadata.class_to_idx

    @property
    def idx_to_class(self) -> dict[int, str]:
        return self.metadata.idx_to_class

    @property
    def num_classes(self) -> int:
        return self.metadata.num_classes

def load_multi_label_dataset_for_split(
    *,
    metadata: CvdmsDatasetMetadata,
    split: SplitName,
    transform: Callable[[Image.Image], torch.Tensor] | None = None,
    image_loader: ImageLoader | None = None,
    s3_client=None,
    validate_unique_ids: bool = True,
    allow_empty_labels: bool = False,
) -> CvdmsMultiLabelDataset:
    """
    Load one CVDMS split manifest and return a multi-label Dataset.
    """
    rows = load_split_manifest_rows(
        metadata,
        split,
        s3_client=s3_client,
    )

    return CvdmsMultiLabelDataset(
        rows=rows,
        metadata=metadata,
        transform=transform,
        image_loader=image_loader,
        validate_unique_ids=validate_unique_ids,
        allow_empty_labels=allow_empty_labels,
    )

def build_default_multi_label_transforms(
    *,
    image_size: int = 224,
    train: bool,
    normalize: bool = True,
) -> transforms.Compose:
    """
    Build simple default transforms for multi-label image classification.

    These are intentionally conservative defaults. Individual training projects
    can and should override them when they need model-specific preprocessing.

    Args:
        image_size:
            Output image size. Images are resized to image_size x image_size.
        train:
            If True, include light training augmentation.
        normalize:
            If True, use ImageNet normalization values, which are appropriate
            for common pretrained torchvision models.
    """
    transform_list: list[Callable] = [
        transforms.Resize((image_size, image_size)),
    ]

    if train:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    transform_list.append(transforms.ToTensor())

    if normalize:
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )

    return transforms.Compose(transform_list)

def _build_multi_label_sample(
    *,
    row: CvdmsManifestRow,
    metadata: CvdmsDatasetMetadata,
    row_idx: int,
    allow_empty_labels: bool,
) -> MultiLabelSample:
    if row.label_type != "multi-label":
        raise ValueError(
            f"multi-label row {row_idx} has label_type={row.label_type!r}; "
            "expected 'multi-label'"
        )

    raw_labels = row.raw.get("labels")
    label_names = _require_label_tuple(
        raw_labels,
        f"row[{row_idx}].labels",
        allow_empty_labels=allow_empty_labels,
    )

    missing_labels = [
        label_name
        for label_name in label_names
        if label_name not in metadata.class_to_idx
    ]

    if missing_labels:
        raise ValueError(
            f"row[{row_idx}] labels contain values not present in "
            f"metadata.class_to_idx: missing={missing_labels!r}; "
            f"valid_keys={sorted(metadata.class_to_idx)}"
        )

    label_indices = tuple(metadata.class_to_idx[label_name] for label_name in label_names)

    return MultiLabelSample(
        image_id=row.image_id,
        source_ref=row.source_ref,
        split=row.split,
        label_names=label_names,
        label_indices=label_indices,
        raw=dict(row.raw),
    )

def _build_multi_hot_target(
    *,
    label_indices: tuple[int, ...],
    num_classes: int,
) -> torch.Tensor:
    target = torch.zeros(num_classes, dtype=torch.float32)

    if label_indices:
        target[list(label_indices)] = 1.0

    return target

def _require_label_tuple(
    value: Any,
    field_name: str,
    *,
    allow_empty_labels: bool,
) -> tuple[str, ...]:
    if value is None:
        raise ValueError(f"{field_name} cannot be null")

    if not isinstance(value, list | tuple):
        raise TypeError(
            f"{field_name} must be a list or tuple of label strings, "
            f"got {type(value).__name__}"
        )

    if not value and not allow_empty_labels:
        raise ValueError(f"{field_name} cannot be empty")

    labels = tuple(
        _require_nonempty_string(label, f"{field_name}[{idx}]")
        for idx, label in enumerate(value)
    )

    duplicates = _find_duplicates(labels)

    if duplicates:
        raise ValueError(f"{field_name} contains duplicate labels: {duplicates!r}")

    return labels

def _find_duplicates(values: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []

    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)

    return duplicates

def _require_nonempty_string(value: Any, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} cannot be null")

    text = str(value).strip()

    if not text:
        raise ValueError(f"{field_name} cannot be empty")

    return text