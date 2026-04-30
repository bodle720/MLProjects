"""
PyTorch Dataset support for CVDMS single-label classification manifests.

Expected CVDMS single-label manifest row shape:

    {
        "image_id": "...",
        "source_ref": "s3://...",
        "split": "train" | "val" | "test",
        "label_type": "single-label",
        "label": "forest"
    }

The string label is mapped to an integer class index using the version-specific
class_to_idx map from CVDMS metadata.json.
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
class SingleLabelSample:
    """
    Metadata for one single-label sample.

    The Dataset returns tensors for training, but this sample object is useful
    for debugging, inspection, and future explainability/reporting.
    """

    image_id: str
    source_ref: str
    split: SplitName
    label_name: str
    label_idx: int
    raw: dict[str, Any]

class CvdmsSingleLabelDataset(Dataset[tuple[torch.Tensor, int]]):
    """
    PyTorch Dataset for CVDMS single-label classification.

    Args:
        rows:
            Normalized CVDMS manifest rows for one split.
        metadata:
            Version-scoped CVDMS metadata. Must have label_type='single-label'.
        transform:
            Optional image transform. If omitted, images are converted with
            torchvision.transforms.ToTensor().
        image_loader:
            Optional image loader. Defaults to S3ImageLoader.
        validate_unique_ids:
            If True, verify there are no duplicate image_id values in this split.
    """

    def __init__(
        self,
        *,
        rows: list[CvdmsManifestRow] | tuple[CvdmsManifestRow, ...],
        metadata: CvdmsDatasetMetadata,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        image_loader: ImageLoader | None = None,
        validate_unique_ids: bool = True,
    ) -> None:
        if metadata.label_type != "single-label":
            raise ValueError(
                "CvdmsSingleLabelDataset requires metadata.label_type='single-label', "
                f"got {metadata.label_type!r}"
            )

        if not rows:
            raise ValueError("CvdmsSingleLabelDataset requires at least one manifest row")

        self.metadata = metadata
        self.rows = tuple(rows)
        self.transform = transform or transforms.ToTensor()
        self.image_loader = image_loader or S3ImageLoader()

        if validate_unique_ids:
            validate_unique_image_ids(self.rows, context="single-label dataset rows")

        self.samples = tuple(
            _build_single_label_sample(row=row, metadata=metadata, row_idx=i)
            for i, row in enumerate(self.rows)
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[idx]

        image = self.image_loader(sample.source_ref)
        image_tensor = self.transform(image)

        return image_tensor, sample.label_idx

    def get_sample_metadata(self, idx: int) -> SingleLabelSample:
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

def load_single_label_dataset_for_split(
    *,
    metadata: CvdmsDatasetMetadata,
    split: SplitName,
    transform: Callable[[Image.Image], torch.Tensor] | None = None,
    image_loader: ImageLoader | None = None,
    s3_client=None,
    validate_unique_ids: bool = True,
) -> CvdmsSingleLabelDataset:
    """
    Load one CVDMS split manifest and return a single-label Dataset.
    """
    rows = load_split_manifest_rows(
        metadata,
        split,
        s3_client=s3_client,
    )

    return CvdmsSingleLabelDataset(
        rows=rows,
        metadata=metadata,
        transform=transform,
        image_loader=image_loader,
        validate_unique_ids=validate_unique_ids,
    )

def build_default_single_label_transforms(
    *,
    image_size: int = 224,
    train: bool,
    normalize: bool = True,
) -> transforms.Compose:
    """
    Build simple default transforms for image classification.
    These are RGB-only classification transforms and used in normalization for ImageNet data.

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

def _build_single_label_sample(
    *,
    row: CvdmsManifestRow,
    metadata: CvdmsDatasetMetadata,
    row_idx: int,
) -> SingleLabelSample:
    if row.label_type != "single-label":
        raise ValueError(
            f"single-label row {row_idx} has label_type={row.label_type!r}; "
            "expected 'single-label'"
        )

    raw_label = row.raw.get("label")
    label_name = _require_nonempty_string(raw_label, f"row[{row_idx}].label")

    if label_name not in metadata.class_to_idx:
        raise ValueError(
            f"row[{row_idx}] label={label_name!r} is not present in "
            f"metadata.class_to_idx keys={sorted(metadata.class_to_idx)}"
        )

    label_idx = metadata.class_to_idx[label_name]

    return SingleLabelSample(
        image_id=row.image_id,
        source_ref=row.source_ref,
        split=row.split,
        label_name=label_name,
        label_idx=label_idx,
        raw=dict(row.raw),
    )

def _require_nonempty_string(value: Any, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} cannot be null")

    text = str(value).strip()

    if not text:
        raise ValueError(f"{field_name} cannot be empty")

    return text