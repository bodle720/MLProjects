"""
CVDMS manifest helpers.

This module loads and validates split JSONL manifests produced by CVDMS dataset
versions.

Expected CVDMS manifest artifact shape:

    manifests/train.jsonl
    manifests/val.jsonl
    manifests/test.jsonl

For single-label classification, each row is expected to include:

    image_id
    source_ref
    split
    label_type
    label

Task-specific Dataset classes can perform stricter validation later, but this
module handles shared manifest concerns such as split validation, S3 row loading,
and basic row normalization.
"""

from dataclasses import dataclass
from typing import Any, Iterable, Literal

from cvdms_training_common.metadata import CvdmsDatasetMetadata
from cvdms_training_common.s3_io import read_s3_jsonl

SplitName = Literal["train", "val", "test"]
_VALID_SPLITS: tuple[SplitName, ...] = ("train", "val", "test")

_VALID_LABEL_TYPES = {
    "single-label",
    "multi-label",
    "object-detection",
    "semantic-segmentation",
    "instance-segmentation",
}

@dataclass(frozen=True)
class CvdmsManifestRow:
    """
    Minimal normalized representation of a CVDMS manifest row.

    This class stores common fields shared across all label types. The original
    row is kept in `raw` so task-specific dataset classes can access fields such
    as label, labels, bbox_annotation_ids, semantic_mask_ids, or
    instance_annotation_ids.
    """

    image_id: str
    source_ref: str
    split: SplitName
    label_type: str
    raw: dict[str, Any]

@dataclass(frozen=True)
class CvdmsManifestBundle:
    """
    Train/val/test manifest rows loaded for one CVDMS dataset version.
    """

    train: tuple[CvdmsManifestRow, ...]
    val: tuple[CvdmsManifestRow, ...]
    test: tuple[CvdmsManifestRow, ...]

    def rows_for_split(self, split: SplitName) -> tuple[CvdmsManifestRow, ...]:
        if split == "train":
            return self.train
        if split == "val":
            return self.val
        if split == "test":
            return self.test
        raise ValueError(f"Invalid split={split!r}; expected one of {_VALID_SPLITS}")

    @property
    def total_count(self) -> int:
        return len(self.train) + len(self.val) + len(self.test)

    @property
    def split_counts(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "val": len(self.val),
            "test": len(self.test),
        }

def get_manifest_uri(metadata: CvdmsDatasetMetadata, split: SplitName) -> str:
    """
    Return a split manifest URI from CVDMS metadata.

    This is a convenience wrapper around CvdmsDatasetMetadata.get_manifest_uri().
    """
    return metadata.get_manifest_uri(split)

def load_manifest_rows(
    manifest_uri: str,
    *,
    expected_split: SplitName | None = None,
    expected_label_type: str | None = None,
    s3_client=None,
) -> list[CvdmsManifestRow]:
    """
    Load and minimally validate a CVDMS JSONL manifest.

    Args:
        manifest_uri:
            S3 URI to a CVDMS split manifest.
        expected_split:
            Optional split that every row must match.
        expected_label_type:
            Optional label type that every row must match.
        s3_client:
            Optional boto3 S3 client.

    Returns:
        List of normalized CvdmsManifestRow objects.
    """
    if expected_split is not None and expected_split not in _VALID_SPLITS:
        raise ValueError(
            f"Invalid expected_split={expected_split!r}; expected one of {_VALID_SPLITS}"
        )

    rows = read_s3_jsonl(manifest_uri, s3_client=s3_client, strict=True)
    normalized_rows: list[CvdmsManifestRow] = []

    for row_idx, row in enumerate(rows):
        normalized = parse_manifest_row(
            row,
            row_idx=row_idx,
            manifest_uri=manifest_uri,
            expected_split=expected_split,
            expected_label_type=expected_label_type,
        )
        normalized_rows.append(normalized)

    if not normalized_rows:
        raise ValueError(f"CVDMS manifest contains zero rows: {manifest_uri}")

    return normalized_rows

def load_split_manifest_rows(
    metadata: CvdmsDatasetMetadata,
    split: SplitName,
    *,
    s3_client=None,
) -> list[CvdmsManifestRow]:
    """
    Load one split manifest using URIs from metadata.json.
    """
    manifest_uri = metadata.get_manifest_uri(split)

    return load_manifest_rows(
        manifest_uri,
        expected_split=split,
        expected_label_type=metadata.label_type,
        s3_client=s3_client,
    )

def load_manifest_bundle(
    metadata: CvdmsDatasetMetadata,
    *,
    s3_client=None,
) -> CvdmsManifestBundle:
    """
    Load train, val, and test manifests for a CVDMS dataset version.
    """
    train_rows = load_split_manifest_rows(metadata, "train", s3_client=s3_client)
    val_rows = load_split_manifest_rows(metadata, "val", s3_client=s3_client)
    test_rows = load_split_manifest_rows(metadata, "test", s3_client=s3_client)

    return CvdmsManifestBundle(
        train=tuple(train_rows),
        val=tuple(val_rows),
        test=tuple(test_rows),
    )

def parse_manifest_row(
    row: dict[str, Any],
    *,
    row_idx: int | None = None,
    manifest_uri: str | None = None,
    expected_split: SplitName | None = None,
    expected_label_type: str | None = None,
) -> CvdmsManifestRow:
    """
    Parse and minimally validate one CVDMS manifest row.

    This validates fields common to all label types. Task-specific validation
    should happen in the relevant Dataset class.
    """
    context = _format_row_context(row_idx=row_idx, manifest_uri=manifest_uri)

    if not isinstance(row, dict):
        raise TypeError(f"{context} expected dict row, got {type(row).__name__}")

    image_id = _require_nonempty_string(row.get("image_id"), f"{context}.image_id")
    source_ref = _require_s3_uri(row.get("source_ref"), f"{context}.source_ref")
    split = _require_valid_split(row.get("split"), f"{context}.split")
    label_type = _require_valid_label_type(row.get("label_type"), f"{context}.label_type")

    if expected_split is not None and split != expected_split:
        raise ValueError(
            f"{context} has split={split!r}, expected {expected_split!r}"
        )

    if expected_label_type is not None and label_type != expected_label_type:
        raise ValueError(
            f"{context} has label_type={label_type!r}, expected {expected_label_type!r}"
        )

    return CvdmsManifestRow(
        image_id=image_id,
        source_ref=source_ref,
        split=split,
        label_type=label_type,
        raw=dict(row),
    )

def validate_unique_image_ids(
    rows: Iterable[CvdmsManifestRow],
    *,
    context: str = "manifest rows",
) -> None:
    """
    Validate that rows contain no duplicate image_id values.

    This is useful for single-label classification and most dataset-version
    manifests, where CVDMS should produce one row per image per version.
    """
    seen: set[str] = set()

    for row in rows:
        if row.image_id in seen:
            raise ValueError(
                f"{context} contains duplicate image_id={row.image_id!r}"
            )
        seen.add(row.image_id)

def validate_no_split_overlap(bundle: CvdmsManifestBundle) -> None:
    """
    Validate that image IDs do not overlap between train, val, and test splits.
    """
    split_to_ids = {
        "train": {row.image_id for row in bundle.train},
        "val": {row.image_id for row in bundle.val},
        "test": {row.image_id for row in bundle.test},
    }

    train_val = split_to_ids["train"] & split_to_ids["val"]
    train_test = split_to_ids["train"] & split_to_ids["test"]
    val_test = split_to_ids["val"] & split_to_ids["test"]

    if train_val:
        raise ValueError(
            f"train/val split overlap contains {len(train_val)} image IDs; "
            f"example={sorted(train_val)[0]!r}"
        )

    if train_test:
        raise ValueError(
            f"train/test split overlap contains {len(train_test)} image IDs; "
            f"example={sorted(train_test)[0]!r}"
        )

    if val_test:
        raise ValueError(
            f"val/test split overlap contains {len(val_test)} image IDs; "
            f"example={sorted(val_test)[0]!r}"
        )

def validate_bundle_basic(bundle: CvdmsManifestBundle) -> None:
    """
    Run basic sanity checks on a manifest bundle.
    """
    if bundle.total_count <= 0:
        raise ValueError("CVDMS manifest bundle has zero total rows")

    validate_unique_image_ids(bundle.train, context="train manifest")
    validate_unique_image_ids(bundle.val, context="val manifest")
    validate_unique_image_ids(bundle.test, context="test manifest")
    validate_no_split_overlap(bundle)

def _format_row_context(
    *,
    row_idx: int | None,
    manifest_uri: str | None,
) -> str:
    parts: list[str] = ["CVDMS manifest row"]

    if manifest_uri:
        parts.append(f"uri={manifest_uri}")

    if row_idx is not None:
        parts.append(f"row_idx={row_idx}")

    return " ".join(parts)

def _require_nonempty_string(value: Any, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} cannot be null")

    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} cannot be empty")

    return text

def _require_s3_uri(value: Any, field_name: str) -> str:
    text = _require_nonempty_string(value, field_name)

    if not text.startswith("s3://"):
        raise ValueError(f"{field_name} must be an S3 URI, got {text!r}")

    return text

def _require_valid_split(value: Any, field_name: str) -> SplitName:
    text = _require_nonempty_string(value, field_name)

    if text not in _VALID_SPLITS:
        raise ValueError(
            f"{field_name} must be one of {_VALID_SPLITS}, got {text!r}"
        )

    return text  # type: ignore[return-value]

def _require_valid_label_type(value: Any, field_name: str) -> str:
    text = _require_nonempty_string(value, field_name)

    if text not in _VALID_LABEL_TYPES:
        raise ValueError(
            f"{field_name} must be one of {sorted(_VALID_LABEL_TYPES)}, got {text!r}"
        )

    return text