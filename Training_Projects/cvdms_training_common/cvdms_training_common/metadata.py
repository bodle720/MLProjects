"""
CVDMS dataset metadata helpers.

This module loads and validates the version-scoped metadata.json artifact
written by CVDMS dataset creation/update workflows.

Expected CVDMS path shape:

    s3://<datasets-bucket>/datasets/<dataset_id>/v<version>/metadata/metadata.json

The metadata file is expected to include version-specific training information,
including effective_classes, class_to_idx, idx_to_class, and artifact URIs for
the train/val/test manifests.
"""

from dataclasses import dataclass
from typing import Any, Literal

from cvdms_training_common.s3_io import read_s3_json

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
class CvdmsDatasetMetadata:
    """
    Version-scoped metadata for a CVDMS dataset.

    This object should describe the exact dataset version being used for
    training, not just the dataset-level defaults.

    Important:
        class_to_idx and idx_to_class are derived from effective_classes and
        are version-specific. If a later dataset version removes classes, this
        metadata should reflect the reduced class set.
    """

    dataset_id: str
    version: int
    label_type: str
    effective_classes: tuple[str, ...]
    class_to_idx: dict[str, int]
    idx_to_class: dict[int, str]
    artifacts: dict[str, str]
    raw: dict[str, Any]

    @property
    def num_classes(self) -> int:
        return len(self.effective_classes)

    @property
    def train_manifest_uri(self) -> str:
        return self.get_manifest_uri("train")

    @property
    def val_manifest_uri(self) -> str:
        return self.get_manifest_uri("val")

    @property
    def test_manifest_uri(self) -> str:
        return self.get_manifest_uri("test")

    def get_manifest_uri(self, split: SplitName) -> str:
        """
        Return the manifest URI for a split.

        Expected artifact keys:
            train_manifest_uri
            val_manifest_uri
            test_manifest_uri
        """
        if split not in _VALID_SPLITS:
            raise ValueError(f"Invalid split={split!r}; expected one of {_VALID_SPLITS}")

        artifact_key = f"{split}_manifest_uri"
        uri = self.artifacts.get(artifact_key)

        if not uri:
            raise ValueError(
                f"metadata.artifacts is missing non-empty {artifact_key!r}"
            )

        if not isinstance(uri, str) or not uri.startswith("s3://"):
            raise ValueError(
                f"metadata.artifacts[{artifact_key!r}] must be an S3 URI, got {uri!r}"
            )

        return uri

    def as_training_provenance(self) -> dict[str, Any]:
        """
        Return a compact dictionary suitable for saving into checkpoints,
        MLflow artifacts, TensorBoard text, or evaluation summaries.
        """
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "label_type": self.label_type,
            "effective_classes": list(self.effective_classes),
            "class_to_idx": dict(self.class_to_idx),
            "idx_to_class": {str(k): v for k, v in self.idx_to_class.items()},
            "num_classes": self.num_classes,
            "artifacts": dict(self.artifacts),
        }

def load_cvdms_metadata(
    metadata_uri: str,
    *,
    s3_client=None,
    min_classes: int | None = None,
) -> CvdmsDatasetMetadata:
    """
    Load and validate a CVDMS metadata.json file from S3.

    Args:
        metadata_uri:
            S3 URI to metadata/metadata.json.
        s3_client:
            Optional boto3 S3 client. Useful for local runs with a named profile.
        min_classes:
            Optional minimum number of effective classes. For single-label
            classification training this is usually 2, but the metadata loader
            itself does not force that globally because some future tasks may
            have different requirements.

    Returns:
        CvdmsDatasetMetadata.
    """
    raw = read_s3_json(metadata_uri, s3_client=s3_client)

    metadata = parse_cvdms_metadata(raw)

    if min_classes is not None and metadata.num_classes < min_classes:
        raise ValueError(
            f"metadata effective_classes has {metadata.num_classes} classes, "
            f"but min_classes={min_classes}"
        )

    return metadata

def parse_cvdms_metadata(raw: dict[str, Any]) -> CvdmsDatasetMetadata:
    """
    Validate and parse a raw metadata.json dictionary.

    This function is useful for tests because it does not require S3.
    """
    if not isinstance(raw, dict):
        raise TypeError(f"metadata must be a dict, got {type(raw).__name__}")

    dataset_id = _require_nonempty_string(raw.get("dataset_id"), "dataset_id")
    version = _require_positive_int(raw.get("version"), "version")
    label_type = _require_nonempty_string(raw.get("label_type"), "label_type")

    if label_type not in _VALID_LABEL_TYPES:
        raise ValueError(
            f"Unsupported label_type={label_type!r}; expected one of "
            f"{sorted(_VALID_LABEL_TYPES)}"
        )

    effective_classes = _parse_effective_classes(raw.get("effective_classes"))

    class_to_idx = _parse_class_to_idx(raw.get("class_to_idx"))
    idx_to_class = _parse_idx_to_class(raw.get("idx_to_class"))

    _validate_class_maps(
        effective_classes=effective_classes,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
    )

    artifacts = _parse_artifacts(raw.get("artifacts"))
    _validate_manifest_artifacts(artifacts)

    return CvdmsDatasetMetadata(
        dataset_id=dataset_id,
        version=version,
        label_type=label_type,
        effective_classes=tuple(effective_classes),
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        artifacts=artifacts,
        raw=dict(raw),
    )

def get_manifest_uri(metadata: CvdmsDatasetMetadata, split: SplitName) -> str:
    """
    Convenience wrapper around metadata.get_manifest_uri(split).

    This exists so callers can choose either:

        metadata.get_manifest_uri("train")

    or:

        get_manifest_uri(metadata, "train")
    """
    return metadata.get_manifest_uri(split)

def _parse_effective_classes(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(
            f"metadata.effective_classes must be list[str], got {type(value).__name__}"
        )

    classes: list[str] = []
    seen: set[str] = set()

    for idx, item in enumerate(value):
        class_name = _require_nonempty_string(
            item,
            f"effective_classes[{idx}]",
        )

        if class_name in seen:
            raise ValueError(
                f"metadata.effective_classes contains duplicate class {class_name!r}"
            )

        seen.add(class_name)
        classes.append(class_name)

    if not classes:
        raise ValueError("metadata.effective_classes cannot be empty")

    expected_sorted = sorted(classes)
    if classes != expected_sorted:
        raise ValueError(
            "metadata.effective_classes must be sorted deterministically. "
            f"Expected {expected_sorted}, got {classes}"
        )

    return classes

def _parse_class_to_idx(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(
            f"metadata.class_to_idx must be an object, got {type(value).__name__}"
        )

    out: dict[str, int] = {}

    for raw_class_name, raw_idx in value.items():
        class_name = _require_nonempty_string(
            raw_class_name,
            "class_to_idx key",
        )

        if isinstance(raw_idx, bool) or not isinstance(raw_idx, int):
            raise ValueError(
                f"metadata.class_to_idx[{class_name!r}] must be int, got {raw_idx!r}"
            )

        if raw_idx < 0:
            raise ValueError(
                f"metadata.class_to_idx[{class_name!r}] cannot be negative: {raw_idx}"
            )

        out[class_name] = raw_idx

    if not out:
        raise ValueError("metadata.class_to_idx cannot be empty")

    return out

def _parse_idx_to_class(value: Any) -> dict[int, str]:
    if not isinstance(value, dict):
        raise ValueError(
            f"metadata.idx_to_class must be an object, got {type(value).__name__}"
        )

    out: dict[int, str] = {}

    for raw_idx, raw_class_name in value.items():
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"metadata.idx_to_class key must be int-like, got {raw_idx!r}"
            ) from exc

        if idx < 0:
            raise ValueError(f"metadata.idx_to_class key cannot be negative: {idx}")

        class_name = _require_nonempty_string(
            raw_class_name,
            f"idx_to_class[{raw_idx!r}]",
        )

        out[idx] = class_name

    if not out:
        raise ValueError("metadata.idx_to_class cannot be empty")

    return out

def _validate_class_maps(
    *,
    effective_classes: list[str],
    class_to_idx: dict[str, int],
    idx_to_class: dict[int, str],
) -> None:
    expected_class_to_idx = {
        class_name: idx
        for idx, class_name in enumerate(effective_classes)
    }

    if class_to_idx != expected_class_to_idx:
        raise ValueError(
            "metadata.class_to_idx must match effective_classes order. "
            f"Expected {expected_class_to_idx}, got {class_to_idx}"
        )

    expected_idx_to_class = {
        idx: class_name
        for class_name, idx in class_to_idx.items()
    }

    if idx_to_class != expected_idx_to_class:
        raise ValueError(
            "metadata.idx_to_class must be the inverse of class_to_idx. "
            f"Expected {expected_idx_to_class}, got {idx_to_class}"
        )

    expected_indices = set(range(len(effective_classes)))
    actual_indices = set(class_to_idx.values())

    if actual_indices != expected_indices:
        raise ValueError(
            "metadata.class_to_idx indices must be contiguous starting at 0. "
            f"Expected {sorted(expected_indices)}, got {sorted(actual_indices)}"
        )

def _parse_artifacts(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(
            f"metadata.artifacts must be an object, got {type(value).__name__}"
        )

    artifacts: dict[str, str] = {}

    for raw_key, raw_uri in value.items():
        key = _require_nonempty_string(raw_key, "artifacts key")
        uri = _require_nonempty_string(raw_uri, f"artifacts[{key!r}]")

        if not uri.startswith("s3://"):
            raise ValueError(
                f"metadata.artifacts[{key!r}] must be an S3 URI, got {uri!r}"
            )

        artifacts[key] = uri

    return artifacts

def _validate_manifest_artifacts(artifacts: dict[str, str]) -> None:
    required_keys = [
        "train_manifest_uri",
        "val_manifest_uri",
        "test_manifest_uri",
    ]

    for key in required_keys:
        uri = artifacts.get(key)
        if not uri:
            raise ValueError(f"metadata.artifacts is missing required key {key!r}")

        if not uri.startswith("s3://"):
            raise ValueError(
                f"metadata.artifacts[{key!r}] must be an S3 URI, got {uri!r}"
            )

def _require_nonempty_string(value: Any, field_name: str) -> str:
    if value is None:
        raise ValueError(f"metadata.{field_name} cannot be null")

    text = str(value).strip()

    if not text:
        raise ValueError(f"metadata.{field_name} cannot be empty")

    return text

def _require_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"metadata.{field_name} must be a positive int, got {value!r}"
        )

    if value < 1:
        raise ValueError(
            f"metadata.{field_name} must be >= 1, got {value}"
        )

    return value