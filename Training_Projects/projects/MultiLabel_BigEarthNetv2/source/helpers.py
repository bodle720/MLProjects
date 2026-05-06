import json
from typing import Any, Iterable

from botocore.exceptions import ClientError

from cvdms_training_common.s3_io import parse_s3_uri
from cvdms_training_common.image_loading import (
    S3ImageLoader,
    LocalMirrorImageLoader,
)

SPLITS = ("train", "val", "test")

def require_nonempty_string(value: Any, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} cannot be null")

    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} cannot be empty")

    return text

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

def require_positive_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a positive number, got {value!r}")

    number = float(value)

    if number <= 0:
        raise ValueError(f"{field_name} must be > 0, got {number}")

    return number

def require_probability_threshold(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a float in [0, 1], got {value!r}")

    number = float(value)

    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1], got {number}")

    return number

def require_threshold_sweep_values(value: Any, field_name: str) -> list[float]:
    if value is None:
        return []

    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list, got {type(value).__name__}")

    thresholds = [
        require_probability_threshold(item, f"{field_name}[{idx}]")
        for idx, item in enumerate(value)
    ]

    return sorted(set(thresholds))

def build_project_image_loader(
    *,
    data_config: dict[str, Any],
    s3_client,
):
    loader_config = data_config.get("image_loader") or {}

    if not isinstance(loader_config, dict):
        raise TypeError(
            f"data.image_loader must be a dictionary, got {type(loader_config).__name__}"
        )

    mode = str(loader_config.get("mode", "s3")).strip().lower()

    if mode == "s3":
        return S3ImageLoader(s3_client=s3_client)

    if mode == "local_mirror":
        cache_dir = require_nonempty_string(
            loader_config.get("cache_dir"),
            "data.image_loader.cache_dir",
        )

        return LocalMirrorImageLoader(
            local_root=cache_dir
        )

    raise ValueError(
        "data.image_loader.mode must be one of {'s3', 'local_mirror'}, "
        f"got {mode!r}"
    )

def _read_s3_bytes(uri: str, *, s3_client) -> bytes:
    parsed = parse_s3_uri(uri)

    try:
        response = s3_client.get_object(
            Bucket=parsed.bucket,
            Key=parsed.key,
        )
        return response["Body"].read()
    except ClientError as exc:
        raise RuntimeError(f"Failed to read S3 object: {uri}") from exc

def iter_jsonl_s3(uri: str, *, s3_client) -> Iterable[tuple[int, dict[str, Any]]]:
    data = _read_s3_bytes(uri, s3_client=s3_client)

    for line_number, line in enumerate(data.decode("utf-8-sig").splitlines(), start=1):
        text = line.strip()

        if not text:
            continue

        try:
            row = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL in {uri} at line {line_number}") from exc

        if not isinstance(row, dict):
            raise TypeError(
                f"Manifest row must be a JSON object in {uri} at line {line_number}, "
                f"got {type(row).__name__}"
            )

        yield line_number, row

def read_json_from_s3(uri: str, *, s3_client) -> dict[str, Any]:
    data = _read_s3_bytes(uri, s3_client=s3_client)

    try:
        payload = json.loads(data.decode("utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"S3 object is not valid JSON: {uri}") from exc

    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object from {uri}, got {type(payload).__name__}")

    return payload

def resolve_manifest_uris(metadata: dict[str, Any]) -> dict[str, str]:
    """
    Resolve train/val/test manifest URIs from CVDMS metadata.json.

    This searches common top-level layouts and recursively searches nested
    structures such as metadata["artifacts"].

    Supported examples include:

        manifest_uris: {train: "...", val: "...", test: "..."}
        manifests: {train: "...", val: "...", test: "..."}
        artifacts: {manifests: {train: "...", val: "...", test: "..."}}
        artifacts: {train: {manifest_uri: "..."}, ...}
        train_manifest_uri / val_manifest_uri / test_manifest_uri
    """
    resolved: dict[str, str] = {}

    preferred_roots = [
        metadata.get("manifest_uris"),
        metadata.get("manifest_s3_uris"),
        metadata.get("split_manifest_uris"),
        metadata.get("split_manifests"),
        metadata.get("manifests"),
        metadata.get("splits"),
        metadata.get("artifacts"),
        metadata,
    ]

    for root in preferred_roots:
        _collect_manifest_uris_recursive(root, resolved)

        if all(split in resolved for split in SPLITS):
            return {split: resolved[split] for split in SPLITS}

    available_keys = sorted(str(key) for key in metadata.keys())
    artifacts = metadata.get("artifacts")
    artifact_keys = sorted(str(key) for key in artifacts.keys()) if isinstance(artifacts, dict) else None

    raise ValueError(
        "Could not resolve train/val/test manifest URIs from metadata.json. "
        f"Available top-level keys: {available_keys}. "
        f"Artifact keys: {artifact_keys}"
    )

def _collect_manifest_uris_recursive(value: Any, resolved: dict[str, str]) -> None:
    if all(split in resolved for split in SPLITS):
        return

    if isinstance(value, dict):
        _collect_manifest_uris_from_current_dict(value, resolved)

        for child in value.values():
            _collect_manifest_uris_recursive(child, resolved)

            if all(split in resolved for split in SPLITS):
                return

    elif isinstance(value, list):
        for child in value:
            _collect_manifest_uris_recursive(child, resolved)

            if all(split in resolved for split in SPLITS):
                return

def _collect_manifest_uris_from_current_dict(
    value: dict[str, Any],
    resolved: dict[str, str],
) -> None:
    for split in SPLITS:
        if split in resolved:
            continue

        item = value.get(split)
        uri = _manifest_uri_from_item(item)
        if uri is not None:
            resolved[split] = uri
            continue

        for key in (
            f"{split}_manifest_uri",
            f"{split}_manifest_s3_uri",
            f"{split}_manifest",
            f"{split}_jsonl_uri",
            f"{split}_jsonl_s3_uri",
            f"{split}_uri",
            f"{split}_s3_uri",
        ):
            uri = _manifest_uri_from_item(value.get(key))
            if uri is not None:
                resolved[split] = uri
                break

    split_value = value.get("split")
    if isinstance(split_value, str):
        split = split_value.strip().lower()

        if split in SPLITS and split not in resolved:
            uri = _manifest_uri_from_item(value)
            if uri is not None:
                resolved[split] = uri

    name_value = value.get("name")
    if isinstance(name_value, str):
        split = name_value.strip().lower()

        if split in SPLITS and split not in resolved:
            uri = _manifest_uri_from_item(value)
            if uri is not None:
                resolved[split] = uri

def _manifest_uri_from_item(item: Any) -> str | None:
    if isinstance(item, str):
        text = item.strip()
        if _looks_like_manifest_uri(text):
            return text
        return None

    if isinstance(item, dict):
        for key in (
            "uri",
            "s3_uri",
            "manifest_uri",
            "manifest_s3_uri",
            "jsonl_uri",
            "jsonl_s3_uri",
            "path",
            "s3_path",
        ):
            value = item.get(key)

            if isinstance(value, str):
                text = value.strip()
                if _looks_like_manifest_uri(text):
                    return text

    return None

def _looks_like_manifest_uri(value: str) -> bool:
    text = value.strip()

    if not text.startswith("s3://"):
        return False

    lowered = text.lower()

    return (
        "manifest" in lowered
        or lowered.endswith(".jsonl")
        or "/manifests/" in lowered
    )