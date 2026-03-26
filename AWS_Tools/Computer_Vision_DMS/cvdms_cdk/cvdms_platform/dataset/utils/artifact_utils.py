import csv
import io
import json
from collections import Counter
from typing import Any

_VALID_SPLITS = ("train", "val", "test")

def write_dataset_artifacts(
    *,
    s3_client: Any,
    dataset_bucket_name: str,
    dataset_id: str,
    version: int,
    label_type: str,
    split_strategy_name: str,
    selection_sql: str,
    selection_config: dict[str, Any],
    split_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Write all dataset-version S3 artifacts under:

    s3://<dataset_bucket_name>/datasets/<dataset_id>/v<version>/
    """
    if version < 1:
        raise ValueError(f"version must be >= 1, got {version}")

    base_prefix = build_dataset_version_prefix(
        dataset_id=dataset_id,
        version=version,
    )

    metadata_uris = write_selection_and_metadata_inputs(
        s3_client=s3_client,
        dataset_bucket_name=dataset_bucket_name,
        base_prefix=base_prefix,
        selection_sql=selection_sql,
        selection_config=selection_config,
    )

    manifest_uris = write_manifest_artifacts(
        s3_client=s3_client,
        dataset_bucket_name=dataset_bucket_name,
        base_prefix=base_prefix,
        label_type=label_type,
        split_rows=split_rows,
    )

    membership_enriched_csv_uri = write_membership_enriched_csv(
        s3_client=s3_client,
        dataset_bucket_name=dataset_bucket_name,
        base_prefix=base_prefix,
        split_rows=split_rows,
    )

    metadata_json_uri = write_metadata_json(
        s3_client=s3_client,
        dataset_bucket_name=dataset_bucket_name,
        base_prefix=base_prefix,
        dataset_id=dataset_id,
        version=version,
        label_type=label_type,
        split_strategy_name=split_strategy_name,
        split_rows=split_rows,
        manifest_uris=manifest_uris,
        membership_enriched_csv_uri=membership_enriched_csv_uri,
        selection_sql_uri=metadata_uris["selection_sql_uri"],
        selection_config_uri=metadata_uris["selection_config_uri"],
    )

    return {
        "base_prefix": base_prefix,
        "selection_sql_uri": metadata_uris["selection_sql_uri"],
        "selection_config_uri": metadata_uris["selection_config_uri"],
        "metadata_json_uri": metadata_json_uri,
        "membership_enriched_csv_uri": membership_enriched_csv_uri,
        "manifest_uris": manifest_uris,
    }

def build_dataset_version_prefix(*, dataset_id: str, version: int) -> str:
    dataset_id = str(dataset_id).strip()
    if not dataset_id:
        raise ValueError("dataset_id cannot be empty")

    return f"datasets/{dataset_id}/v{version}"

def write_selection_and_metadata_inputs(
    *,
    s3_client: Any,
    dataset_bucket_name: str,
    base_prefix: str,
    selection_sql: str,
    selection_config: dict[str, Any],
) -> dict[str, str]:
    selection_sql_key = f"{base_prefix}/metadata/selection.sql"
    selection_config_key = f"{base_prefix}/metadata/selection_config.json"

    _put_s3_text(
        s3_client=s3_client,
        bucket=dataset_bucket_name,
        key=selection_sql_key,
        body=selection_sql.rstrip() + "\n",
        content_type="text/plain",
    )

    _put_s3_text(
        s3_client=s3_client,
        bucket=dataset_bucket_name,
        key=selection_config_key,
        body=json.dumps(selection_config, indent=2, sort_keys=True) + "\n",
        content_type="application/json",
    )

    return {
        "selection_sql_uri": _s3_uri(dataset_bucket_name, selection_sql_key),
        "selection_config_uri": _s3_uri(dataset_bucket_name, selection_config_key),
    }

def write_manifest_artifacts(
    *,
    s3_client: Any,
    dataset_bucket_name: str,
    base_prefix: str,
    label_type: str,
    split_rows: list[dict[str, Any]],
) -> dict[str, str]:
    """
    Write:
    - manifests/all.jsonl
    - manifests/train.jsonl
    - manifests/val.jsonl
    - manifests/test.jsonl

    Manifest records are emitted in a canonical dataset-export shape, not yet
    a task-specific SageMaker-native schema.
    """
    rows_by_split: dict[str, list[dict[str, Any]]] = {s: [] for s in _VALID_SPLITS}

    for row in split_rows:
        split = _require_valid_split(row.get("split"))
        rows_by_split[split].append(row)

    all_records = [_build_manifest_record(label_type=label_type, row=row) for row in split_rows]
    train_records = [_build_manifest_record(label_type=label_type, row=row) for row in rows_by_split["train"]]
    val_records = [_build_manifest_record(label_type=label_type, row=row) for row in rows_by_split["val"]]
    test_records = [_build_manifest_record(label_type=label_type, row=row) for row in rows_by_split["test"]]

    manifest_payloads = {
        "all": _jsonl_dumps(all_records),
        "train": _jsonl_dumps(train_records),
        "val": _jsonl_dumps(val_records),
        "test": _jsonl_dumps(test_records),
    }

    uris: dict[str, str] = {}
    for split_name, payload in manifest_payloads.items():
        key = f"{base_prefix}/manifests/{split_name}.jsonl"
        _put_s3_text(
            s3_client=s3_client,
            bucket=dataset_bucket_name,
            key=key,
            body=payload,
            content_type="application/x-ndjson",
        )
        uris[f"{split_name}_manifest_uri"] = _s3_uri(dataset_bucket_name, key)

    return uris

def write_membership_enriched_csv(
    *,
    s3_client: Any,
    dataset_bucket_name: str,
    base_prefix: str,
    split_rows: list[dict[str, Any]],
) -> str:
    """
    Write a CSV with the enriched split rows for later analysis.

    Arrays are serialized as JSON strings for readability and round-tripping.
    """
    key = f"{base_prefix}/profile/membership_enriched.csv"

    fieldnames = _infer_csv_fieldnames(split_rows)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    for row in split_rows:
        writer.writerow(_csv_safe_row(row=row, fieldnames=fieldnames))

    _put_s3_text(
        s3_client=s3_client,
        bucket=dataset_bucket_name,
        key=key,
        body=buffer.getvalue(),
        content_type="text/csv",
    )

    return _s3_uri(dataset_bucket_name, key)

def write_metadata_json(
    *,
    s3_client: Any,
    dataset_bucket_name: str,
    base_prefix: str,
    dataset_id: str,
    version: int,
    label_type: str,
    split_strategy_name: str,
    split_rows: list[dict[str, Any]],
    manifest_uris: dict[str, str],
    membership_enriched_csv_uri: str,
    selection_sql_uri: str,
    selection_config_uri: str,
) -> str:
    key = f"{base_prefix}/metadata/metadata.json"

    summary = build_dataset_metadata_summary(
        dataset_id=dataset_id,
        version=version,
        label_type=label_type,
        split_strategy_name=split_strategy_name,
        split_rows=split_rows,
    )

    metadata = {
        **summary,
        "artifacts": {
            "selection_sql_uri": selection_sql_uri,
            "selection_config_uri": selection_config_uri,
            "membership_enriched_csv_uri": membership_enriched_csv_uri,
            **manifest_uris,
        },
    }

    _put_s3_text(
        s3_client=s3_client,
        bucket=dataset_bucket_name,
        key=key,
        body=json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        content_type="application/json",
    )

    return _s3_uri(dataset_bucket_name, key)

def build_dataset_metadata_summary(
    *,
    dataset_id: str,
    version: int,
    label_type: str,
    split_strategy_name: str,
    split_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    split_counts = Counter()
    source_counts_by_split: dict[str, Counter[str]] = {s: Counter() for s in _VALID_SPLITS}
    class_counts_by_split: dict[str, Counter[str]] = {s: Counter() for s in _VALID_SPLITS}
    lighting_counts_by_split: dict[str, Counter[str]] = {s: Counter() for s in _VALID_SPLITS}
    blur_counts_by_split: dict[str, Counter[str]] = {s: Counter() for s in _VALID_SPLITS}
    contrast_counts_by_split: dict[str, Counter[str]] = {s: Counter() for s in _VALID_SPLITS}
    color_counts_by_split: dict[str, Counter[str]] = {s: Counter() for s in _VALID_SPLITS}

    for row in split_rows:
        split = _require_valid_split(row.get("split"))
        split_counts[split] += 1

        data_source = _optional_string(row.get("data_source"))
        if data_source:
            source_counts_by_split[split][data_source] += 1

        for class_name in _normalize_string_array(row.get("classes_present")):
            class_counts_by_split[split][class_name] += 1

        lighting_bucket = _optional_string(row.get("lighting_bucket"))
        if lighting_bucket:
            lighting_counts_by_split[split][lighting_bucket] += 1

        blur_bucket = _optional_string(row.get("blur_bucket"))
        if blur_bucket:
            blur_counts_by_split[split][blur_bucket] += 1

        contrast_bucket = _optional_string(row.get("contrast_bucket"))
        if contrast_bucket:
            contrast_counts_by_split[split][contrast_bucket] += 1

        color_bucket = _optional_string(row.get("color_bucket"))
        if color_bucket:
            color_counts_by_split[split][color_bucket] += 1

    return {
        "dataset_id": dataset_id,
        "version": version,
        "label_type": label_type,
        "split_strategy_name": split_strategy_name,
        "row_count": len(split_rows),
        "split_counts": {split: split_counts.get(split, 0) for split in _VALID_SPLITS},
        "class_counts_by_split": {
            split: dict(sorted(class_counts_by_split[split].items()))
            for split in _VALID_SPLITS
        },
        "source_counts_by_split": {
            split: dict(sorted(source_counts_by_split[split].items()))
            for split in _VALID_SPLITS
        },
        "lighting_counts_by_split": {
            split: dict(sorted(lighting_counts_by_split[split].items()))
            for split in _VALID_SPLITS
        },
        "blur_counts_by_split": {
            split: dict(sorted(blur_counts_by_split[split].items()))
            for split in _VALID_SPLITS
        },
        "contrast_counts_by_split": {
            split: dict(sorted(contrast_counts_by_split[split].items()))
            for split in _VALID_SPLITS
        },
        "color_counts_by_split": {
            split: dict(sorted(color_counts_by_split[split].items()))
            for split in _VALID_SPLITS
        },
    }

def _build_manifest_record(*, label_type: str, row: dict[str, Any]) -> dict[str, Any]:
    """
    Canonical dataset-export manifest record.

    This is intentionally simple and consistent across task types for now.
    """
    base = {
        "image_id": _require_nonempty_string(row.get("image_id"), field_name="image_id"),
        "source_ref": _require_nonempty_string(row.get("source_ref"), field_name="source_ref"),
        "split": _require_valid_split(row.get("split")),
        "label_type": label_type,
    }

    if label_type == "single-label":
        base["label"] = _require_nonempty_string(row.get("label"), field_name="label")
        return base

    if label_type == "multi-label":
        base["labels"] = _normalize_string_array(row.get("labels"), require_nonempty=True)
        return base

    if label_type == "object-detection":
        base["bbox_annotation_ids"] = _normalize_string_array(
            row.get("bbox_annotation_ids"),
            require_nonempty=True,
        )
        return base

    if label_type == "semantic-segmentation":
        base["semantic_mask_ids"] = _normalize_string_array(
            row.get("semantic_mask_ids"),
            require_nonempty=True,
        )
        return base

    if label_type == "instance-segmentation":
        base["instance_annotation_ids"] = _normalize_string_array(
            row.get("instance_annotation_ids"),
            require_nonempty=True,
        )
        return base

    raise ValueError(f"Unsupported label_type: {label_type}")

def _infer_csv_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred_order = [
        "image_id",
        "source_ref",
        "split",
        "label",
        "labels",
        "bbox_annotation_ids",
        "semantic_mask_ids",
        "instance_annotation_ids",
        "classes_present",
        "sha256_hash",
        "data_source",
        "uploaded_at",
        "img_type",
        "img_height",
        "img_width",
        "num_channels",
        "dtype",
        "file_size_mb",
        "luma_mean",
        "luma_p10",
        "luma_p90",
        "dark_frac",
        "bright_frac",
        "contrast_luma_std",
        "contrast_luma_p90_p10",
        "blur_laplacian_var",
        "sat_mean",
        "colorfulness",
        "lighting_bucket",
        "blur_bucket",
        "contrast_bucket",
        "color_bucket",
        "dataset_label_type",
    ]

    all_keys: set[str] = set()
    for row in rows:
        all_keys.update(row.keys())

    ordered = [key for key in preferred_order if key in all_keys]
    remainder = sorted(all_keys - set(ordered))
    return ordered + remainder

def _csv_safe_row(*, row: dict[str, Any], fieldnames: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    for field in fieldnames:
        value = row.get(field)

        if isinstance(value, list):
            out[field] = json.dumps(value, sort_keys=True)
        elif isinstance(value, dict):
            out[field] = json.dumps(value, sort_keys=True)
        elif value is None:
            out[field] = ""
        else:
            out[field] = value

    return out

def _jsonl_dumps(records: list[dict[str, Any]]) -> str:
    if not records:
        return ""

    return "".join(
        json.dumps(record, sort_keys=True) + "\n"
        for record in records
    )

def _normalize_string_array(value: Any, *, require_nonempty: bool = False) -> list[str]:
    if value is None:
        values: list[str] = []
    elif isinstance(value, list):
        values = [str(v).strip() for v in value if str(v).strip()]
    else:
        raise TypeError(f"Expected list[str] | None, got {type(value).__name__}")

    values = sorted(set(values))

    if require_nonempty and not values:
        raise ValueError("Expected non-empty string array")

    return values

def _optional_string(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    return text or None

def _require_nonempty_string(value: Any, *, field_name: str) -> str:
    text = _optional_string(value)
    if not text:
        raise ValueError(f"{field_name} cannot be empty")
    return text

def _require_valid_split(value: Any) -> str:
    split = _require_nonempty_string(value, field_name="split")
    if split not in _VALID_SPLITS:
        raise ValueError(f"Invalid split: {split!r}")
    return split

def _put_s3_text(
    *,
    s3_client: Any,
    bucket: str,
    key: str,
    body: str,
    content_type: str,
) -> None:
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body.encode("utf-8"),
        ContentType=content_type,
    )

def _s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"