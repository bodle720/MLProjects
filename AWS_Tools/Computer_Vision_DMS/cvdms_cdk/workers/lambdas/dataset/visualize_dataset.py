import os
import io
import csv
import json
import math
from collections import Counter
from typing import Any

import boto3

from common.general_utils.logging_utils import log
from common.general_utils.s3_utils import write_s3_obj

DATASETS_BUCKET_NAME = os.environ["DATASETS_BUCKET_NAME"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]

TASK_NAME = "[VISUALIZATION_DATASET]"

s3 = boto3.client("s3")

_VALID_SPLITS = ("train", "val", "test")

_NUMERIC_FIELDS = [
    "img_height",
    "img_width",
    "num_channels",
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
]

_BUCKET_FIELDS = [
    "lighting_bucket",
    "blur_bucket",
    "contrast_bucket",
    "color_bucket",
]

def _require_nonempty_string(value: Any, *, field_name: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"{field_name} cannot be empty")
    return text

def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None

def _parse_optional_float(value: Any) -> float | None:
    text = _optional_string(value)
    if text is None:
        return None
    try:
        return float(text)
    except Exception:
        return None

def _parse_optional_int(value: Any) -> int | None:
    text = _optional_string(value)
    if text is None:
        return None
    try:
        # handles things like "10" and "10.0"
        f = float(text)
        if not math.isfinite(f):
            return None
        if abs(f - round(f)) > 1e-9:
            return None
        return int(round(f))
    except Exception:
        return None

def _parse_json_array_field(value: Any) -> list[str]:
    text = _optional_string(value)
    if text is None:
        return []

    try:
        obj = json.loads(text)
    except Exception:
        return []

    if not isinstance(obj, list):
        return []

    out: list[str] = []
    seen: set[str] = set()
    for item in obj:
        s = str(item).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out

def _normalize_row(row: dict[str, Any], label_type: str) -> dict[str, Any]:
    out = dict(row)

    out["image_id"] = _require_nonempty_string(out.get("image_id"), field_name="image_id")
    out["source_ref"] = _optional_string(out.get("source_ref"))
    out["split"] = _require_nonempty_string(out.get("split"), field_name="split")
    if out["split"] not in _VALID_SPLITS:
        raise ValueError(f"Invalid split: {out['split']!r}")

    out["label"] = _optional_string(out.get("label"))
    out["data_source"] = _optional_string(out.get("data_source"))
    out["uploaded_at"] = _optional_string(out.get("uploaded_at"))
    out["img_type"] = _optional_string(out.get("img_type"))
    out["dtype"] = _optional_string(out.get("dtype"))
    out["sha256_hash"] = _optional_string(out.get("sha256_hash"))
    out["dataset_label_type"] = _optional_string(out.get("dataset_label_type")) or label_type

    out["labels"] = _parse_json_array_field(out.get("labels"))
    out["bbox_annotation_ids"] = _parse_json_array_field(out.get("bbox_annotation_ids"))
    out["semantic_mask_ids"] = _parse_json_array_field(out.get("semantic_mask_ids"))
    out["instance_annotation_ids"] = _parse_json_array_field(out.get("instance_annotation_ids"))
    out["classes_present"] = _parse_json_array_field(out.get("classes_present"))

    for field in ("img_height", "img_width", "num_channels"):
        out[field] = _parse_optional_int(out.get(field))

    for field in (
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
    ):
        out[field] = _parse_optional_float(out.get(field))

    for field in _BUCKET_FIELDS:
        out[field] = _optional_string(out.get(field))

    # Ensure classes_present exists for class-oriented charts
    if not out["classes_present"]:
        if label_type == "single-label" and out["label"]:
            out["classes_present"] = [out["label"]]
        elif label_type == "multi-label" and out["labels"]:
            out["classes_present"] = list(out["labels"])

    return out

def _read_membership_enriched_csv(dataset_id: str, version: int, label_type: str) -> list[dict[str, Any]]:
    key = f"datasets/{dataset_id}/v{version}/profile/membership_enriched.csv"
    resp = s3.get_object(Bucket=DATASETS_BUCKET_NAME, Key=key)
    text = resp["Body"].read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    return [_normalize_row(row, label_type=label_type) for row in reader]

def _build_overview(
    dataset_id: str,
    version: int,
    label_type: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    split_counts = Counter(row["split"] for row in rows)
    total = len(rows)

    return {
        "dataset_id": dataset_id,
        "version": version,
        "label_type": label_type,
        "row_count": total,
        "split_counts": {s: split_counts.get(s, 0) for s in _VALID_SPLITS},
        "split_percentages": {
            s: (split_counts.get(s, 0) / total if total > 0 else 0.0)
            for s in _VALID_SPLITS
        },
    }

def _build_distribution_by_split(
    rows: list[dict[str, Any]],
    category_key: str,
) -> dict[str, Any]:
    counts_by_split: dict[str, Counter[str]] = {s: Counter() for s in _VALID_SPLITS}

    for row in rows:
        split = row["split"]

        if category_key == "classes_present":
            values = row.get("classes_present", [])
        else:
            value = row.get(category_key)
            values = [value] if value else []

        for value in values:
            text = _optional_string(value)
            if text:
                counts_by_split[split][text] += 1

    percentages_by_split: dict[str, dict[str, float]] = {}
    for split in _VALID_SPLITS:
        total = sum(counts_by_split[split].values())
        percentages_by_split[split] = {
            k: (v / total if total > 0 else 0.0)
            for k, v in sorted(counts_by_split[split].items())
        }

    return {
        "counts_by_split": {
            split: dict(sorted(counts_by_split[split].items()))
            for split in _VALID_SPLITS
        },
        "percentages_by_split": percentages_by_split,
    }

def _build_quality_distribution_by_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    for field in _BUCKET_FIELDS:
        counts_by_split: dict[str, Counter[str]] = {s: Counter() for s in _VALID_SPLITS}
        for row in rows:
            split = row["split"]
            value = _optional_string(row.get(field))
            if value:
                counts_by_split[split][value] += 1

        out[field] = {
            "counts_by_split": {
                split: dict(sorted(counts_by_split[split].items()))
                for split in _VALID_SPLITS
            }
        }

    return out

def _percentile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]

    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac

def _summarize_numeric_values(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
        }

    vals = sorted(values)
    return {
        "count": len(vals),
        "min": vals[0],
        "max": vals[-1],
        "mean": sum(vals) / len(vals),
        "p10": _percentile(vals, 0.10),
        "p25": _percentile(vals, 0.25),
        "p50": _percentile(vals, 0.50),
        "p75": _percentile(vals, 0.75),
        "p90": _percentile(vals, 0.90),
    }

def _build_numeric_feature_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"overall": {}, "by_split": {s: {} for s in _VALID_SPLITS}}

    for field in _NUMERIC_FIELDS:
        overall_values = [row[field] for row in rows if isinstance(row.get(field), (int, float))]
        out["overall"][field] = _summarize_numeric_values([float(v) for v in overall_values])

        for split in _VALID_SPLITS:
            split_values = [
                row[field]
                for row in rows
                if row["split"] == split and isinstance(row.get(field), (int, float))
            ]
            out["by_split"][split][field] = _summarize_numeric_values([float(v) for v in split_values])

    return out

def _build_histogram(values: list[float], bins: int = 20) -> dict[str, Any]:
    if not values:
        return {"bin_edges": [], "counts": []}

    vals = sorted(values)
    vmin = vals[0]
    vmax = vals[-1]

    if math.isclose(vmin, vmax):
        return {
            "bin_edges": [vmin, vmax],
            "counts": [len(vals)],
        }

    width = (vmax - vmin) / bins
    counts = [0 for _ in range(bins)]
    edges = [vmin + i * width for i in range(bins + 1)]

    for value in vals:
        idx = int((value - vmin) / width)
        if idx >= bins:
            idx = bins - 1
        counts[idx] += 1

    return {
        "bin_edges": edges,
        "counts": counts,
    }

def _build_numeric_feature_histograms(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"overall": {}, "by_split": {s: {} for s in _VALID_SPLITS}}

    for field in _NUMERIC_FIELDS:
        overall_values = [float(row[field]) for row in rows if isinstance(row.get(field), (int, float))]
        out["overall"][field] = _build_histogram(overall_values)

        for split in _VALID_SPLITS:
            split_values = [
                float(row[field])
                for row in rows
                if row["split"] == split and isinstance(row.get(field), (int, float))
            ]
            out["by_split"][split][field] = _build_histogram(split_values)

    return out

def _build_split_comparison_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    class_dist = _build_distribution_by_split(rows, "classes_present")
    source_dist = _build_distribution_by_split(rows, "data_source")

    train_class = class_dist["percentages_by_split"]["train"]
    train_source = source_dist["percentages_by_split"]["train"]

    def build_delta_block(train_map: dict[str, float], compare_map: dict[str, float]) -> dict[str, Any]:
        keys = sorted(set(train_map.keys()) | set(compare_map.keys()))
        deltas = {}
        max_delta = 0.0
        total_delta = 0.0

        for key in keys:
            train_value = train_map.get(key, 0.0)
            compare_value = compare_map.get(key, 0.0)
            abs_delta = abs(compare_value - train_value)
            deltas[key] = {
                "train_percentage": train_value,
                "compare_percentage": compare_value,
                "absolute_delta": abs_delta,
            }
            max_delta = max(max_delta, abs_delta)
            total_delta += abs_delta

        mean_delta = (total_delta / len(keys)) if keys else 0.0
        return {
            "deltas": deltas,
            "max_absolute_delta": max_delta,
            "mean_absolute_delta": mean_delta,
        }

    return {
        "class_comparison": {
            "val_vs_train": build_delta_block(
                train_class,
                class_dist["percentages_by_split"]["val"],
            ),
            "test_vs_train": build_delta_block(
                train_class,
                class_dist["percentages_by_split"]["test"],
            ),
        },
        "source_comparison": {
            "val_vs_train": build_delta_block(
                train_source,
                source_dist["percentages_by_split"]["val"],
            ),
            "test_vs_train": build_delta_block(
                train_source,
                source_dist["percentages_by_split"]["test"],
            ),
        },
    }

def _write_visualization_json(
    dataset_id: str,
    version: int,
    filename: str,
    payload: dict[str, Any],
) -> str:
    key = f"datasets/{dataset_id}/v{version}/visualization/{filename}"
    return write_s3_obj(
        bucket=DATASETS_BUCKET_NAME,
        key=key,
        content=json.dumps(payload, indent=2, sort_keys=True) + "\n",
        content_type="application/json",
        task_name=TASK_NAME,
    )

def handler(event, context):
    job_id = "unknown"
    user = "unknown"
    event_type = "DATASET_OP"

    try:
        job_id = _require_nonempty_string(event.get("job_id"), field_name="job_id")
        user = _require_nonempty_string(event.get("user"), field_name="user")
        event_type = _require_nonempty_string(event.get("event_type"), field_name="event_type")
        task_type = _require_nonempty_string(event.get("task_type"), field_name="task_type")

        dataset_context = event.get("dataset_context")
        if not isinstance(dataset_context, dict):
            raise ValueError("dataset_context must be an object")

        dataset_id = _require_nonempty_string(
            dataset_context.get("dataset_id"),
            field_name="dataset_context.dataset_id",
        )
        label_type = _require_nonempty_string(
            dataset_context.get("label_type"),
            field_name="dataset_context.label_type",
        )

        version_raw = dataset_context.get("new_version")
        if type(version_raw) is not int or version_raw < 1:
            raise ValueError("dataset_context.new_version must be an integer >= 1")

        version = version_raw

        if task_type not in {"create_dataset", "update_dataset"}:
            raise ValueError(
                f"{TASK_NAME} expected task_type create_dataset/update_dataset, got {task_type!r}"
            )

        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} Starting visualization generation for dataset_id={dataset_id}, version={version}",
            level="info",
        )

        rows = _read_membership_enriched_csv(
            dataset_id=dataset_id,
            version=version,
            label_type=label_type,
        )

        if not rows:
            raise ValueError(
                f"{TASK_NAME} membership_enriched.csv contained zero rows for dataset_id={dataset_id}, version={version}"
            )

        overview = _build_overview(dataset_id, version, label_type, rows)
        class_distribution = _build_distribution_by_split(rows, "classes_present")
        source_distribution = _build_distribution_by_split(rows, "data_source")
        quality_distribution = _build_quality_distribution_by_split(rows)
        numeric_summary = _build_numeric_feature_summary(rows)
        numeric_histograms = _build_numeric_feature_histograms(rows)
        split_comparison = _build_split_comparison_metrics(rows)

        written_files = {
            "overview_json_uri": _write_visualization_json(
                dataset_id, version, "overview.json", overview
            ),
            "class_distribution_by_split_json_uri": _write_visualization_json(
                dataset_id, version, "class_distribution_by_split.json", class_distribution
            ),
            "source_distribution_by_split_json_uri": _write_visualization_json(
                dataset_id, version, "source_distribution_by_split.json", source_distribution
            ),
            "quality_distribution_by_split_json_uri": _write_visualization_json(
                dataset_id, version, "quality_distribution_by_split.json", quality_distribution
            ),
            "numeric_feature_summary_json_uri": _write_visualization_json(
                dataset_id, version, "numeric_feature_summary.json", numeric_summary
            ),
            "numeric_feature_histograms_json_uri": _write_visualization_json(
                dataset_id, version, "numeric_feature_histograms.json", numeric_histograms
            ),
            "split_comparison_metrics_json_uri": _write_visualization_json(
                dataset_id, version, "split_comparison_metrics.json", split_comparison
            ),
        }

        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} Wrote visualization artifacts for dataset_id={dataset_id}, version={version}",
            level="info",
        )

        return {
            "status": "ok",
            "job_id": job_id,
            "user": user,
            "event_type": event_type,
            "task_type": task_type,
            "dataset_context": dataset_context,
            "visualization_prefix": f"s3://{DATASETS_BUCKET_NAME}/datasets/{dataset_id}/v{version}/visualization/",
            "row_count": len(rows),
            "written_files": written_files,
        }

    except Exception as e:
        error_message = f"{TASK_NAME} Failed: {type(e).__name__}: {e}"
        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            error_message,
            level="error",
        )
        raise