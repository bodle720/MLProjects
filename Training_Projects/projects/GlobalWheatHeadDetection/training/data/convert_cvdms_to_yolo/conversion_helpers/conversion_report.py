import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_jsonable(value: Any) -> Any:
    """
    Convert common Python objects into JSON-serializable values.

    This keeps report-writing code tolerant of dataclasses, Paths, sets,
    Decimals, and nested structures.
    """
    if is_dataclass(value):
        return to_jsonable(asdict(value))

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Decimal):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): to_jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, list | tuple):
        return [
            to_jsonable(item)
            for item in value
        ]

    if isinstance(value, set):
        return sorted(to_jsonable(item) for item in value)

    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=2, sort_keys=True)


def build_conversion_report(
    dataset_name: str,
    output_dir: Path,
    dataset_yaml_path: Path,
    metadata_summary: dict[str, Any],
    split_summary: dict[str, Any],
    split_stats: dict[str, Any],
    config_summary: dict[str, Any],
    warnings: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = {
        "created_at_utc": utc_now_iso(),
        "dataset_name": dataset_name,
        "output_dir": str(output_dir),
        "dataset_yaml_path": str(dataset_yaml_path),
        "metadata_summary": metadata_summary,
        "split_summary": split_summary,
        "split_stats": split_stats,
        "config_summary": config_summary,
        "warnings": warnings or [],
    }

    if extra:
        report["extra"] = extra

    return report


def write_conversion_report(
    report_path: Path,
    dataset_name: str,
    output_dir: Path,
    dataset_yaml_path: Path,
    metadata_summary: dict[str, Any],
    split_summary: dict[str, Any],
    split_stats: dict[str, Any],
    config_summary: dict[str, Any],
    warnings: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    report = build_conversion_report(
        dataset_name=dataset_name,
        output_dir=output_dir,
        dataset_yaml_path=dataset_yaml_path,
        metadata_summary=metadata_summary,
        split_summary=split_summary,
        split_stats=split_stats,
        config_summary=config_summary,
        warnings=warnings,
        extra=extra,
    )

    write_json(report_path, report)
    return report_path


def summarize_split_stats(split_stats: dict[str, Any]) -> dict[str, Any]:
    """
    Build a compact top-level summary from per-split write stats.

    Expected split stat fields are intentionally simple so this can work with
    either dataclasses converted to dicts or plain dicts.
    """
    summary: dict[str, Any] = {
        "rows_seen": 0,
        "images_written": 0,
        "labels_written": 0,
        "boxes_seen": 0,
        "boxes_written": 0,
        "boxes_dropped_class_not_selected": 0,
        "boxes_dropped_invalid": 0,
        "images_skipped_no_kept_boxes": 0,
        "images_skipped_missing_cached_image": 0,
        "failures": 0,
        "by_split": {},
    }

    for split_name, stats in split_stats.items():
        stats_dict = to_jsonable(stats)

        split_summary = {
            "rows_seen": stats_dict.get("rows_seen", 0),
            "images_written": stats_dict.get("images_written", 0),
            "labels_written": stats_dict.get("labels_written", 0),
            "boxes_seen": stats_dict.get("boxes_seen", 0),
            "boxes_written": stats_dict.get("boxes_written", 0),
            "boxes_dropped_class_not_selected": stats_dict.get("boxes_dropped_class_not_selected", 0),
            "boxes_dropped_invalid": stats_dict.get("boxes_dropped_invalid", 0),
            "images_skipped_no_kept_boxes": stats_dict.get("images_skipped_no_kept_boxes", 0),
            "images_skipped_missing_cached_image": stats_dict.get("images_skipped_missing_cached_image", 0),
            "failures": len(stats_dict.get("failures", [])),
        }

        summary["by_split"][split_name] = split_summary

        for key, value in split_summary.items():
            if key != "failures":
                summary[key] = summary.get(key, 0) + value

        summary["failures"] += split_summary["failures"]

    return summary


def validate_report_has_no_failures(report: dict[str, Any]) -> None:
    """
    Optional strict check for scripts that want to fail if conversion had any
    recorded failures.
    """
    split_stats = report.get("split_stats", {})
    compact_summary = summarize_split_stats(split_stats)

    if compact_summary["failures"] > 0:
        raise ValueError(
            f"Conversion report contains {compact_summary['failures']} recorded failures."
        )

    if compact_summary["images_skipped_missing_cached_image"] > 0:
        raise ValueError(
            "Conversion report contains missing cached images: "
            f"{compact_summary['images_skipped_missing_cached_image']}"
        )