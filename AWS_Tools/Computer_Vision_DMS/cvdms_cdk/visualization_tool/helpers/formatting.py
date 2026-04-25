"""
Formatting helpers for the local CVDMS Dataset Viewer.

These functions keep charts.py and cvdms_dataset_viewer.py cleaner.
"""

from typing import Any

import pandas as pd


SPLIT_ORDER = ["train", "val", "test"]


FEATURE_LABELS: dict[str, str] = {
    "img_height": "Image Height",
    "img_width": "Image Width",
    "num_channels": "Number of Channels",
    "file_size_mb": "File Size (MB)",
    "luma_mean": "Luma Mean",
    "luma_p10": "Luma P10",
    "luma_p90": "Luma P90",
    "dark_frac": "Dark Fraction",
    "bright_frac": "Bright Fraction",
    "contrast_luma_std": "Contrast: Luma Std",
    "contrast_luma_p90_p10": "Contrast: Luma P90 - P10",
    "blur_laplacian_var": "Blur: Laplacian Variance",
    "sat_mean": "Saturation Mean",
    "colorfulness": "Colorfulness",
}


BUCKET_LABELS: dict[str, str] = {
    "lighting_bucket": "Lighting",
    "blur_bucket": "Blur",
    "contrast_bucket": "Contrast",
    "color_bucket": "Color",
}


def pct(value: Any, *, digits: int = 1) -> str:
    """
    Format a decimal fraction as a percentage string.

    Example:
        0.1234 -> "12.3%"
    """
    try:
        f = float(value)
    except Exception:
        return "—"

    return f"{100.0 * f:.{digits}f}%"


def number(value: Any, *, digits: int = 2) -> str:
    """
    Format a number for display.
    """
    if value is None:
        return "—"

    try:
        f = float(value)
    except Exception:
        return str(value)

    if abs(f - round(f)) < 1e-9:
        return f"{int(round(f)):,}"

    return f"{f:,.{digits}f}"


def integer(value: Any) -> str:
    """
    Format an integer-like value for display.
    """
    if value is None:
        return "—"

    try:
        return f"{int(value):,}"
    except Exception:
        return str(value)


def clean_label(value: Any) -> str:
    """
    Convert internal snake_case-ish labels into readable UI labels.

    This does not alter canonical dataset values for computation; it is only
    for display.
    """
    if value is None:
        return "Unknown"

    text = str(value).strip()
    if not text:
        return "Unknown"

    return text.replace("_", " ")


def feature_label(feature_name: str) -> str:
    """
    Human-readable label for a numeric feature.
    """
    return FEATURE_LABELS.get(feature_name, clean_label(feature_name).title())


def bucket_label(bucket_field: str) -> str:
    """
    Human-readable label for a quality bucket field.
    """
    return BUCKET_LABELS.get(bucket_field, clean_label(bucket_field).title())


def split_sort_key(split: str) -> tuple[int, str]:
    """
    Sort train/val/test in the natural order, then unknown splits alphabetically.
    """
    if split in SPLIT_ORDER:
        return (SPLIT_ORDER.index(split), split)
    return (len(SPLIT_ORDER), split)


def sorted_splits(values: list[str] | set[str]) -> list[str]:
    """
    Sort split names with train/val/test first.
    """
    return sorted(values, key=split_sort_key)


def counts_by_split_to_dataframe(
    counts_by_split: dict[str, dict[str, int | float]],
    *,
    category_name: str = "category",
    value_name: str = "count",
) -> pd.DataFrame:
    """
    Convert nested distribution data into a long dataframe.

    Input shape:
        {
            "train": {"forest": 10, "highway": 5},
            "val": {"forest": 2, "highway": 1},
            "test": {"forest": 3}
        }

    Output columns:
        split, <category_name>, <value_name>
    """
    rows: list[dict[str, Any]] = []

    for split, category_counts in counts_by_split.items():
        if not isinstance(category_counts, dict):
            continue

        for category, value in category_counts.items():
            rows.append(
                {
                    "split": split,
                    category_name: category,
                    value_name: value,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["split", category_name, value_name])

    df["split"] = pd.Categorical(
        df["split"],
        categories=SPLIT_ORDER,
        ordered=True,
    )
    df = df.sort_values(["split", category_name]).reset_index(drop=True)
    return df


def percentages_by_split_to_dataframe(
    percentages_by_split: dict[str, dict[str, int | float]],
    *,
    category_name: str = "category",
    value_name: str = "percentage",
) -> pd.DataFrame:
    """
    Convert nested percentage data into a long dataframe.

    The Lambda stores percentages as decimal fractions, not 0-100 values.
    """
    return counts_by_split_to_dataframe(
        percentages_by_split,
        category_name=category_name,
        value_name=value_name,
    )


def summary_stats_to_dataframe(summary: dict[str, Any]) -> pd.DataFrame:
    """
    Convert a numeric summary block into a dataframe.

    Expected input shape:
        {
            "img_height": {"count": ..., "min": ..., "p50": ...},
            "img_width": {...}
        }
    """
    rows: list[dict[str, Any]] = []

    for feature_name, stats in summary.items():
        if not isinstance(stats, dict):
            continue

        row = {"feature": feature_name, "feature_label": feature_label(feature_name)}
        row.update(stats)
        rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=[
                "feature",
                "feature_label",
                "count",
                "min",
                "max",
                "mean",
                "p10",
                "p25",
                "p50",
                "p75",
                "p90",
            ]
        )

    return pd.DataFrame(rows)


def histogram_to_dataframe(histogram: dict[str, Any]) -> pd.DataFrame:
    """
    Convert one histogram block into a dataframe.

    Expected Lambda shape:
        {
            "bin_edges": [0, 1, 2, 3],
            "counts": [10, 12, 8]
        }

    Output columns:
        bin_left, bin_right, bin_mid, count
    """
    edges = histogram.get("bin_edges", [])
    counts = histogram.get("counts", [])

    if not isinstance(edges, list) or not isinstance(counts, list):
        return pd.DataFrame(columns=["bin_left", "bin_right", "bin_mid", "count"])

    if len(edges) == 2 and len(counts) == 1:
        left = edges[0]
        right = edges[1]
        return pd.DataFrame(
            [
                {
                    "bin_left": left,
                    "bin_right": right,
                    "bin_mid": (float(left) + float(right)) / 2.0,
                    "count": counts[0],
                }
            ]
        )

    if len(edges) != len(counts) + 1:
        return pd.DataFrame(columns=["bin_left", "bin_right", "bin_mid", "count"])

    rows: list[dict[str, Any]] = []

    for idx, count in enumerate(counts):
        left = edges[idx]
        right = edges[idx + 1]

        try:
            mid = (float(left) + float(right)) / 2.0
        except Exception:
            mid = None

        rows.append(
            {
                "bin_left": left,
                "bin_right": right,
                "bin_mid": mid,
                "count": count,
            }
        )

    return pd.DataFrame(rows)


def nested_delta_block_to_dataframe(delta_block: dict[str, Any]) -> pd.DataFrame:
    """
    Convert a split comparison delta block into a dataframe.

    Expected input shape:
        {
            "deltas": {
                "forest": {
                    "train_percentage": 0.2,
                    "compare_percentage": 0.1,
                    "absolute_delta": 0.1
                }
            },
            "max_absolute_delta": 0.1,
            "mean_absolute_delta": 0.04
        }
    """
    deltas = delta_block.get("deltas", {})
    if not isinstance(deltas, dict):
        return pd.DataFrame(
            columns=[
                "category",
                "train_percentage",
                "compare_percentage",
                "absolute_delta",
            ]
        )

    rows: list[dict[str, Any]] = []

    for category, values in deltas.items():
        if not isinstance(values, dict):
            continue

        rows.append(
            {
                "category": category,
                "train_percentage": values.get("train_percentage", 0.0),
                "compare_percentage": values.get("compare_percentage", 0.0),
                "absolute_delta": values.get("absolute_delta", 0.0),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "category",
                "train_percentage",
                "compare_percentage",
                "absolute_delta",
            ]
        )

    return df.sort_values("absolute_delta", ascending=False).reset_index(drop=True)