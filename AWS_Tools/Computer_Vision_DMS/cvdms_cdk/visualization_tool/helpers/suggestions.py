"""
Deterministic diagnostic suggestions for the local CVDMS Dataset Viewer.

This module turns visualization artifacts into human-readable warnings,
observations, and recommendations.

The goal is not to be "smart AI" yet. The goal is a transparent rules-based
dataset reviewer that helps the user quickly notice:
- split size problems
- missing classes in val/test
- class/source distribution drift
- unresolved or inconsistent source split records
- suspicious quality bucket drift
"""

from dataclasses import dataclass
from typing import Any


_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Suggestion:
    severity: str  # "info", "warning", "critical", "success"
    title: str
    detail: str
    category: str

    def as_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "title": self.title,
            "detail": self.detail,
            "category": self.category,
        }


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
        return f
    except Exception:
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _get_nested_dict(obj: dict[str, Any], *keys: str) -> dict[str, Any]:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict):
            return {}
        cur = cur.get(key, {})
    return cur if isinstance(cur, dict) else {}


def _add_split_size_suggestions(
    suggestions: list[Suggestion],
    *,
    overview: dict[str, Any],
    min_val_or_test_count: int = 10,
    min_val_or_test_pct: float = 0.05,
) -> None:
    split_counts = overview.get("split_counts", {})
    split_percentages = overview.get("split_percentages", {})

    if not isinstance(split_counts, dict):
        split_counts = {}
    if not isinstance(split_percentages, dict):
        split_percentages = {}

    total = _as_int(overview.get("row_count", 0))

    if total <= 0:
        suggestions.append(
            Suggestion(
                severity="critical",
                title="Dataset version has no rows",
                detail="The selected dataset version reports zero rows. This version cannot be used for training.",
                category="overview",
            )
        )
        return

    train_count = _as_int(split_counts.get("train", 0))
    val_count = _as_int(split_counts.get("val", 0))
    test_count = _as_int(split_counts.get("test", 0))

    if train_count <= 0:
        suggestions.append(
            Suggestion(
                severity="critical",
                title="Training split is empty",
                detail="The train split has zero rows. A model cannot be trained from this dataset version.",
                category="overview",
            )
        )

    for split, count in (("val", val_count), ("test", test_count)):
        pct = _as_float(split_percentages.get(split, 0.0))

        if count <= 0:
            suggestions.append(
                Suggestion(
                    severity="critical",
                    title=f"{split} split is empty",
                    detail=f"The {split} split has zero rows. Evaluation on this split will not be possible.",
                    category="overview",
                )
            )
        elif count < min_val_or_test_count:
            suggestions.append(
                Suggestion(
                    severity="warning",
                    title=f"{split} split is very small",
                    detail=(
                        f"The {split} split has only {count:,} rows. Metrics from this split may be unstable."
                    ),
                    category="overview",
                )
            )
        elif pct < min_val_or_test_pct:
            suggestions.append(
                Suggestion(
                    severity="warning",
                    title=f"{split} split is a small fraction of the dataset",
                    detail=(
                        f"The {split} split is only {_pct(pct)} of the dataset. "
                        f"Consider whether this is enough for reliable evaluation."
                    ),
                    category="overview",
                )
            )


def _add_missing_category_suggestions(
    suggestions: list[Suggestion],
    *,
    distribution: dict[str, Any],
    category_label: str,
    category: str,
) -> None:
    counts_by_split = distribution.get("counts_by_split", {})
    if not isinstance(counts_by_split, dict):
        return

    train_counts = counts_by_split.get("train", {})
    if not isinstance(train_counts, dict):
        return

    train_categories = {k for k, v in train_counts.items() if _as_int(v) > 0}
    if not train_categories:
        return

    for split in ("val", "test"):
        split_counts = counts_by_split.get(split, {})
        if not isinstance(split_counts, dict):
            split_counts = {}

        split_categories = {k for k, v in split_counts.items() if _as_int(v) > 0}
        missing = sorted(train_categories - split_categories)

        if not missing:
            continue

        shown = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f", and {len(missing) - 8} more"

        severity = "critical" if category_label == "class" else "warning"

        suggestions.append(
            Suggestion(
                severity=severity,
                title=f"{split} is missing {len(missing)} {category_label}(s) present in train",
                detail=(
                    f"The {split} split has no examples for: {shown}{suffix}. "
                    f"This can make {split} metrics incomplete or misleading."
                ),
                category=category,
            )
        )


def _add_distribution_delta_suggestions(
    suggestions: list[Suggestion],
    *,
    split_comparison: dict[str, Any],
    block_name: str,
    display_name: str,
    category: str,
    warning_threshold: float = 0.10,
    critical_threshold: float = 0.20,
    top_k: int = 5,
) -> None:
    comparison_block = split_comparison.get(block_name, {})
    if not isinstance(comparison_block, dict):
        return

    for comparison_name in ("val_vs_train", "test_vs_train"):
        block = comparison_block.get(comparison_name, {})
        if not isinstance(block, dict):
            continue

        max_delta = _as_float(block.get("max_absolute_delta", 0.0))
        mean_delta = _as_float(block.get("mean_absolute_delta", 0.0))
        deltas = block.get("deltas", {})
        if not isinstance(deltas, dict):
            deltas = {}

        if max_delta < warning_threshold:
            continue

        split = "val" if comparison_name == "val_vs_train" else "test"
        severity = "critical" if max_delta >= critical_threshold else "warning"

        sorted_items = sorted(
            deltas.items(),
            key=lambda item: _as_float(
                item[1].get("absolute_delta", 0.0) if isinstance(item[1], dict) else 0.0
            ),
            reverse=True,
        )

        offenders: list[str] = []
        for name, values in sorted_items[:top_k]:
            if not isinstance(values, dict):
                continue
            delta = _as_float(values.get("absolute_delta", 0.0))
            train_pct = _as_float(values.get("train_percentage", 0.0))
            compare_pct = _as_float(values.get("compare_percentage", 0.0))
            offenders.append(
                f"{name} ({split}: {_pct(compare_pct)}, train: {_pct(train_pct)}, delta: {_pct(delta)})"
            )

        offender_text = "; ".join(offenders) if offenders else "No detailed offenders available."

        suggestions.append(
            Suggestion(
                severity=severity,
                title=f"{display_name} distribution drift in {split}",
                detail=(
                    f"The largest {display_name.lower()} percentage difference between {split} and train "
                    f"is {_pct(max_delta)}; mean absolute delta is {_pct(mean_delta)}. "
                    f"Largest differences: {offender_text}"
                ),
                category=category,
            )
        )


def _add_source_split_resolution_suggestions(
    suggestions: list[Suggestion],
    *,
    source_split_resolution: dict[str, Any],
    overview: dict[str, Any],
) -> None:
    honor_source_splits = overview.get("honor_source_splits")
    status_counts = source_split_resolution.get("source_split_status_counts_by_split", {})

    if not isinstance(status_counts, dict):
        return

    unresolved_total = 0
    inconsistent_total = 0
    resolved_total = 0

    for split in _SPLITS:
        counts = status_counts.get(split, {})
        if not isinstance(counts, dict):
            continue

        unresolved_total += _as_int(counts.get("unresolved", 0))
        inconsistent_total += _as_int(counts.get("inconsistent", 0))
        resolved_total += _as_int(counts.get("resolved", 0))

    if unresolved_total > 0:
        suggestions.append(
            Suggestion(
                severity="warning",
                title="Some rows have unresolved source splits",
                detail=(
                    f"{unresolved_total:,} row(s) have source_split_status='unresolved'. "
                    f"If source splits are important, inspect whether these rows should be excluded, "
                    f"reassigned, or traced back to missing provenance."
                ),
                category="source split resolution",
            )
        )

    if inconsistent_total > 0:
        suggestions.append(
            Suggestion(
                severity="critical",
                title="Some rows have inconsistent source split information",
                detail=(
                    f"{inconsistent_total:,} row(s) have source_split_status='inconsistent'. "
                    f"This means source provenance disagrees across one or more inputs for those rows."
                ),
                category="source split resolution",
            )
        )

    if honor_source_splits is True and resolved_total == 0:
        suggestions.append(
            Suggestion(
                severity="warning",
                title="Source splits are honored, but no resolved source splits were found",
                detail=(
                    "The dataset metadata says honor_source_splits=True, but the visualization artifact "
                    "does not show any resolved source split rows. Confirm that image_source_membership "
                    "contains source_split values."
                ),
                category="source split resolution",
            )
        )

    if unresolved_total == 0 and inconsistent_total == 0 and resolved_total > 0:
        suggestions.append(
            Suggestion(
                severity="success",
                title="Source split resolution looks healthy",
                detail=(
                    f"All {resolved_total:,} source split records represented in the visualization are resolved."
                ),
                category="source split resolution",
            )
        )


def _add_quality_bucket_suggestions(
    suggestions: list[Suggestion],
    *,
    quality_distribution: dict[str, Any],
    warning_threshold: float = 0.15,
    critical_threshold: float = 0.30,
) -> None:
    """
    Detect quality bucket drift by comparing bucket percentages in val/test to train.

    The Lambda currently stores only counts for quality buckets, not percentages,
    so this computes percentages locally from the artifact counts.
    """
    if not isinstance(quality_distribution, dict):
        return

    for bucket_field, block in quality_distribution.items():
        if not isinstance(block, dict):
            continue

        counts_by_split = block.get("counts_by_split", {})
        if not isinstance(counts_by_split, dict):
            continue

        train_counts = counts_by_split.get("train", {})
        if not isinstance(train_counts, dict):
            continue

        train_total = sum(_as_int(v) for v in train_counts.values())
        if train_total <= 0:
            continue

        train_pct = {
            bucket: _as_int(count) / train_total
            for bucket, count in train_counts.items()
            if _as_int(count) > 0
        }

        for split in ("val", "test"):
            split_counts = counts_by_split.get(split, {})
            if not isinstance(split_counts, dict):
                continue

            split_total = sum(_as_int(v) for v in split_counts.values())
            if split_total <= 0:
                continue

            split_pct = {
                bucket: _as_int(count) / split_total
                for bucket, count in split_counts.items()
                if _as_int(count) > 0
            }

            buckets = sorted(set(train_pct.keys()) | set(split_pct.keys()))
            if not buckets:
                continue

            max_bucket = None
            max_delta = 0.0
            train_value = 0.0
            split_value = 0.0

            for bucket in buckets:
                t = train_pct.get(bucket, 0.0)
                s = split_pct.get(bucket, 0.0)
                delta = abs(s - t)
                if delta > max_delta:
                    max_delta = delta
                    max_bucket = bucket
                    train_value = t
                    split_value = s

            if max_bucket is None or max_delta < warning_threshold:
                continue

            severity = "critical" if max_delta >= critical_threshold else "warning"
            pretty_field = bucket_field.replace("_bucket", "").replace("_", " ")

            suggestions.append(
                Suggestion(
                    severity=severity,
                    title=f"{pretty_field.title()} quality drift in {split}",
                    detail=(
                        f"Bucket '{max_bucket}' differs by {_pct(max_delta)} between {split} and train "
                        f"({split}: {_pct(split_value)}, train: {_pct(train_value)})."
                    ),
                    category="quality buckets",
                )
            )


def build_suggestions(bundle: Any) -> list[Suggestion]:
    """
    Build all suggestions for a loaded VisualizationBundle.

    The function accepts `Any` instead of importing VisualizationBundle to avoid
    a circular dependency in simple Streamlit execution contexts. It only relies
    on bundle.get(name).
    """
    overview = bundle.get("overview")
    class_distribution = bundle.get("class_distribution")
    source_distribution = bundle.get("source_distribution")
    source_split_resolution = bundle.get("source_split_resolution")
    quality_distribution = bundle.get("quality_distribution")
    split_comparison = bundle.get("split_comparison")

    suggestions: list[Suggestion] = []

    _add_split_size_suggestions(suggestions, overview=overview)

    _add_missing_category_suggestions(
        suggestions,
        distribution=class_distribution,
        category_label="class",
        category="class balance",
    )

    _add_missing_category_suggestions(
        suggestions,
        distribution=source_distribution,
        category_label="source",
        category="source balance",
    )

    _add_distribution_delta_suggestions(
        suggestions,
        split_comparison=split_comparison,
        block_name="class_comparison",
        display_name="Class",
        category="class balance",
        warning_threshold=0.10,
        critical_threshold=0.20,
    )

    _add_distribution_delta_suggestions(
        suggestions,
        split_comparison=split_comparison,
        block_name="source_comparison",
        display_name="Source",
        category="source balance",
        warning_threshold=0.10,
        critical_threshold=0.25,
    )

    _add_source_split_resolution_suggestions(
        suggestions,
        source_split_resolution=source_split_resolution,
        overview=overview,
    )

    _add_quality_bucket_suggestions(
        suggestions,
        quality_distribution=quality_distribution,
    )

    if not suggestions:
        suggestions.append(
            Suggestion(
                severity="success",
                title="No major dataset balance issues detected",
                detail=(
                    "The current rules did not find empty splits, major class/source drift, "
                    "source split resolution problems, or large quality bucket drift."
                ),
                category="overall",
            )
        )

    severity_rank = {
        "critical": 0,
        "warning": 1,
        "info": 2,
        "success": 3,
    }

    return sorted(
        suggestions,
        key=lambda s: (severity_rank.get(s.severity, 99), s.category, s.title),
    )