from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha1
from typing import Any

_SPLITS: tuple[str, ...] = ("train", "val", "test")
_SPLIT_RATIOS: dict[str, float] = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

# Objective weights, in priority order.
# These are intentionally simple and interpretable for v1.
_CLASS_WEIGHT = 8.0
_SIZE_WEIGHT = 4.0
_SOURCE_WEIGHT = 2.0
_LIGHTING_WEIGHT = 1.5
_BLUR_WEIGHT = 1.0
_CONTRAST_WEIGHT = 0.75
_COLOR_WEIGHT = 0.5

# Soft overflow penalty for exceeding target split size.
_OVERFLOW_WEIGHT = 6.0

@dataclass(frozen=True)
class GroupSummary:
    group_key: str
    rows: list[dict[str, Any]]
    row_count: int
    classes: tuple[str, ...]
    class_counts: dict[str, int]
    source_counts: dict[str, int]
    lighting_counts: dict[str, int]
    blur_counts: dict[str, int]
    contrast_counts: dict[str, int]
    color_counts: dict[str, int]
    rarity_score: float
    stable_tiebreak: str

def assign_splits_stratified_v1(*, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Deterministic greedy multi-objective split assignment.

    Primary goals:
    1. keep leakage-aware duplicate-content groups together
    2. preserve class proportions across train/val/test
    3. roughly preserve split sizes
    4. secondarily balance source and image-condition buckets

    Input:
        candidates: image-level candidate rows already normalized by Athena utilities

    Output:
        same rows, with 'split' attached
    """
    if not candidates:
        return []

    _validate_candidates(candidates)

    groups = _build_groups(candidates)
    overall = _compute_overall_counts(candidates)

    target_total_rows = _compute_target_counter(
        total=sum(1 for _ in candidates),
        ratios=_SPLIT_RATIOS,
    )

    target_class_counts = _compute_target_nested_counts(
        overall["class_counts"],
        ratios=_SPLIT_RATIOS,
    )
    target_source_counts = _compute_target_nested_counts(
        overall["source_counts"],
        ratios=_SPLIT_RATIOS,
    )
    target_lighting_counts = _compute_target_nested_counts(
        overall["lighting_counts"],
        ratios=_SPLIT_RATIOS,
    )
    target_blur_counts = _compute_target_nested_counts(
        overall["blur_counts"],
        ratios=_SPLIT_RATIOS,
    )
    target_contrast_counts = _compute_target_nested_counts(
        overall["contrast_counts"],
        ratios=_SPLIT_RATIOS,
    )
    target_color_counts = _compute_target_nested_counts(
        overall["color_counts"],
        ratios=_SPLIT_RATIOS,
    )

    assigned_total_rows: Counter[str] = Counter()
    assigned_class_counts: dict[str, Counter[str]] = {s: Counter() for s in _SPLITS}
    assigned_source_counts: dict[str, Counter[str]] = {s: Counter() for s in _SPLITS}
    assigned_lighting_counts: dict[str, Counter[str]] = {s: Counter() for s in _SPLITS}
    assigned_blur_counts: dict[str, Counter[str]] = {s: Counter() for s in _SPLITS}
    assigned_contrast_counts: dict[str, Counter[str]] = {s: Counter() for s in _SPLITS}
    assigned_color_counts: dict[str, Counter[str]] = {s: Counter() for s in _SPLITS}

    group_to_split: dict[str, str] = {}

    ordered_groups = sorted(
        groups,
        key=lambda g: (
            -g.rarity_score,
            -len(g.classes),
            -g.row_count,
            g.stable_tiebreak,
        ),
    )

    for group in ordered_groups:
        best_split = min(
            _SPLITS,
            key=lambda split: _score_group_for_split(
                group=group,
                split=split,
                assigned_total_rows=assigned_total_rows,
                assigned_class_counts=assigned_class_counts,
                assigned_source_counts=assigned_source_counts,
                assigned_lighting_counts=assigned_lighting_counts,
                assigned_blur_counts=assigned_blur_counts,
                assigned_contrast_counts=assigned_contrast_counts,
                assigned_color_counts=assigned_color_counts,
                target_total_rows=target_total_rows,
                target_class_counts=target_class_counts,
                target_source_counts=target_source_counts,
                target_lighting_counts=target_lighting_counts,
                target_blur_counts=target_blur_counts,
                target_contrast_counts=target_contrast_counts,
                target_color_counts=target_color_counts,
            ),
        )

        group_to_split[group.group_key] = best_split

        assigned_total_rows[best_split] += group.row_count
        assigned_class_counts[best_split].update(group.class_counts)
        assigned_source_counts[best_split].update(group.source_counts)
        assigned_lighting_counts[best_split].update(group.lighting_counts)
        assigned_blur_counts[best_split].update(group.blur_counts)
        assigned_contrast_counts[best_split].update(group.contrast_counts)
        assigned_color_counts[best_split].update(group.color_counts)

    return _attach_split_to_rows(candidates=candidates, group_to_split=group_to_split)

def _validate_candidates(candidates: list[dict[str, Any]]) -> None:
    required_fields = {"image_id", "dataset_label_type", "classes_present"}
    for idx, row in enumerate(candidates):
        missing = [field for field in required_fields if field not in row]
        if missing:
            raise ValueError(f"Candidate row {idx} missing required fields: {missing}")

        if not row["image_id"]:
            raise ValueError(f"Candidate row {idx} has empty image_id")

        if not isinstance(row["classes_present"], list):
            raise TypeError(
                f"Candidate row {idx} field classes_present must be list[str], "
                f"got {type(row['classes_present']).__name__}"
            )

        # classes_present should never be empty by this point for a selected candidate.
        if len(row["classes_present"]) == 0:
            raise ValueError(f"Candidate row {idx} has empty classes_present")

def _build_groups(candidates: list[dict[str, Any]]) -> list[GroupSummary]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in candidates:
        group_key = _group_key_for_row(row)
        grouped[group_key].append(row)

    overall_class_freq = Counter()
    for row in candidates:
        overall_class_freq.update(_deduped_strings(row.get("classes_present", [])))

    groups: list[GroupSummary] = []

    for group_key, rows in grouped.items():
        class_counts = Counter()
        source_counts = Counter()
        lighting_counts = Counter()
        blur_counts = Counter()
        contrast_counts = Counter()
        color_counts = Counter()

        class_union: set[str] = set()

        for row in rows:
            row_classes = _deduped_strings(row.get("classes_present", []))
            class_union.update(row_classes)
            class_counts.update(row_classes)

            if row.get("data_source"):
                source_counts[str(row["data_source"])] += 1
            if row.get("lighting_bucket"):
                lighting_counts[str(row["lighting_bucket"])] += 1
            if row.get("blur_bucket"):
                blur_counts[str(row["blur_bucket"])] += 1
            if row.get("contrast_bucket"):
                contrast_counts[str(row["contrast_bucket"])] += 1
            if row.get("color_bucket"):
                color_counts[str(row["color_bucket"])] += 1

        # Higher rarity score => assign earlier.
        rarity_score = 0.0
        for class_name in class_union:
            freq = overall_class_freq[class_name]
            if freq > 0:
                rarity_score += 1.0 / freq

        stable_tiebreak = sha1(group_key.encode("utf-8")).hexdigest()

        groups.append(
            GroupSummary(
                group_key=group_key,
                rows=rows,
                row_count=len(rows),
                classes=tuple(sorted(class_union)),
                class_counts=dict(class_counts),
                source_counts=dict(source_counts),
                lighting_counts=dict(lighting_counts),
                blur_counts=dict(blur_counts),
                contrast_counts=dict(contrast_counts),
                color_counts=dict(color_counts),
                rarity_score=rarity_score,
                stable_tiebreak=stable_tiebreak,
            )
        )

    return groups

def _group_key_for_row(row: dict[str, Any]) -> str:
    sha256_hash = row.get("sha256_hash")
    if sha256_hash:
        return f"sha256:{sha256_hash}"
    return f"image:{row['image_id']}"

def _compute_overall_counts(candidates: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    class_counts = Counter()
    source_counts = Counter()
    lighting_counts = Counter()
    blur_counts = Counter()
    contrast_counts = Counter()
    color_counts = Counter()

    for row in candidates:
        class_counts.update(_deduped_strings(row.get("classes_present", [])))

        if row.get("data_source"):
            source_counts[str(row["data_source"])] += 1
        if row.get("lighting_bucket"):
            lighting_counts[str(row["lighting_bucket"])] += 1
        if row.get("blur_bucket"):
            blur_counts[str(row["blur_bucket"])] += 1
        if row.get("contrast_bucket"):
            contrast_counts[str(row["contrast_bucket"])] += 1
        if row.get("color_bucket"):
            color_counts[str(row["color_bucket"])] += 1

    return {
        "class_counts": class_counts,
        "source_counts": source_counts,
        "lighting_counts": lighting_counts,
        "blur_counts": blur_counts,
        "contrast_counts": contrast_counts,
        "color_counts": color_counts,
    }

def _compute_target_counter(*, total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {split: total * ratios[split] for split in _SPLITS}
    floors = {split: int(raw[split]) for split in _SPLITS}
    remainder = total - sum(floors.values())

    frac_order = sorted(
        _SPLITS,
        key=lambda split: (raw[split] - floors[split], split),
        reverse=True,
    )

    result = dict(floors)
    for split in frac_order[:remainder]:
        result[split] += 1

    return result

def _compute_target_nested_counts(
    overall_counts: Counter[str],
    ratios: dict[str, float],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {split: {} for split in _SPLITS}
    for key, total in overall_counts.items():
        targets = _compute_target_counter(total=total, ratios=ratios)
        for split in _SPLITS:
            result[split][key] = targets[split]
    return result

def _score_group_for_split(
    *,
    group: GroupSummary,
    split: str,
    assigned_total_rows: Counter[str],
    assigned_class_counts: dict[str, Counter[str]],
    assigned_source_counts: dict[str, Counter[str]],
    assigned_lighting_counts: dict[str, Counter[str]],
    assigned_blur_counts: dict[str, Counter[str]],
    assigned_contrast_counts: dict[str, Counter[str]],
    assigned_color_counts: dict[str, Counter[str]],
    target_total_rows: dict[str, int],
    target_class_counts: dict[str, dict[str, int]],
    target_source_counts: dict[str, dict[str, int]],
    target_lighting_counts: dict[str, dict[str, int]],
    target_blur_counts: dict[str, dict[str, int]],
    target_contrast_counts: dict[str, dict[str, int]],
    target_color_counts: dict[str, dict[str, int]],
) -> float:
    score = 0.0

    # Primary: class-balance penalty
    score += _CLASS_WEIGHT * _dimension_penalty(
        current_counts=assigned_class_counts[split],
        addition_counts=group.class_counts,
        target_counts=target_class_counts[split],
    )

    # Secondary: overall split-size penalty
    current_total = assigned_total_rows[split]
    projected_total = current_total + group.row_count
    target_total = target_total_rows[split]

    score += _SIZE_WEIGHT * abs(projected_total - target_total)

    overflow = max(0, projected_total - target_total)
    score += _OVERFLOW_WEIGHT * overflow

    # Tertiary: source and image-condition balancing
    score += _SOURCE_WEIGHT * _dimension_penalty(
        current_counts=assigned_source_counts[split],
        addition_counts=group.source_counts,
        target_counts=target_source_counts[split],
    )
    score += _LIGHTING_WEIGHT * _dimension_penalty(
        current_counts=assigned_lighting_counts[split],
        addition_counts=group.lighting_counts,
        target_counts=target_lighting_counts[split],
    )
    score += _BLUR_WEIGHT * _dimension_penalty(
        current_counts=assigned_blur_counts[split],
        addition_counts=group.blur_counts,
        target_counts=target_blur_counts[split],
    )
    score += _CONTRAST_WEIGHT * _dimension_penalty(
        current_counts=assigned_contrast_counts[split],
        addition_counts=group.contrast_counts,
        target_counts=target_contrast_counts[split],
    )
    score += _COLOR_WEIGHT * _dimension_penalty(
        current_counts=assigned_color_counts[split],
        addition_counts=group.color_counts,
        target_counts=target_color_counts[split],
    )

    # Deterministic tie-break bias: very tiny, stable.
    score += _stable_small_bias(group.group_key, split)

    return score

def _dimension_penalty(
    *,
    current_counts: Counter[str],
    addition_counts: dict[str, int],
    target_counts: dict[str, int],
) -> float:
    """
    Penalize projected distance from target, with extra penalty for overshooting.
    """
    penalty = 0.0

    for key, add in addition_counts.items():
        current = current_counts.get(key, 0)
        projected = current + add
        target = target_counts.get(key, 0)

        base_distance = abs(projected - target)
        overshoot = max(0, projected - target)

        penalty += base_distance + (2.0 * overshoot)

    return penalty

def _attach_split_to_rows(
    *,
    candidates: list[dict[str, Any]],
    group_to_split: dict[str, str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    for row in candidates:
        group_key = _group_key_for_row(row)
        split = group_to_split[group_key]
        out.append({**row, "split": split})

    return out

def _deduped_strings(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    for value in values:
        text = str(value).strip()
        if not text:
            continue
        if text not in seen:
            seen.add(text)
            out.append(text)

    return out

def _stable_small_bias(group_key: str, split: str) -> float:
    """
    Tiny deterministic perturbation so ties break reproducibly.
    """
    digest = sha1(f"{group_key}|{split}".encode("utf-8")).hexdigest()
    # convert a few hex chars into a tiny fraction in [0, 1)
    return int(digest[:8], 16) / 16**8 / 1000.0