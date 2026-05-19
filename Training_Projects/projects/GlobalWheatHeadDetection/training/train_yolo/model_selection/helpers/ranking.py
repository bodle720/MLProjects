from helpers import sweep_settings as settings


def _to_float(value):
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_value(record: dict, metric_name: str) -> float | None:
    return _to_float(record.get(metric_name))


def _has_valid_selection_metric(record: dict) -> bool:
    return _metric_value(record, settings.SELECTION_METRIC) is not None


def _sort_key_for_metric(record: dict, metric_name: str):
    value = _metric_value(record, metric_name)
    if value is None:
        return float("-inf")

    return value


def sort_by_selection_metric(records: list[dict]) -> list[dict]:
    valid_records = [record for record in records if _has_valid_selection_metric(record)]

    return sorted(
        valid_records,
        key=lambda record: _sort_key_for_metric(record, settings.SELECTION_METRIC),
        reverse=True,
    )


def get_best_overall(records: list[dict]) -> dict | None:
    sorted_records = sort_by_selection_metric(records)

    if not sorted_records:
        return None

    best = dict(sorted_records[0])
    best["selection_type"] = "best_overall"
    best["selection_metric"] = settings.SELECTION_METRIC
    best["selection_metric_value"] = _metric_value(best, settings.SELECTION_METRIC)

    return best


def get_best_lightweight(records: list[dict]) -> dict | None:
    lightweight_records = [
        record for record in records
        if record.get("is_lightweight_candidate") is True
    ]

    sorted_records = sort_by_selection_metric(lightweight_records)

    if not sorted_records:
        return None

    best = dict(sorted_records[0])
    best["selection_type"] = "best_lightweight"
    best["selection_metric"] = settings.SELECTION_METRIC
    best["selection_metric_value"] = _metric_value(best, settings.SELECTION_METRIC)

    return best


def _higher_is_better_at_least_as_good(candidate: dict, other: dict, column: str) -> bool:
    candidate_value = _to_float(candidate.get(column))
    other_value = _to_float(other.get(column))

    if candidate_value is None or other_value is None:
        return True

    return other_value >= candidate_value


def _higher_is_better_strictly_better(candidate: dict, other: dict, column: str) -> bool:
    candidate_value = _to_float(candidate.get(column))
    other_value = _to_float(other.get(column))

    if candidate_value is None or other_value is None:
        return False

    return other_value > candidate_value


def _lower_is_better_at_least_as_good(candidate: dict, other: dict, column: str) -> bool:
    candidate_value = _to_float(candidate.get(column))
    other_value = _to_float(other.get(column))

    if candidate_value is None or other_value is None:
        return True

    return other_value <= candidate_value


def _lower_is_better_strictly_better(candidate: dict, other: dict, column: str) -> bool:
    candidate_value = _to_float(candidate.get(column))
    other_value = _to_float(other.get(column))

    if candidate_value is None or other_value is None:
        return False

    return other_value < candidate_value


def is_dominated(candidate: dict, other: dict) -> bool:
    if candidate is other:
        return False

    at_least_as_good_checks = []
    strictly_better_checks = []

    for column in settings.PARETO_HIGHER_IS_BETTER:
        at_least_as_good_checks.append(
            _higher_is_better_at_least_as_good(candidate, other, column)
        )
        strictly_better_checks.append(
            _higher_is_better_strictly_better(candidate, other, column)
        )

    for column in settings.PARETO_LOWER_IS_BETTER:
        at_least_as_good_checks.append(
            _lower_is_better_at_least_as_good(candidate, other, column)
        )
        strictly_better_checks.append(
            _lower_is_better_strictly_better(candidate, other, column)
        )

    return all(at_least_as_good_checks) and any(strictly_better_checks)


def get_pareto_candidates(records: list[dict]) -> list[dict]:
    valid_records = [record for record in records if _has_valid_selection_metric(record)]

    pareto_records = []
    for candidate in valid_records:
        dominated = False

        for other in valid_records:
            if is_dominated(candidate, other):
                dominated = True
                break

        if not dominated:
            pareto_record = dict(candidate)
            pareto_record["selection_type"] = "pareto_candidate"
            pareto_record["selection_metric"] = settings.SELECTION_METRIC
            pareto_record["selection_metric_value"] = _metric_value(
                pareto_record,
                settings.SELECTION_METRIC,
            )
            pareto_records.append(pareto_record)

    return sorted(
        pareto_records,
        key=lambda record: (
            _sort_key_for_metric(record, settings.SELECTION_METRIC),
            -(_to_float(record.get("model_file_size_mb")) or float("inf")),
            -(_to_float(record.get("eval_runtime_seconds")) or float("inf")),
        ),
        reverse=True,
    )


def add_rank_columns(records: list[dict]) -> list[dict]:
    sorted_records = sort_by_selection_metric(records)
    ranked_by_identity = {}

    for rank, record in enumerate(sorted_records, start=1):
        identity = _record_identity(record)
        ranked_by_identity[identity] = rank

    enriched = []
    for record in records:
        new_record = dict(record)
        identity = _record_identity(record)
        new_record["validation_rank"] = ranked_by_identity.get(identity)
        enriched.append(new_record)

    return enriched


def _record_identity(record: dict) -> tuple:
    return (
        record.get("run_id"),
        record.get("best_pt_local_path"),
        record.get("imgsz"),
        record.get("iou"),
        record.get("max_det"),
    )


def summarize_rankings(records: list[dict]) -> dict:
    ranked_records = add_rank_columns(records)

    best_overall = get_best_overall(ranked_records)
    best_lightweight = get_best_lightweight(ranked_records)
    pareto_candidates = get_pareto_candidates(ranked_records)

    return {
        "ranked_records": ranked_records,
        "best_overall": best_overall,
        "best_lightweight": best_lightweight,
        "pareto_candidates": pareto_candidates,
    }