from datetime import datetime
from typing import Any, Literal

_ALLOWED_LABEL_TYPES = {
    "single-label",
    "multi-label",
    "object-detection",
    "semantic-segmentation",
    "instance-segmentation",
}

_ALLOWED_SPLIT_STRATEGIES = {
    "stratified_v1",
}

_ALLOWED_OPERATIONS = {
    "add",
    "remove",
}

_ALLOWED_SPLIT_APPROACHES = {
    "maintain",
    "rebalance",
}

_ALLOWED_LIGHTING_BUCKETS = {
    "night",
    "low_light",
    "normal",
    "bright",
    "glare",
}

_ALLOWED_BLUR_BUCKETS = {
    "sharp",
    "mild_blur",
    "blurry",
}

_ALLOWED_CONTRAST_BUCKETS = {
    "low",
    "medium",
    "high",
}

_ALLOWED_COLOR_BUCKETS = {
    "low",
    "medium",
    "high",
}

_ALLOWED_SELECTION_CONFIG_KEYS = {
    "allowed_classes",
    "allowed_sources",
    "upload_date_range",
    "width_range",
    "height_range",
    "lighting_buckets",
    "blur_buckets",
    "contrast_buckets",
    "color_buckets",
}

#############################################################
# The validation helper functions
#############################################################
def validate_dataset_id(dataset_id: str) -> str:
    """
    Validate dataset_id.

    Rules:
    - required
    - string
    - 1..128 chars
    - lowercase letters, digits, hyphens only
    - must start/end with alphanumeric
    """
    if not isinstance(dataset_id, str):
        raise TypeError("dataset_id must be a string.")

    dataset_id = dataset_id.strip().lower()

    if not dataset_id:
        raise ValueError("dataset_id must not be empty.")

    if len(dataset_id) > 128:
        raise ValueError("dataset_id must be at most 128 characters long.")

    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789-")
    if any(ch not in allowed for ch in dataset_id):
        raise ValueError(
            "dataset_id may contain only lowercase letters, digits, and hyphens."
        )

    if not dataset_id[0].isalnum() or not dataset_id[-1].isalnum():
        raise ValueError(
            "dataset_id must start and end with a lowercase letter or digit."
        )

    if "--" in dataset_id:
        raise ValueError("dataset_id must not contain consecutive hyphens.")

    return dataset_id

def validate_label_type(label_type: str) -> str:
    if not isinstance(label_type, str):
        raise TypeError("label_type must be a string.")

    label_type = label_type.strip()

    if label_type not in _ALLOWED_LABEL_TYPES:
        raise ValueError(
            f"label_type must be one of: {sorted(_ALLOWED_LABEL_TYPES)}"
        )

    return label_type

def validate_description(description: str) -> str:
    if not isinstance(description, str):
        raise TypeError("description must be a string.")

    description = description.strip()

    if not description:
        raise ValueError("description must not be empty.")

    if len(description) > 500:
        raise ValueError("description must be at most 500 characters long.")

    return description

def validate_optional_description(description: str | None) -> str | None:
    if description is None:
        return None
    return validate_description(description)

def validate_split_strategy_name(split_strategy_name: str) -> str:
    if not isinstance(split_strategy_name, str):
        raise TypeError("split_strategy_name must be a string.")

    split_strategy_name = split_strategy_name.strip()

    if split_strategy_name not in _ALLOWED_SPLIT_STRATEGIES:
        raise ValueError(
            f"split_strategy_name must be one of: {sorted(_ALLOWED_SPLIT_STRATEGIES)}"
        )

    return split_strategy_name

def validate_optional_split_strategy_name(
    split_strategy_name: str | None,
) -> str | None:

    if split_strategy_name is None:
        return None

    return validate_split_strategy_name(split_strategy_name)

def validate_operation(operation: str) -> str:
    if not isinstance(operation, str):
        raise TypeError("operation must be a string.")

    operation = operation.strip().lower()

    if operation not in _ALLOWED_OPERATIONS:
        raise ValueError(
            f"operation must be one of: {sorted(_ALLOWED_OPERATIONS)}"
        )

    return operation

def validate_split_approach(split_approach: str) -> str:
    if not isinstance(split_approach, str):
        raise TypeError("split_approach must be a string.")

    split_approach = split_approach.strip().lower()

    if split_approach not in _ALLOWED_SPLIT_APPROACHES:
        raise ValueError(
            f"split_approach must be one of: {sorted(_ALLOWED_SPLIT_APPROACHES)}"
        )

    return split_approach

def _validate_nonempty_string_list(
    *,
    name: str,
    value: Any,
    normalize: bool,
) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list of strings.")

    if not value:
        raise ValueError(f"{name} must not be an empty list.")

    cleaned: list[str] = []
    seen: set[str] = set()

    for item in value:
        if not isinstance(item, str):
            raise TypeError(f"Every item in {name} must be a string.")

        item_clean = item.strip()
        if normalize:
            item_clean = item_clean.lower()

        if not item_clean:
            raise ValueError(f"{name} must not contain empty strings.")

        if item_clean not in seen:
            seen.add(item_clean)
            cleaned.append(item_clean)

    return cleaned

def _validate_enum_string_list(
    *,
    name: str,
    value: Any,
    allowed_values: set[str],
) -> list[str]:
    cleaned = _validate_nonempty_string_list(
        name=name,
        value=value,
        normalize=False,
    )

    invalid = [item for item in cleaned if item not in allowed_values]
    if invalid:
        raise ValueError(
            f"{name} contains invalid values {invalid}. Allowed values: {sorted(allowed_values)}"
        )

    return cleaned

def _validate_date_range(*, name: str, value: Any) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a 2-element list of ISO date strings.")

    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly 2 items: [start, end].")

    start_raw, end_raw = value

    if not isinstance(start_raw, str) or not isinstance(end_raw, str):
        raise TypeError(f"{name} values must both be strings in YYYY-MM-DD format.")

    start = start_raw.strip()
    end = end_raw.strip()

    if not start or not end:
        raise ValueError(f"{name} values must not be empty.")

    try:
        start_dt = datetime.strptime(start, "%Y-%m-%d").date()
        end_dt = datetime.strptime(end, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(
            f"{name} values must be valid ISO dates in YYYY-MM-DD format."
        ) from exc

    if start_dt > end_dt:
        raise ValueError(f"{name} start date must be <= end date.")

    return [start, end]

def _validate_int_range(
    *,
    name: str,
    value: Any,
    min_allowed: int,
) -> list[int]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a 2-element list of integers.")

    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly 2 items: [min, max].")

    low, high = value

    if type(low) is not int or type(high) is not int:
        raise TypeError(f"{name} values must both be integers.")

    if low < min_allowed or high < min_allowed:
        raise ValueError(f"{name} values must both be >= {min_allowed}.")

    if low > high:
        raise ValueError(f"{name} minimum must be <= maximum.")

    return [low, high]

def validate_selection_config(selection_config: dict[str, Any]) -> dict[str, Any]:
    """
    Validate selection_config for dataset create/update selection.

    Required:
    - allowed_classes

    Optional:
    - allowed_sources
    - upload_date_range
    - width_range
    - height_range
    - lighting_buckets
    - blur_buckets
    - contrast_buckets
    - color_buckets
    """
    if not isinstance(selection_config, dict):
        raise TypeError("selection_config must be a dict.")

    unknown_keys = set(selection_config.keys()) - _ALLOWED_SELECTION_CONFIG_KEYS
    if unknown_keys:
        raise ValueError(
            f"selection_config contains unsupported keys: {sorted(unknown_keys)}"
        )

    if "allowed_classes" not in selection_config:
        raise ValueError("selection_config must include required key: 'allowed_classes'.")

    validated: dict[str, Any] = {}

    validated["allowed_classes"] = _validate_nonempty_string_list(
        name="allowed_classes",
        value=selection_config["allowed_classes"],
        normalize=True,
    )

    if "allowed_sources" in selection_config:
        validated["allowed_sources"] = _validate_nonempty_string_list(
            name="allowed_sources",
            value=selection_config["allowed_sources"],
            normalize=False,
        )

    if "upload_date_range" in selection_config:
        validated["upload_date_range"] = _validate_date_range(
            name="upload_date_range",
            value=selection_config["upload_date_range"],
        )

    if "width_range" in selection_config:
        validated["width_range"] = _validate_int_range(
            name="width_range",
            value=selection_config["width_range"],
            min_allowed=1,
        )

    if "height_range" in selection_config:
        validated["height_range"] = _validate_int_range(
            name="height_range",
            value=selection_config["height_range"],
            min_allowed=1,
        )

    if "lighting_buckets" in selection_config:
        validated["lighting_buckets"] = _validate_enum_string_list(
            name="lighting_buckets",
            value=selection_config["lighting_buckets"],
            allowed_values=_ALLOWED_LIGHTING_BUCKETS,
        )

    if "blur_buckets" in selection_config:
        validated["blur_buckets"] = _validate_enum_string_list(
            name="blur_buckets",
            value=selection_config["blur_buckets"],
            allowed_values=_ALLOWED_BLUR_BUCKETS,
        )

    if "contrast_buckets" in selection_config:
        validated["contrast_buckets"] = _validate_enum_string_list(
            name="contrast_buckets",
            value=selection_config["contrast_buckets"],
            allowed_values=_ALLOWED_CONTRAST_BUCKETS,
        )

    if "color_buckets" in selection_config:
        validated["color_buckets"] = _validate_enum_string_list(
            name="color_buckets",
            value=selection_config["color_buckets"],
            allowed_values=_ALLOWED_COLOR_BUCKETS,
        )

    return validated

#############################################################
# The four main validator entrypoints
#############################################################
def validate_create_dataset_inputs(
    *,
    dataset_id: str,
    label_type: str,
    description: str,
    selection_config: dict[str, Any],
    split_strategy_name: str,
) -> dict[str, Any]:
    """
    Top-level validator for DatasetClient.create_dataset(...).
    """
    return {
        "dataset_id": validate_dataset_id(dataset_id),
        "label_type": validate_label_type(label_type),
        "description": validate_description(description),
        "selection_config": validate_selection_config(selection_config),
        "split_strategy_name": validate_split_strategy_name(split_strategy_name),
    }

def validate_update_dataset_inputs(
    *,
    dataset_id: str,
    operation: Literal["add", "remove"] | str,
    selection_config: dict[str, Any],
    split_approach: Literal["maintain", "rebalance"] | str = "maintain",
    split_strategy_name: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """
    Top-level validator for DatasetClient.update_dataset(...).

    Rules:
    - dataset_id is required
    - operation must be add/remove
    - selection_config is validated the same way as create
    - split_approach must be maintain/rebalance
    - if split_approach == 'rebalance', split_strategy_name is required
    - if split_approach == 'maintain', split_strategy_name is optional
    - description is optional
    """
    validated_dataset_id = validate_dataset_id(dataset_id)
    validated_operation = validate_operation(operation)
    validated_selection_config = validate_selection_config(selection_config)
    validated_split_approach = validate_split_approach(split_approach)
    validated_description = validate_optional_description(description)
    validated_split_strategy_name = validate_optional_split_strategy_name(split_strategy_name)

    if validated_split_approach == "rebalance" and validated_split_strategy_name is None:
        raise ValueError(
            "split_strategy_name is required when split_approach='rebalance'."
        )
    elif validated_split_approach == "maintain" and validated_split_strategy_name is not None:
        raise ValueError(
            "split_strategy_name must be None when split_approach='maintain'."
        )

    return {
        "dataset_id": validated_dataset_id,
        "operation": validated_operation,
        "selection_config": validated_selection_config,
        "split_approach": validated_split_approach,
        "split_strategy_name": validated_split_strategy_name,
        "description": validated_description,
    }

def validate_delete_dataset_inputs(
    *,
    dataset_id: str
) -> dict[str, Any]:
    """
    Top-level validator for DatasetClient.delete_dataset_all_versions(...).

    Rules:
    - dataset_id is required
    """
    validated_dataset_id = validate_dataset_id(dataset_id)

    return {
        "dataset_id": validated_dataset_id
    }

def validate_get_dataset_inputs(
    *,
    dataset_id: str
) -> dict[str, Any]:
    """
    Top-level validator for DatasetClient.get_dataset(...).

    Rules:
    - dataset_id is required
    """
    validated_dataset_id = validate_dataset_id(dataset_id)

    return {
        "dataset_id": validated_dataset_id
    }