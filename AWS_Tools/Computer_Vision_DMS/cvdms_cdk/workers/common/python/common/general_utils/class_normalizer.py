import re
import unicodedata
from typing import Any

_CLASS_NAME_MAX_LEN = 50
_RESERVED_CLASS_NAMES_LC = {"bg", "background"}

def canonicalize_class_name(
    value: Any,
    *,
    field_name: str,
    allow_background: bool = False,
    max_length: int = _CLASS_NAME_MAX_LEN,
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")

    s = value.strip().lower()
    if s == "":
        raise ValueError(f"{field_name} cannot be empty after stripping")

    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")

    # remove apostrophes so "animal's" -> "animals"
    s = re.sub(r"[\'’`]+", "", s)

    # any run of non-alphanumeric chars becomes one underscore
    s = re.sub(r"[^a-z0-9]+", "_", s)

    s = re.sub(r"_+", "_", s).strip("_")

    if s == "":
        raise ValueError(f"{field_name} became empty after normalization")

    s = s[:max_length].rstrip("_")
    if s == "":
        raise ValueError(f"{field_name} became empty after length truncation")

    if not allow_background and s in _RESERVED_CLASS_NAMES_LC:
        raise ValueError(f"{field_name} uses reserved class name: {s}")

    return s