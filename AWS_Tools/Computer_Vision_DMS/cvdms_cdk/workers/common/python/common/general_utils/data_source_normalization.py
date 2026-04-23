import re
import unicodedata
from typing import Any

_DATA_SOURCE_MAX_LEN = 35

def canonicalize_data_source(
    value: Any,
    *,
    field_name: str = "data_source",
    max_length: int = _DATA_SOURCE_MAX_LEN,
) -> str:
    """
    Normalize a user-provided data source identifier into a compact,
    lowercase, ASCII-safe token for storage and filtering.

    Rules:
    - must be a string
    - strip leading/trailing whitespace
    - lowercase
    - Unicode NFKD normalize, then drop non-ASCII chars
    - remove apostrophes
    - remove all non-alphanumeric characters
    - must remain non-empty
    - truncate to max_length
    - must remain non-empty after truncation

    Examples:
    - "COCO 2017" -> "coco2017"
    - "BigEarthNet v2" -> "bigearthnetv2"
    - " EuroSAT " -> "eurosat"
    - "my-custom dataset!" -> "mycustomdataset"
    """
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")

    s = value.strip().lower()
    if s == "":
        raise ValueError(f"{field_name} cannot be empty after stripping")

    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")

    # remove apostrophes so "dataset's" -> "datasets"
    s = re.sub(r"[\'’`]+", "", s)

    # keep only lowercase letters and digits
    s = re.sub(r"[^a-z0-9]+", "", s)

    if s == "":
        raise ValueError(f"{field_name} became empty after normalization")

    s = s[:max_length]

    if s == "":
        raise ValueError(f"{field_name} became empty after length truncation")

    return s