from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class YoloSplitRows:
    train: list[dict[str, Any]]
    val: list[dict[str, Any]]
    test: list[dict[str, Any]]


def get_image_id(row: dict[str, Any], source_name: str) -> str:
    image_id = row.get("image_id")

    if not isinstance(image_id, str) or not image_id.strip():
        raise ValueError(f"{source_name} row is missing a valid image_id: {row}")

    return image_id.strip()


def collect_image_ids(rows: list[dict[str, Any]], source_name: str) -> set[str]:
    return {
        get_image_id(row, source_name)
        for row in rows
    }


def validate_no_split_overlap(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> None:
    train_ids = collect_image_ids(train_rows, "YOLO train")
    val_ids = collect_image_ids(val_rows, "YOLO val")
    test_ids = collect_image_ids(test_rows, "YOLO test")

    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids

    if train_val_overlap:
        raise ValueError(
            "Image leakage detected between YOLO train and YOLO val. "
            f"Examples: {sorted(train_val_overlap)[:10]}"
        )

    if train_test_overlap:
        raise ValueError(
            "Image leakage detected between YOLO train and YOLO test. "
            f"Examples: {sorted(train_test_overlap)[:10]}"
        )

    if val_test_overlap:
        raise ValueError(
            "Image leakage detected between YOLO val and YOLO test. "
            f"Examples: {sorted(val_test_overlap)[:10]}"
        )


def build_preserved_yolo_split_rows(
    cvdms_train_rows: list[dict[str, Any]],
    cvdms_val_rows: list[dict[str, Any]],
    cvdms_test_rows: list[dict[str, Any]],
) -> YoloSplitRows:
    """
    Apply the Project 3 split policy:

        CVDMS train.jsonl -> YOLO train
        CVDMS val.jsonl   -> YOLO val
        CVDMS test.jsonl  -> YOLO test

    This preserves the official Global Wheat Head Detection 2021 source splits.
    """
    validate_no_split_overlap(
        train_rows=cvdms_train_rows,
        val_rows=cvdms_val_rows,
        test_rows=cvdms_test_rows,
    )

    return YoloSplitRows(
        train=cvdms_train_rows,
        val=cvdms_val_rows,
        test=cvdms_test_rows,
    )


def summarize_split_rows(split_rows: YoloSplitRows) -> dict[str, int]:
    return {
        "policy": "preserve_cvdms_train_val_test",
        "train_rows": len(split_rows.train),
        "val_rows": len(split_rows.val),
        "test_rows": len(split_rows.test),
        "total_rows": len(split_rows.train) + len(split_rows.val) + len(split_rows.test),
        "train_unique_images": len(collect_image_ids(split_rows.train, "YOLO train")),
        "val_unique_images": len(collect_image_ids(split_rows.val, "YOLO val")),
        "test_unique_images": len(collect_image_ids(split_rows.test, "YOLO test")),
    }