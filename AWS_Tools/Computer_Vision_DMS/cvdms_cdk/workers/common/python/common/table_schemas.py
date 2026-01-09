from typing import Literal
from dataclasses import dataclass

CANONICAL_IMAGERY_TABLE_NAME = "canonical_imagery"
IMAGE_LABELS_TABLE_NAME = "image_labels"
CANONICAL_BBOX_TABLE_NAME = "canonical_bounding_boxes"
CANONICAL_SEMANTIC_TABLE_NAME = "canonical_semantic_masks"
CANONICAL_INSTANCE_TABLE_NAME = "canonical_instance_annotations"
UPLOAD_STAGING_TABLE_NAME = "upload_staging"

SqlType = Literal["string", "int", "float", "timestamp", "array_string"]

@dataclass(frozen=True)
class TableSchema:
    cols: list[str]
    key_cols: list[str]
    types: dict[str, SqlType]  # per-column declared type

    def __post_init__(self):
        cols_set = set(self.cols)
        types_set = set(self.types.keys())
        if cols_set != types_set:
            missing = cols_set - types_set
            extra = types_set - cols_set
            raise ValueError(f"Schema mismatch. missing_types={missing}, extra_types={extra}")
        for k in self.key_cols:
            if k not in cols_set:
                raise ValueError(f"key_col {k} not in cols")

CANONICAL_IMAGERY_COLS = [
    "image_id",
    "source_ref",
    "img_type",
    "img_height",
    "img_width",
    "num_channels",
    "dtype",
    "file_size_mb",
    "uploaded_at",
    "data_source",
    "sha256_hash"
]
CANONICAL_IMAGERY_SCHEMA = TableSchema(
    cols=CANONICAL_IMAGERY_COLS,
    key_cols=["image_id"],
    types={
        "image_id": "string",
        "source_ref": "string",
        "img_type": "string",
        "img_height": "int",
        "img_width": "int",
        "num_channels": "int",
        "dtype": "string",
        "file_size_mb": "float",
        "uploaded_at": "timestamp",
        "data_source": "string",
        "sha256_hash": "string",
    },
)

IMAGE_LABELS_COLS = [
    "image_id",
    "label_id",
    "label_type"
]
IMAGE_LABELS_SCHEMA = TableSchema(
    cols=IMAGE_LABELS_COLS,
    key_cols=["image_id", "label_id", "label_type"],
    types={
        "image_id": "string",
        "label_id": "string",
        "label_type": "string"
    },
)

CANONICAL_BOUNDING_BOXES_COLS = [
    "bbox_annotation_id",
    "image_id",
    "source_ref_meta",
    "classes_present"
]
CANONICAL_BOUNDING_BOXES_SCHEMA = TableSchema(
    cols=CANONICAL_BOUNDING_BOXES_COLS,
    key_cols=["bbox_annotation_id"],
    types={
        "bbox_annotation_id": "string",
        "image_id": "string",
        "source_ref_meta": "string",
        "classes_present": "array_string"
    },
)

CANONICAL_SEMANTIC_MASKS_COLS = [
    "semantic_mask_id",
    "image_id",
    "source_ref_png",
    "source_ref_meta",
    "classes_present"
]
CANONICAL_SEMANTIC_MASKS_SCHEMA = TableSchema(
    cols=CANONICAL_SEMANTIC_MASKS_COLS,
    key_cols=["semantic_mask_id"],
    types={
        "semantic_mask_id": "string",
        "image_id": "string",
        "source_ref_png": "string",
        "source_ref_meta": "string",
        "classes_present": "array_string"
    },
)

CANONICAL_INSTANCE_ANNOTATIONS_COLS = [
    "instance_annotation_id",
    "image_id",
    "source_ref_png",
    "source_ref_meta",
    "classes_present"
]
CANONICAL_INSTANCE_ANNOTATIONS_SCHEMA = TableSchema(
    cols=CANONICAL_INSTANCE_ANNOTATIONS_COLS,
    key_cols=["instance_annotation_id"],
    types={
        "instance_annotation_id": "string",
        "image_id": "string",
        "source_ref_png": "string",
        "source_ref_meta": "string",
        "classes_present": "array_string"
    },
)

UPLOAD_STAGING_COLS = [
    "job_id",
    "image_id",
    "temp_source_ref",
    "img_type",
    "img_height",
    "img_width",
    "num_channels",
    "dtype",
    "file_size_mb",
    "uploaded_at",
    "data_source",
    "sha256_hash",
    "string_labels",
    "temp_source_ref_bbox_meta",
    "temp_source_ref_semantic_png",
    "temp_source_ref_semantic_meta",
    "temp_source_ref_instance_png",
    "temp_source_ref_instance_meta",
    "label_fingerprint",
    "classes_present",
    "validation_status",
    "validation_error",
    "dedup_status",
    "dedup_error",
    "registration_status",
    "registration_error",
    "matched_image_id"
]
UPLOAD_STAGING_SCHEMA = TableSchema(
    cols=UPLOAD_STAGING_COLS,
    key_cols=["job_id", "image_id"],
    types={
        "job_id": "string",
        "image_id": "string",
        "temp_source_ref": "string",
        "img_type": "string",
        "img_height": "int",
        "img_width": "int",
        "num_channels": "int",
        "dtype": "string",
        "file_size_mb": "float",
        "uploaded_at": "timestamp",
        "data_source": "string",
        "sha256_hash": "string",
        "string_labels": "array_string",
        "temp_source_ref_bbox_meta": "string",
        "temp_source_ref_semantic_png": "string",
        "temp_source_ref_semantic_meta": "string",
        "temp_source_ref_instance_png": "string",
        "temp_source_ref_instance_meta": "string",
        "label_fingerprint": "string",
        "classes_present": "array_string",
        "validation_status": "string",
        "validation_error": "string",
        "dedup_status": "string",
        "dedup_error": "string",
        "registration_status": "string",
        "registration_error": "string",
        "matched_image_id": "string"
    },
)

TABLES: dict[str, TableSchema] = {
    CANONICAL_IMAGERY_TABLE_NAME: CANONICAL_IMAGERY_SCHEMA,
    IMAGE_LABELS_TABLE_NAME: IMAGE_LABELS_SCHEMA,
    CANONICAL_BBOX_TABLE_NAME: CANONICAL_BOUNDING_BOXES_SCHEMA,
    CANONICAL_SEMANTIC_TABLE_NAME: CANONICAL_SEMANTIC_MASKS_SCHEMA,
    CANONICAL_INSTANCE_TABLE_NAME: CANONICAL_INSTANCE_ANNOTATIONS_SCHEMA,
    UPLOAD_STAGING_TABLE_NAME: UPLOAD_STAGING_SCHEMA
}