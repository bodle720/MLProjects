CREATE TABLE IF NOT EXISTS ${ICEBERG_DATABASE_NAME}.canonical_imagery (
    image_id string COMMENT 'UUID primary key for the image',
    source_ref string COMMENT 'S3 URI to file in file bucket',
    img_type string COMMENT 'L or RGB',
    img_height int,
    img_width int,
    num_channels int,
    dtype string,
    file_size_mb double,
    uploaded_at timestamp COMMENT 'UTC upload time in ISO8601',
    uploaded_day date,
    source string,
    sha256_hash string COMMENT '64-char hex',
    phash string COMMENT 'grayscale = 64‑bit hex/base64, rgb = 192‑bit concat.',
    string_labels array<string>,
    bboxes array<string>,
    semantic_masks array<string>,
    instance_annotations array<string>
)
PARTITIONED BY (uploaded_day)
LOCATION 's3://${ICEBERG_BUCKET_NAME}/canonical/imagery/'
TBLPROPERTIES ('table_type'='ICEBERG');

CREATE TABLE IF NOT EXISTS ${ICEBERG_DATABASE_NAME}.canonical_bounding_boxes (
    bbox_annotation_id string,
    image_id string,
    source_ref string,
    classes_present array<string>,
    uploaded_at timestamp,
    uploaded_day date
)
PARTITIONED BY (uploaded_day)
LOCATION 's3://${ICEBERG_BUCKET_NAME}/canonical/bounding-boxes/'
TBLPROPERTIES ('table_type'='ICEBERG');

CREATE TABLE IF NOT EXISTS ${ICEBERG_DATABASE_NAME}.canonical_semantic_masks (
    semantic_mask_id string,
    image_id string,
    source_ref string,
    mask_map map<int,string>,
    classes_present array<string>,
    uploaded_at timestamp,
    uploaded_day date
)
PARTITIONED BY (uploaded_day)
LOCATION 's3://${ICEBERG_BUCKET_NAME}/canonical/semantic-masks/'
TBLPROPERTIES ('table_type'='ICEBERG');

CREATE TABLE IF NOT EXISTS ${ICEBERG_DATABASE_NAME}.canonical_instance_annotations (
    instance_annotation_id string,
    image_id string,
    source_ref string,
    classes_present array<string>,
    annotation_format string,
    uploaded_at timestamp,
    uploaded_day date
)
PARTITIONED BY (uploaded_day)
LOCATION 's3://${ICEBERG_BUCKET_NAME}/canonical/instance-annotations/'
TBLPROPERTIES ('table_type'='ICEBERG');

CREATE TABLE IF NOT EXISTS ${ICEBERG_DATABASE_NAME}.upload_staging (
    job_id string,
    image_id string,
    temp_source_ref string,
    copy_to string,
    img_type string,
    img_height int,
    img_width int,
    num_channels int,
    dtype string,
    file_size_mb double,
    uploaded_at timestamp,
    source string,
    sha256_hash string,
    phash string,
    temp_string_labels_path string,
    temp_bbox_path string,
    temp_semantic_mask_path string,
    temp_instance_annotation_path string,
    validation_status string,
    validation_error string,
    dedup_status string,
    matched_image_id string,
    merge_action string
)
PARTITIONED BY (job_id)
LOCATION 's3://${ICEBERG_BUCKET_NAME}/upload_staging/'
TBLPROPERTIES ('table_type'='ICEBERG');

CREATE TABLE IF NOT EXISTS ${ICEBERG_DATABASE_NAME}.single_label (
    dataset_id string,
    job_id string,
    image_id string,
    added_at timestamp,
    label string
)
PARTITIONED BY (dataset_id)
LOCATION 's3://${ICEBERG_BUCKET_NAME}/single_label/'
TBLPROPERTIES ('table_type'='ICEBERG');

CREATE TABLE IF NOT EXISTS ${ICEBERG_DATABASE_NAME}.multi_label (
    dataset_id string,
    job_id string,
    image_id string,
    added_at timestamp,
    labels array<string>
)
PARTITIONED BY (dataset_id)
LOCATION 's3://${ICEBERG_BUCKET_NAME}/multi_label/'
TBLPROPERTIES ('table_type'='ICEBERG');

CREATE TABLE IF NOT EXISTS ${ICEBERG_DATABASE_NAME}.object_detection (
    dataset_id string,
    job_id string,
    image_id string,
    added_at timestamp,
    bbox_annotation_id string
)
PARTITIONED BY (dataset_id)
LOCATION 's3://${ICEBERG_BUCKET_NAME}/object_detection/'
TBLPROPERTIES ('table_type'='ICEBERG');

CREATE TABLE IF NOT EXISTS ${ICEBERG_DATABASE_NAME}.semantic_segmentation (
    dataset_id string,
    job_id string,
    image_id string,
    added_at timestamp,
    semantic_mask_id string
)
PARTITIONED BY (dataset_id)
LOCATION 's3://${ICEBERG_BUCKET_NAME}/semantic_segmentation/'
TBLPROPERTIES ('table_type'='ICEBERG');

CREATE TABLE IF NOT EXISTS ${ICEBERG_DATABASE_NAME}.instance_segmentation (
    dataset_id string,
    job_id string,
    image_id string,
    added_at timestamp,
    instance_annotation_id string
)
PARTITIONED BY (dataset_id)
LOCATION 's3://${ICEBERG_BUCKET_NAME}/instance_segmentation/'
TBLPROPERTIES ('table_type'='ICEBERG');