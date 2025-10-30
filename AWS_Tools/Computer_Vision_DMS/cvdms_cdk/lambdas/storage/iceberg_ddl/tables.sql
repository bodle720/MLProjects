-- canonical imagery
CREATE TABLE IF NOT EXISTS ${ICEBERG_DATABASE_NAME}.canonical_imagery (
    image_id string,
    source_ref string,
    img_type string,
    img_height int,
    img_width int,
    num_channels int,
    uploaded_at timestamp,
    source string,
    sha256_hash string,
    phash string,
    string_labels array<string>,
    bboxes array<struct<label:string,xmin:int,ymin:int,xmax:int,ymax:int>>,
    semantic_masks array<string>,
    instance_annotations array<string>
)
LOCATION 's3://${ICEBERG_BUCKET_NAME}/canonical_imagery/'
TBLPROPERTIES ('table_type'='ICEBERG');

-- canonical semantic masks
CREATE TABLE IF NOT EXISTS ${ICEBERG_DATABASE_NAME}.canonical_semantic_masks (
    mask_id string,
    image_id string,
    source_ref string,
    mask_map map<int,string>,
    uploaded_at timestamp
)
LOCATION 's3://${ICEBERG_BUCKET_NAME}/canonical_semantic_masks/'
TBLPROPERTIES ('table_type'='ICEBERG');

-- canonical annotations
CREATE TABLE IF NOT EXISTS ${ICEBERG_DATABASE_NAME}.canonical_annotations (
    annotation_id string,
    image_id string,
    source_ref string,
    classes_present array<string>,
    annotation_format string,
    uploaded_at timestamp
)
LOCATION 's3://${ICEBERG_BUCKET_NAME}/canonical_annotations/'
TBLPROPERTIES ('table_type'='ICEBERG');

-- upload staging
CREATE TABLE IF NOT EXISTS ${ICEBERG_DATABASE_NAME}.upload_staging (
    job_id string,
    image_id string,
    source string,
    temp_source_ref string,
    sha256_hash string,
    phash string,
    num_channels int,
    img_type string,
    string_labels array<string>,
    bboxes array<struct<label:string,xmin:int,ymin:int,xmax:int,ymax:int>>,
    semantic_masks_temp_path string,
    instance_annotations_temp_path string,
    validation_status string,
    validation_errors array<string>,
    dedup_status string,
    matched_image_id string,
    uploaded_at timestamp,
    merge_action string
)
LOCATION 's3://${ICEBERG_BUCKET_NAME}/upload_staging/'
TBLPROPERTIES ('table_type'='ICEBERG');

-- task-specific tables
CREATE TABLE IF NOT EXISTS ${ICEBERG_DATABASE_NAME}.single_label (
    dataset string,
    job_id string,
    image_id string,
    label string
)
LOCATION 's3://${ICEBERG_BUCKET_NAME}/single_label/'
TBLPROPERTIES ('table_type'='ICEBERG');

CREATE TABLE IF NOT EXISTS ${ICEBERG_DATABASE_NAME}.multi_label (
    dataset string,
    job_id string,
    image_id string,
    labels array<string>
)
LOCATION 's3://${ICEBERG_BUCKET_NAME}/multi_label/'
TBLPROPERTIES ('table_type'='ICEBERG');

CREATE TABLE IF NOT EXISTS ${ICEBERG_DATABASE_NAME}.object_detection (
    dataset string,
    job_id string,
    image_id string,
    bboxes array<struct<label:string,xmin:int,ymin:int,xmax:int,ymax:int>>
)
LOCATION 's3://${ICEBERG_BUCKET_NAME}/object_detection/'
TBLPROPERTIES ('table_type'='ICEBERG');

CREATE TABLE IF NOT EXISTS ${ICEBERG_DATABASE_NAME}.semantic_segmentation (
    dataset string,
    job_id string,
    image_id string,
    mask_id string
)
LOCATION 's3://${ICEBERG_BUCKET_NAME}/semantic_segmentation/'
TBLPROPERTIES ('table_type'='ICEBERG');

CREATE TABLE IF NOT EXISTS ${ICEBERG_DATABASE_NAME}.instance_segmentation (
    dataset string,
    job_id string,
    image_id string,
    annotation_id string
)
LOCATION 's3://${ICEBERG_BUCKET_NAME}/instance_segmentation/'
TBLPROPERTIES ('table_type'='ICEBERG');
