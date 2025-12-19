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
    data_source string,
    sha256_hash string COMMENT '64-char hex',
    string_labels array<string>,
    bbox_annotation_ids array<string>,
    semantic_mask_ids array<string>,
    instance_annotation_ids array<string>
)
PARTITIONED BY (day(uploaded_at))
LOCATION 's3://${ICEBERG_BUCKET_NAME}/canonical/imagery/'
TBLPROPERTIES (
  'table_type'='ICEBERG',
  'format'='parquet'
);

CREATE TABLE IF NOT EXISTS ${ICEBERG_DATABASE_NAME}.canonical_bounding_boxes (
    bbox_annotation_id string,
    image_id string COMMENT 'UUID foreign key for the image in the table canonical_imagery',
    source_ref_meta string,
    classes_present array<string>
)
LOCATION 's3://${ICEBERG_BUCKET_NAME}/canonical/bounding-boxes/'
TBLPROPERTIES (
  'table_type'='ICEBERG',
  'format'='parquet'
);

CREATE TABLE IF NOT EXISTS ${ICEBERG_DATABASE_NAME}.canonical_semantic_masks (
    semantic_mask_id string,
    image_id string COMMENT 'UUID foreign key for the image in the table canonical_imagery',
    source_ref_png string,
    source_ref_meta string,
    classes_present array<string>
)
LOCATION 's3://${ICEBERG_BUCKET_NAME}/canonical/semantic-masks/'
TBLPROPERTIES (
  'table_type'='ICEBERG',
  'format'='parquet'
);

CREATE TABLE IF NOT EXISTS ${ICEBERG_DATABASE_NAME}.canonical_instance_annotations (
    instance_annotation_id string,
    image_id string COMMENT 'UUID foreign key for the image in the table canonical_imagery',
    source_ref_png string,
    source_ref_meta string,
    classes_present array<string>
)
LOCATION 's3://${ICEBERG_BUCKET_NAME}/canonical/instance-annotations/'
TBLPROPERTIES (
  'table_type'='ICEBERG',
  'format'='parquet'
);

CREATE TABLE IF NOT EXISTS ${ICEBERG_DATABASE_NAME}.upload_staging (
    job_id string,
    image_id string,
    temp_source_ref string,
    canonical_copy_to string,
    img_type string,
    img_height int,
    img_width int,
    num_channels int,
    dtype string,
    file_size_mb double,
    uploaded_at timestamp,
    data_source string,
    sha256_hash string,
    string_labels array<string>,
    temp_source_ref_bbox_meta string,
    temp_source_ref_mask_png string,
    temp_source_ref_mask_meta string,
    temp_source_ref_instance_annotation_png string,
    temp_source_ref_instance_annotation_meta string,
    classes_present array<string> COMMENT 'Relevant for semantic masks, bboxes, and instance annotations',
    validation_status string,
    validation_error string,
    dedup_status string
    dedup_error string
)
PARTITIONED BY (job_id)
LOCATION 's3://${ICEBERG_BUCKET_NAME}/upload_staging/'
TBLPROPERTIES (
  'table_type'='ICEBERG',
  'format'='parquet'
);

CREATE TABLE IF NOT EXISTS ${ICEBERG_DATABASE_NAME}.single_label (
    dataset_id string,
    image_id string,
    label string
)
PARTITIONED BY (dataset_id)
LOCATION 's3://${ICEBERG_BUCKET_NAME}/single_label/'
TBLPROPERTIES (
  'table_type'='ICEBERG',
  'format'='parquet'
);

CREATE TABLE IF NOT EXISTS ${ICEBERG_DATABASE_NAME}.multi_label (
    dataset_id string,
    image_id string,
    labels array<string>
)
PARTITIONED BY (dataset_id)
LOCATION 's3://${ICEBERG_BUCKET_NAME}/multi_label/'
TBLPROPERTIES (
  'table_type'='ICEBERG',
  'format'='parquet'
);

CREATE TABLE IF NOT EXISTS ${ICEBERG_DATABASE_NAME}.object_detection (
    dataset_id string,
    image_id string,
    bbox_annotation_id string
)
PARTITIONED BY (dataset_id)
LOCATION 's3://${ICEBERG_BUCKET_NAME}/object_detection/'
TBLPROPERTIES (
  'table_type'='ICEBERG',
  'format'='parquet'
);

CREATE TABLE IF NOT EXISTS ${ICEBERG_DATABASE_NAME}.semantic_segmentation (
    dataset_id string,
    image_id string,
    semantic_mask_id string
)
PARTITIONED BY (dataset_id)
LOCATION 's3://${ICEBERG_BUCKET_NAME}/semantic_segmentation/'
TBLPROPERTIES (
  'table_type'='ICEBERG',
  'format'='parquet'
);

CREATE TABLE IF NOT EXISTS ${ICEBERG_DATABASE_NAME}.instance_segmentation (
    dataset_id string,
    image_id string,
    instance_annotation_id string
)
PARTITIONED BY (dataset_id)
LOCATION 's3://${ICEBERG_BUCKET_NAME}/instance_segmentation/'
TBLPROPERTIES (
  'table_type'='ICEBERG',
  'format'='parquet'
);