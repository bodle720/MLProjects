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
    sha256_hash string COMMENT '64-char hex'
)
PARTITIONED BY (day(uploaded_at))
LOCATION 's3://${ICEBERG_BUCKET_NAME}/canonical/imagery/'
TBLPROPERTIES (
  'table_type'='ICEBERG',
  'format'='parquet'
);

CREATE TABLE IF NOT EXISTS ${ICEBERG_DATABASE_NAME}.image_labels (
    image_id string COMMENT 'UUID foreign key for the image from canonical_imagery.image_id',
    label_id string COMMENT 'Can be a string label lowercase, or a label id from a label table: bbox_annotation_id or semantic_mask_id or instance_annotation_id',
    label_type string COMMENT 'one of single-label, object-detection, semantic-segmentation, or instance-segmentation'
)
PARTITIONED BY (label_type)
LOCATION 's3://${ICEBERG_BUCKET_NAME}/canonical/image-labels/'
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
    temp_source_ref_semantic_png string,
    temp_source_ref_semantic_meta string,
    temp_source_ref_instance_png string,
    temp_source_ref_instance_meta string,
    label_fingerprint string COMMENT 'A unique hash identifying this label for purpose of comparison to bbox, semantic, or instance labels.',
    classes_present array<string> COMMENT 'String class names in this image label pair.',
    validation_status string COMMENT 'one of pending, passed, failed',
    validation_error string,
    dedup_status string COMMENT 'one of pending, passed, internal_duplicate, external_duplicate',
    dedup_error string,
    registration_status string COMMENT 'one of pending, passed, failed',
    registration_error string,
    matched_image_id string COMMENT 'present when dedup_status = external_duplicate'
)
PARTITIONED BY (job_id)
LOCATION 's3://${ICEBERG_BUCKET_NAME}/upload_staging/'
TBLPROPERTIES (
  'table_type'='ICEBERG',
  'format'='parquet'
);

CREATE TABLE IF NOT EXISTS ${ICEBERG_DATABASE_NAME}.single_label (
    dataset_id string,
    dataset_version int,
    image_id string,
    label string
)
PARTITIONED BY (dataset_id, dataset_version)
LOCATION 's3://${ICEBERG_BUCKET_NAME}/single_label/'
TBLPROPERTIES (
  'table_type'='ICEBERG',
  'format'='parquet'
);

CREATE TABLE IF NOT EXISTS ${ICEBERG_DATABASE_NAME}.multi_label (
    dataset_id string,
    dataset_version int,
    image_id string,
    labels array<string>
)
PARTITIONED BY (dataset_id, dataset_version)
LOCATION 's3://${ICEBERG_BUCKET_NAME}/multi_label/'
TBLPROPERTIES (
  'table_type'='ICEBERG',
  'format'='parquet'
);

CREATE TABLE IF NOT EXISTS ${ICEBERG_DATABASE_NAME}.object_detection (
    dataset_id string,
    dataset_version int,
    image_id string,
    bbox_annotation_id string
)
PARTITIONED BY (dataset_id, dataset_version)
LOCATION 's3://${ICEBERG_BUCKET_NAME}/object_detection/'
TBLPROPERTIES (
  'table_type'='ICEBERG',
  'format'='parquet'
);

CREATE TABLE IF NOT EXISTS ${ICEBERG_DATABASE_NAME}.semantic_segmentation (
    dataset_id string,
    dataset_version int,
    image_id string,
    semantic_mask_id string
)
PARTITIONED BY (dataset_id, dataset_version)
LOCATION 's3://${ICEBERG_BUCKET_NAME}/semantic_segmentation/'
TBLPROPERTIES (
  'table_type'='ICEBERG',
  'format'='parquet'
);

CREATE TABLE IF NOT EXISTS ${ICEBERG_DATABASE_NAME}.instance_segmentation (
    dataset_id string,
    dataset_version int,
    image_id string,
    instance_annotation_id string
)
PARTITIONED BY (dataset_id, dataset_version)
LOCATION 's3://${ICEBERG_BUCKET_NAME}/instance_segmentation/'
TBLPROPERTIES (
  'table_type'='ICEBERG',
  'format'='parquet'
);