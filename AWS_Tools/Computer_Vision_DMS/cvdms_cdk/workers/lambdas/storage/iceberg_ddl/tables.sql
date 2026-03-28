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
    luma_mean double,
    luma_p10 double,
    luma_p90 double,
    dark_frac double,
    bright_frac double,
    contrast_luma_std double,
    contrast_luma_p90_p10 double,
    blur_laplacian_var double,
    sat_mean double,
    colorfulness double,
    lighting_bucket string COMMENT 'One of night, low_light, normal, bright, glare',
    blur_bucket string COMMENT 'One of sharp, mild_blur, blurry',
    contrast_bucket string COMMENT 'One of low, medium, or high',
    color_bucket string COMMENT 'One of low, medium, or high'
)
PARTITIONED BY (day(uploaded_at))
LOCATION 's3://${ICEBERG_BUCKET_NAME}/canonical/imagery/'
TBLPROPERTIES (
  'table_type'='ICEBERG',
  'format'='parquet'
);

CREATE TABLE IF NOT EXISTS ${ICEBERG_DATABASE_NAME}.image_labels (
    image_id string COMMENT 'UUID foreign key for the image from canonical_imagery.image_id',
    label_id string COMMENT 'Either a lowercase string label (for classification tasks) or the ID of a structured label artifact (bbox_annotation_id, semantic_mask_id, or instance_annotation_id)',
    label_type string COMMENT 'one of string-label, object-detection, semantic-segmentation, or instance-segmentation'
)
PARTITIONED BY (label_type)
LOCATION 's3://${ICEBERG_BUCKET_NAME}/canonical/image-labels/'
TBLPROPERTIES (
  'table_type'='ICEBERG',
  'format'='parquet'
);

CREATE TABLE IF NOT EXISTS ${ICEBERG_DATABASE_NAME}.canonical_bounding_boxes (
    bbox_annotation_id string COMMENT 'The _id in the label tables here and below double as the deterministic label fingerprint',
    source_ref_meta string,
    classes_present array<string>
)
LOCATION 's3://${ICEBERG_BUCKET_NAME}/canonical/bounding-boxes/'
TBLPROPERTIES (
  'table_type'='ICEBERG',
  'format'='parquet'
);

CREATE TABLE IF NOT EXISTS ${ICEBERG_DATABASE_NAME}.canonical_semantic_masks (
    semantic_mask_id string COMMENT 'Also deterministic  label fingerprint',
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
    instance_annotation_id string COMMENT 'Also deterministic label fingerprint',
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
    luma_mean double,
    luma_p10 double,
    luma_p90 double,
    dark_frac double,
    bright_frac double,
    contrast_luma_std double,
    contrast_luma_p90_p10 double,
    blur_laplacian_var double,
    sat_mean double,
    colorfulness double,
    lighting_bucket string COMMENT 'One of night, low_light, normal, bright, glare',
    blur_bucket string COMMENT 'One of sharp, mild_blur, blurry',
    contrast_bucket string COMMENT 'One of low, medium, or high',
    color_bucket string COMMENT 'One of low, medium, or high',
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
    registration_status string COMMENT 'one of pending, passed, failed, enriched, no_op',
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
    version int,
    image_id string,
    label string,
    split string
)
PARTITIONED BY (dataset_id, version)
LOCATION 's3://${ICEBERG_BUCKET_NAME}/single_label/'
TBLPROPERTIES (
  'table_type'='ICEBERG',
  'format'='parquet'
);

CREATE TABLE IF NOT EXISTS ${ICEBERG_DATABASE_NAME}.multi_label (
    dataset_id string,
    version int,
    image_id string,
    labels array<string>,
    split string
)
PARTITIONED BY (dataset_id, version)
LOCATION 's3://${ICEBERG_BUCKET_NAME}/multi_label/'
TBLPROPERTIES (
  'table_type'='ICEBERG',
  'format'='parquet'
);

CREATE TABLE IF NOT EXISTS ${ICEBERG_DATABASE_NAME}.object_detection (
    dataset_id string,
    version int,
    image_id string,
    bbox_annotation_ids array<string>,
    classes_present array<string> COMMENT 'The deduped allowed-class subset present in the label, not the full raw classes present in the canonical artifact',
    split string
)
PARTITIONED BY (dataset_id, version)
LOCATION 's3://${ICEBERG_BUCKET_NAME}/object_detection/'
TBLPROPERTIES (
  'table_type'='ICEBERG',
  'format'='parquet'
);

CREATE TABLE IF NOT EXISTS ${ICEBERG_DATABASE_NAME}.semantic_segmentation (
    dataset_id string,
    version int,
    image_id string,
    semantic_mask_ids array<string>,
    classes_present array<string>,
    split string
)
PARTITIONED BY (dataset_id, version)
LOCATION 's3://${ICEBERG_BUCKET_NAME}/semantic_segmentation/'
TBLPROPERTIES (
  'table_type'='ICEBERG',
  'format'='parquet'
);

CREATE TABLE IF NOT EXISTS ${ICEBERG_DATABASE_NAME}.instance_segmentation (
    dataset_id string,
    version int,
    image_id string,
    instance_annotation_ids array<string>,
    classes_present array<string>,
    split string
)
PARTITIONED BY (dataset_id, version)
LOCATION 's3://${ICEBERG_BUCKET_NAME}/instance_segmentation/'
TBLPROPERTIES (
  'table_type'='ICEBERG',
  'format'='parquet'
);