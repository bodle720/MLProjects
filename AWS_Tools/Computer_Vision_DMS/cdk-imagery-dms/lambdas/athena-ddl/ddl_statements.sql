-- Create database
CREATE DATABASE IF NOT EXISTS cv_datalake;

-- Imagery metadata
CREATE TABLE IF NOT EXISTS cv_datalake.imagery_metadata (
  uuid string,
  uploaded_at timestamp,
  band_type string,            -- 'rgb' | 'grayscale'
  original_extension string,
  image_s3_uri string,
  mask_s3_uri string,
  labels map<string,string>,
  source_job_id string,
  deleted_at timestamp,
  embedding_version int
)
PARTITIONED BY (date string)
LOCATION 's3://${DATALAKE_BUCKET}/iceberg/imagery_metadata/'
TBLPROPERTIES (
  'table_type'='ICEBERG',
  'write_target_data_file_size_bytes'='134217728',
  'write_compression'='zstd'
);

-- Embeddings
CREATE TABLE IF NOT EXISTS cv_datalake.embeddings (
  uuid string,
  embedding array<float>,
  uploaded_at timestamp,
  source_job_id string,
  embedding_version int
)
PARTITIONED BY (date string, job_id string)
LOCATION 's3://${DATALAKE_BUCKET}/iceberg/embeddings/'
TBLPROPERTIES (
  'table_type'='ICEBERG',
  'write_target_data_file_size_bytes'='134217728',
  'write_compression'='zstd'
);

-- Classification labels
CREATE TABLE IF NOT EXISTS cv_datalake.classification_labels (
  uuid string,
  dataset_id string,
  label string,
  annotator string,
  timestamp timestamp
)
PARTITIONED BY (dataset_id string)
LOCATION 's3://${DATALAKE_BUCKET}/iceberg/classification_labels/'
TBLPROPERTIES (
  'table_type'='ICEBERG',
  'write_target_data_file_size_bytes'='134217728',
  'write_compression'='zstd'
);

-- Bounding boxes
CREATE TABLE IF NOT EXISTS cv_datalake.bbox_labels (
  uuid string,
  dataset_id string,
  x_min int,
  y_min int,
  x_max int,
  y_max int,
  class string,
  annotator string,
  timestamp timestamp
)
PARTITIONED BY (dataset_id string)
LOCATION 's3://${DATALAKE_BUCKET}/iceberg/bbox_labels/'
TBLPROPERTIES (
  'table_type'='ICEBERG',
  'write_target_data_file_size_bytes'='134217728',
  'write_compression'='zstd'
);

-- Dataset membership (derived)
CREATE TABLE IF NOT EXISTS cv_datalake.dataset_membership (
  dataset_id string,
  uuid string,
  snapshot_id string,
  created_at timestamp
)
PARTITIONED BY (dataset_id string)
LOCATION 's3://${DATALAKE_BUCKET}/iceberg/dataset_membership/'
TBLPROPERTIES (
  'table_type'='ICEBERG',
  'write_target_data_file_size_bytes'='134217728',
  'write_compression'='zstd'
);
