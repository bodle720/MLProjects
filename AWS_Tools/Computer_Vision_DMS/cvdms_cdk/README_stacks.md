## <u>Logging Stack</u>

The **Logging Stack** implements a centralized structured logging pipeline for the application.
All services emit logs using a shared logging utility that writes structured JSON records to an
Amazon Kinesis Firehose delivery stream. Firehose buffers incoming log events, invokes a
transformation Lambda to normalize and validate the log schema, converts the records from JSON to
Parquet using AWS Glue schema definitions, and writes the resulting files to partitioned storage
in Amazon S3. The Glue catalog table allows the logs to be queried efficiently with Amazon Athena
for debugging, monitoring, and operational analysis.

---

## <u>Storage Stack</u>

The **Storage Stack** provisions the persistent infrastructure used by the system to manage
imagery, labels, datasets, and workflow state. It creates the core storage resources used by the
application, including S3 buckets for canonical imagery, Iceberg table storage, and dataset
artifacts; DynamoDB tables for job tracking, image deduplication, global workflow locking, and
dataset version metadata; and SQS queues used for upload event processing and centralized failure
handling. The stack also deploys a Lambda-based custom resource that executes Athena DDL to create
the Iceberg database and tables at deploy time, ensuring the analytical storage layer is
initialized automatically.

Image ingestion workflows write canonical imagery and label artifacts to the **file bucket**, while structured image
and label metadata is stored in **Iceberg tables** located in the Iceberg storage bucket and
registered in the Glue Data Catalog. Upload events are generated via S3 notifications when a
job manifest is uploaded and are pushed to the `UploadEventsQueue`, which later triggers the
upload workflow. Failures from asynchronous services are routed to a **global dead-letter queue
(DLQ)** that is consumed by a cleanup Lambda responsible for releasing locks, cleaning temporary
files, and restoring system consistency.

#### Dataset Manifests

Dataset **manifest files** are JSONL files stored separately from the raw imagery in a dedicated **datasets bucket**. Each dataset
is identified by a globally unique `dataset_id` and is versioned to ensure reproducibility.
Each version represents an immutable snapshot that describes the exact images and labels used for
training. Because dataset versions are immutable, training runs remain reproducible even as new
images and labels are added to the system.

Dataset artifacts follow a deterministic layout in S3:

```
datasets_bucket
└── datasets/
    └── <dataset_id>/
        ├── v1/
        │   └── manifest.jsonl
        └── v2/
            └── manifest.jsonl
```

For example:

```
s3://<cvdms-datasets-bucket-name>/datasets/wildlife_detector/v1/manifest.jsonl
```

Metadata about datasets and their versions is stored in DynamoDB, allowing the system to track the latest version
of each dataset while preserving historical snapshots:

**DatasetsTable**
- `dataset_id` (partition key)
- `description`
- `label_type`
- `created_at`
- `latest_version`

**DatasetVersionsTable**
- `dataset_id` (partition key)
- `version` (sort key)
- `manifest_s3_uri`
- `created_at`
- `num_images`

### Images and Labels

The system stores all imagery and label metadata and associations in a normalized set of **Apache Iceberg tables** in the
Iceberg storage bucket. These tables form a queryable data catalog that supports flexible dataset
construction, filtering, and analysis. Iceberg tables are used because they provide efficient
columnar storage (Parquet), scalable metadata management, and strong support for large analytical
queries via Amazon Athena.

### Canonical Imagery Storage

All registered image data are stored in the `canonical_imagery` Iceberg table. Each image that successfully
completes the upload pipeline is assigned a unique `image_id` (UUID) and inserted into this table.
The table contains the canonical reference to the image in the file bucket along with computed
image statistics that can later be used for dataset analysis and filtering (e.g., brightness,
blur metrics, colorfulness, and image dimensions). Images are partitioned by upload date (`day(uploaded_at)`),
allowing efficient time-based queries and improved performance when scanning large datasets in Athena.


### Image Features

These statistics allow dataset creators to analyze dataset quality,
detect problematic imagery (blur, low light, glare), and construct
datasets with controlled distributions of visual conditions. Images
are normalized to either grayscale (1 channel) or RGB (3 channels).
Images with other band counts are rejected during validation. Images
are internally converted to 8-bit luminance for metric computation. For
computational efficiency and consistency, quality metrics are computed
on a downsampled copy of the image with maximum side length of 512 pixels.

Note: For grayscale images, color metrics are not meaningful. The system sets:

- sat_mean = 0
- colorfulness = 0
- color_bucket = low


- **Brightness Metrics**

    - `luma_mean`: Average brightness of the image (computed from grayscale/luminance). Range ≈ **0–255**; lower =
darker (night/underexposed), higher = brighter (sunny/overexposed).
  
    - `luma_p10`: 10th percentile brightness — a “dark baseline” for the image. Range **0–255**; low
values mean lots of very dark pixels/shadows.
  
    - `luma_p90`: 90th percentile brightness — a “bright baseline” for the image. Range **0–255**; high values
mean lots of very bright pixels/highlights/glare.
  
    - `dark_frac`: Fraction of pixels considered “dark” (below the fixed luma threshold of 30).
Range **0.0–1.0**; higher means more of the image is in deep shadow/low-light.
  
    - `bright_frac`: Fraction of pixels considered “bright” (above the fixed luma threshold of 225).
Range **0.0–1.0**; higher means more blown highlights/glare/headlights/overexposure.

- **Contrast Metrics**

    - `contrast_luma_std`: Standard deviation of luma values — a global contrast measure. Range ≈ **0–127+**
(practically 0–100ish); low = flat/low-contrast (foggy/overcast), high = strong contrast (sharp shadows, vivid lighting).
  
    - `contrast_luma_p90_p10`: Difference between bright and dark percentiles (`luma_p90 - luma_p10`), another
robust contrast/dynamic-range measure. Range **0–255**; higher = wider brightness
spread, lower = compressed/flat lighting.

- **Sharpness Metric**

    - `blur_laplacian_var`: Variance of a Laplacian edge filter on grayscale — a “sharpness” proxy. Range is
**unbounded** but typically ~**0–1000+** depending on resolution/downsampling;
lower = blurrier (motion/defocus), higher = sharper edges.

- **Color Metrics**

    - `sat_mean`: Average color saturation (from HSV S channel). Range **0.0–1.0**; low =
gray/washed-out (fog/snow/overcast), high = vivid colors (sunny scenes, strong
chroma).
  
    - `colorfulness`: Hasler–Süsstrunk colorfulness metric combining chroma spread and bias. Range is
**unbounded** but typically ~**0–100+**; low = dull/monochrome scenes, high =
very colorful scenes.

- **Derived Buckets**

  - `lighting_bucket`: Categorical lighting condition derived from luma stats and dark/bright fractions.
One of **night, low_light, normal, bright, glare**; use it for slicing performance
by illumination.

  - `blur_bucket`: Categorical sharpness derived from `blur_laplacian_var`. One of **sharp, mild_blur,
blurry**; helpful for seeing how motion/defocus affects detection.

  - `contrast_bucket`: Categorical contrast derived from `contrast_luma_std`. One of **low, medium,
high**; good for analyzing fog/flat lighting vs normal scenes.

  - `color_bucket`: Categorical color richness derived from `sat_mean` and `colorfulness` (for
grayscale images, defaults to low). One of **low, medium, high**; useful for
separating washed-out vs vivid scenes.

### Image–Label Relationships

Labels are connected to images through the `image_labels` table. This table provides
a **one-to-many relationship** between images and labels. Each row links an `image_id` to a
label identifier (`label_id`) and specifies the label type.

This design supports multiple labeling paradigms:

- **Single-label classification**
- **Multi-label classification**
- **Object detection**
- **Semantic segmentation**
- **Instance segmentation**

For classification tasks, the `label_id` is simply a lowercase string label. For structured
labeling tasks (detection and segmentation), the `label_id` refers to a label artifact stored
in one of the canonical label tables described below.

### Canonical Label Tables

Structured labels are stored in dedicated canonical tables. Each label artifact is identified
by a deterministic **label fingerprint** (a SHA256 hash computed from the normalized label representation), which serves as the unique identifier for that label.
These fingerprints prevent duplicate labels from being registered and allow label enrichment
during future uploads.

The canonical label tables are:

- **`canonical_bounding_boxes`** – stores object detection annotations. Each entry references a JSON metadata file describing bounding boxes and class labels.
- **`canonical_semantic_masks`** – stores semantic segmentation labels consisting of an indexed PNG mask and metadata describing class mappings.
- **`canonical_instance_annotations`** – stores instance segmentation annotations consisting of an indexed PNG mask and metadata describing individual object instances.

Each label artifact stores the set of classes present in the annotation (`classes_present`) to support efficient dataset construction, filtering, and analytics.

### Upload Staging

During ingestion, images are first written to the **`upload_staging`** table. This table acts as
a temporary workspace for the upload workflow and records intermediate processing states,
including:

- validation results
- deduplication status
- registration status
- error messages
- computed image statistics
- temporary references to label artifacts

Once an image successfully completes validation, deduplication, and registration, its
metadata is promoted into the canonical tables.

### Dataset Membership Tables

Datasets are represented by a set of Iceberg tables that record which images belong to a dataset
version. Each dataset is identified by a globally unique `dataset_id`, and each modification
produces a new immutable `version`. These tables enable analytical queries on dataset
composition and allow visualization of dataset statistics.

The dataset membership tables include:

- **`single_label`** – membership for single-label classification datasets
- **`multi_label`** – membership for multi-label classification datasets
- **`object_detection`** – membership for object detection datasets
- **`semantic_segmentation`** – membership for semantic segmentation datasets
- **`instance_segmentation`** – membership for instance segmentation datasets

Each row records a `(dataset_id, version, image_id, label)` or `(dataset_id, version, image_id, annotation_id)` pair depending on the task type.

Dataset membership tables are partitioned by `(dataset_id, version)` to allow efficient retrieval
of specific dataset snapshots.

### Why Iceberg is Used

Apache Iceberg is used for the image catalog and dataset membership tables because it provides:

- Scalable metadata management for very large tables
- ACID table updates with snapshot isolation
- Efficient columnar storage via Parquet
- Time-travel queries for historical dataset inspection
- Seamless integration with Amazon Athena

This allows the system to support analytical queries on millions of images and dataset versions
while maintaining strong consistency guarantees.