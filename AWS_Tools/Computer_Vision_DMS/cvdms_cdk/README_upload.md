# Upload Workflow

This document describes the manifest formats accepted by the CVDMS upload client and the upload workflow that converts raw imagery and labels into standardized CVDMS records.

For a programmatic walkthrough, see:

[`sample_walkthrough_upload.ipynb`](sample_walkthrough_upload.ipynb)

## Supported Task Types

CVDMS supports uploads for:

* single-label classification
* multi-label classification
* object detection
* semantic segmentation
* instance segmentation

The upload client accepts common annotation formats, validates them, and converts them into a deterministic internal schema before starting the ingestion workflow.

## Accepted Manifest Files

Supported manifest file extensions:

* `.csv`
* `.jsonl`
* `.ndjson`
* `.manifest`

Depending on the task, the input file may contain one record per image or one annotation row per object.

The upload client will:

1. Validate the input structure.
2. Convert CSV inputs to Ground Truth-style JSONL when applicable.
3. Normalize all records into the internal CVDMS manifest schema.

Example input files are available in the project’s `_samples` directory.

## Object Detection Input

Object detection uploads may be provided as CSV or JSONL.

### CSV Format

Required columns:

* `source-ref`
* `class-name`
* `top`
* `left`
* `height`
* `width`

Rules:

* `source-ref` must be a valid S3 URI.
* Bounding box values must be finite numbers.
* `height` and `width` must be positive.
* `top` and `left` must be non-negative.
* Class names are normalized to lowercase.
* Reserved class names, such as `bg` and `background`, are not allowed.
* Rows for the same image must be grouped together.
* One image may have one or many rows.

Example:

```csv
source-ref,class-name,top,left,height,width
s3://cv-imagery-for-ml/samples/cvdms_project/coco/val2017/random_sample_obj_det_30/000000055150.jpg,person,110,290,116,72
s3://cv-imagery-for-ml/samples/cvdms_project/coco/val2017/random_sample_obj_det_30/000000055150.jpg,person,127,350,85,59
s3://cv-imagery-for-ml/samples/cvdms_project/coco/val2017/random_sample_obj_det_30/000000055150.jpg,inanimate,49,79,131,119
s3://cv-imagery-for-ml/samples/cvdms_project/coco/val2017/random_sample_obj_det_30/000000055150.jpg,inanimate,27,437,103,160
```

### JSONL Format

JSONL object detection inputs follow the AWS Ground Truth object detection export format.

Example:

```json
{
  "source-ref": "s3://cv-imagery-for-ml/samples/cvdms_project/coco/val2017/random_sample_obj_det_30/000000055150.jpg",
  "object-detection": {
    "image_size": [
      {
        "width": 640,
        "height": 426,
        "depth": 3
      }
    ],
    "annotations": [
      {"class_id": 0, "top": 110, "left": 290, "height": 116, "width": 72},
      {"class_id": 0, "top": 127, "left": 350, "height": 85, "width": 59},
      {"class_id": 2, "top": 49, "left": 79, "height": 131, "width": 119},
      {"class_id": 2, "top": 27, "left": 437, "height": 103, "width": 160}
    ]
  },
  "object-detection-metadata": {
    "class-map": {
      "0": "person",
      "2": "inanimate"
    }
  }
}
```

The upload client extracts the bounding boxes, class IDs, and class-name mappings, then converts them into the internal CVDMS schema.

## Internal Manifest Format

All uploads are normalized to `cvdms.manifest.v1`.

Each record contains common fields such as:

* `schema`
* `label_type`
* `source_ref`

Task-specific records then include fields such as `labels`, `mask_ref`, `color_map`, or `worker_response_ref`.

### Single Label

```json
{
  "schema": "cvdms.manifest.v1",
  "label_type": "single-label",
  "source_ref": "s3://bucket/image.jpg",
  "labels": ["cat"]
}
```

### Multi-Label

```json
{
  "schema": "cvdms.manifest.v1",
  "label_type": "multi-label",
  "source_ref": "s3://bucket/image.jpg",
  "labels": ["animal", "cat"]
}
```

Labels are deduplicated, normalized to lowercase, and sorted for deterministic ordering.

### Object Detection

```json
{
  "schema": "cvdms.manifest.v1",
  "label_type": "object-detection",
  "source_ref": "s3://bucket/image.jpg",
  "labels": {
    "boxes": [
      {
        "class_name": "person",
        "top": 110,
        "left": 290,
        "height": 116,
        "width": 72
      }
    ]
  }
}
```

Bounding box coordinates are stored as validated numeric values.

### Semantic Segmentation

```json
{
  "schema": "cvdms.manifest.v1",
  "label_type": "semantic-segmentation",
  "source_ref": "s3://bucket/image.jpg",
  "mask_ref": "s3://bucket/mask.png",
  "color_map": {
    "person": ["#ff0000"],
    "car": ["#00ff00"]
  }
}
```

Rules:

* Background must be defined in the input as `bg` or `background`.
* Background is internally mapped to pixel value `0`.
* Background entries are excluded from the v1 `color_map`.

### Instance Segmentation

```json
{
  "schema": "cvdms.manifest.v1",
  "label_type": "instance-segmentation",
  "source_ref": "s3://bucket/image.jpg",
  "worker_response_ref": "s3://bucket/annotation.json"
}
```

The worker response JSON contains the instance annotations and segmentation mask produced by AWS Ground Truth.

## Upload Flow

Once the manifest is validated and normalized, the upload client writes workflow input files to S3 and starts the ingestion process.

```text
Upload client
        ↓
Manifest validation
        ↓
Conversion to cvdms.manifest.v1
        ↓
Upload workflow files to S3
        ↓
Workflow kickoff
```

The upload client writes:

```text
s3://<file-bucket>/temp/image-upload/<job_id>/job.json
s3://<file-bucket>/temp/image-upload/<job_id>/<job_id>.manifest
```

### `job.json`

`job.json` contains metadata used to start the workflow.

```json
{
  "job_id": "<job_id>",
  "user": "<username>",
  "event_type": "IMAGE_UPLOAD",
  "label_type": "object-detection",
  "data_source": "<dataset source>",
  "path_prefix": "<image storage prefix>",
  "registration_time": "YYYY-MM-DD HH:MM:SS",
  "original_manifest_s3_uri": "s3://bucket/temp/image-upload/<job_id>/<job_id>.manifest"
}
```

After `job.json` is uploaded, the workflow is triggered through S3, SQS, and Step Functions.

```text
S3 upload
        ↓
UploadEventsQueue
        ↓
Kickoff Lambda
        ↓
Upload Step Functions state machine
```

The state machine runs three main stages:

```text
Validation
        ↓
Deduplication
        ↓
Registration
```

These stages validate image files, compute image features, detect duplicate imagery, register canonical images, enrich labels when applicable, and write ingestion records to Iceberg-backed tables.

## Deterministic Manifest Guarantees

The upload client applies validation and normalization rules so repeated uploads produce stable canonical records.

Key guarantees:

* Class names are normalized to lowercase.
* Reserved class names are restricted.
* Numeric values are validated.
* NaN and Infinity are rejected.
* Multi-label labels are sorted.
* JSON formatting is consistent.

## AWS Ground Truth Instance Segmentation Template

The following template defines a custom AWS Ground Truth labeling task for CVDMS instance segmentation uploads.

```html
<script src="https://assets.crowd.aws/crowd-html-elements.js"></script>

<crowd-form>
  <crowd-instance-segmentation
    name="annotatedResult"
    src="{{ task.input['source-ref'] | grant_read_access }}"
    header="Segment all instances of person, animal, and inanimate"
    labels="['person','animal','inanimate']"
  >
    <full-instructions header="Segmentation Instructions">
      <ol>
        <li>Inspect the image.</li>
        <li>Select a label and draw masks for every visible instance of that label.</li>
        <li>Repeat until all instances are labeled.</li>
      </ol>
    </full-instructions>

    <short-instructions>
      <p>Create a mask for each instance of person, animal, and inanimate.</p>
    </short-instructions>
  </crowd-instance-segmentation>
</crowd-form>
```
