# Upload Workflow

For a step-by-step walkthrough of the upload workflow using the programmatic API located in
`cvdms_platform/`, see `sample_walkthrough.ipynb`.

## Input Formats

The following describes how imagery and labels are uploaded into **CVDMS** and the exact formats
accepted by the upload client.

The upload system is designed to accept **multiple common annotation formats**, normalize them
into a **deterministic internal schema**, and then execute the ingestion pipeline.

The supported computer vision tasks are:

- single-label classification
- multi-label classification
- object detection
- semantic segmentation
- instance segmentation

Example input files for each label type can be found in the project's _samples_ directory.
These examples demonstrate both **CSV** and **JSONL** input formats. Each example
demonstrates the exact format expected by the upload client and can be used as a
template when preparing manifests.

The upload client accepts the following manifest file formats:
- `.csv`
- `.jsonl`
- `.ndjson`
- `.manifest`

The input file must contain **one record per image (JSONL)** or **annotation rows (CSV)**
depending on the task.

The upload client will automatically:

1. Validate the input file structure
2. Convert CSV inputs (if input file is .csv) to **Ground Truth–style JSONL**
3. Normalize all inputs into the internal schema

This guarantees that the downstream workflow receives **fully standardized data**
regardless of the input format.

### Example: Object Detection Input

Object detection uploads may be provided as either **CSV** or **JSONL**.

### CSV Input Format for Object Detection

Required columns:
- source-ref
- class-name
- top
- left
- height
- width

Rules:

- `source-ref` must be a valid S3 URI
- bounding box values must be finite numbers
- `height` and `width` must be positive
- `top` and `left` must be non-negative
- class names are normalized to lowercase
- reserved class names (`bg`, `background`) are not allowed 
- rows for the same image **must be grouped together** in the CSV file
- one image can have one or many rows (one row for each box)

Example:

```csv
source-ref,class-name,top,left,height,width
s3://cv-imagery-for-ml/samples/cvdms_project/coco/val2017/random_sample_obj_det_30/000000055150.jpg,person,110,290,116,72
s3://cv-imagery-for-ml/samples/cvdms_project/coco/val2017/random_sample_obj_det_30/000000055150.jpg,person,127,350,85,59
s3://cv-imagery-for-ml/samples/cvdms_project/coco/val2017/random_sample_obj_det_30/000000055150.jpg,inanimate,49,79,131,119
s3://cv-imagery-for-ml/samples/cvdms_project/coco/val2017/random_sample_obj_det_30/000000055150.jpg,inanimate,27,437,103,160
```
### JSONL Input Format for Object Detection

JSONL inputs must follow the **AWS Ground Truth Object Detection
export format**.

Corresponding JSONL entry representing the above CSV annotations:

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
      {"class_id":0,"top":110,"left":290,"height":116,"width":72},
      {"class_id":0,"top":127,"left":350,"height":85,"width":59},
      {"class_id":2,"top":49,"left":79,"height":131,"width":119},
      {"class_id":2,"top":27,"left":437,"height":103,"width":160}
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

The upload client extracts:

- bounding box coordinates
- class IDs
- class name mappings

and converts them into the internal standardized format.

---

## Internal Standardized Manifest Format

Regardless of the input type, all manifests are converted to the internal
schema. Each line contains:

- schema 
- label_type 
- source_ref 
- labels / mask_ref / worker_response_ref

depending on the task.

### Single Label Format
```json
{
  "schema": "cvdms.manifest.v1",
  "label_type": "single-label",
  "source_ref": "s3://bucket/image.jpg",
  "labels": ["cat"]
}
```

### Multi-Label Format
```json
{
  "schema": "cvdms.manifest.v1",
  "label_type": "multi-label",
  "source_ref": "s3://bucket/image.jpg",
  "labels": ["animal","cat"]
}
```
Labels are automatically:

- deduplicated
- normalized to lowercase
- sorted for deterministic ordering

### Object Detection Format
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

### Semantic Segmentation Format
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
Important rules:

- background must be defined in the input as `bg` or `background`
- background is internally mapped to pixel value `0`
- background entries are **excluded from the v1 color_map**

### Instance Segmentation Format
```json
{
  "schema": "cvdms.manifest.v1",
  "label_type": "instance-segmentation",
  "source_ref": "s3://bucket/image.jpg",
  "worker_response_ref": "s3://bucket/annotation.json"
}
```

The worker response JSON contains the instance annotations and
segmentation mask produced by AWS Ground Truth.

---

## Upload Workflow

Once the manifest has been validated and normalized, the upload client
uploads the manifest to S3 and initiates the ingestion workflow.

The following steps occur:
```
Upload Client
        ↓
Manifest validation
        ↓
Conversion to cvdms.manifest.v1
        ↓
Upload to S3
        ↓
Workflow kickoff
```

The upload client writes two files to S3:

```
s3://<file-bucket>/temp/image-upload/<job_id>/job.json
s3://<file-bucket>/temp/image-upload/<job_id>/<job_id>.manifest
```

### job.json

The `job.json` file contains metadata used to start the workflow.

Example:
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
Once `job.json` is uploaded, the following workflow is triggered:

```
S3 Upload
        ↓
UploadEventsQueue (SQS)
        ↓
Kickoff Lambda
        ↓
Upload Step Functions State Machine
```

The state machine executes the upload pipeline:

```
Validation Stage
       ↓
Deduplication Stage
       ↓
Registration Stage
       ↓
Ingest Stage
```

These stages:

- validate image files
- compute image features
- detect duplicate imagery
- register canonical images
- enrich labels when applicable
- write records to Iceberg tables

### Deterministic Manifest Guarantees

The upload client enforces strict validation to ensure reproducible results.

Key guarantees:

- class names normalized to lowercase
- reserved class names (`bg`, `background`) restricted
- numeric values validated
- NaN and Infinity rejected
- labels sorted for deterministic ordering
- consistent JSON formatting

These guarantees ensure that repeated uploads produce **identical canonical
representations** and avoid duplication.

---

## AWS Ground Truth for Instance Segmentation

The following template defines a custom labeling task for AWS Ground Truth that can be used for
the CVDMS instance segmentation workflow. It enables annotators to easily label imagery containing
multiple instances across one or more classes.

In the example below, three classes are used: *person*, *animal*, and *inanimate*.

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