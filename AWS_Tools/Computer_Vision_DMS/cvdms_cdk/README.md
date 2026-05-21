# Computer Vision Dataset Management System

CVDMS is an AWS CDK-based system for managing computer vision imagery, labels, and reproducible dataset versions.

It provides a durable, canonical data layer beneath model training workflows. Images and labels are ingested once, normalized into consistent internal schemas, deduplicated, registered as canonical assets, and then reused to create versioned datasets for training and evaluation.

The goal is to make computer vision data more reproducible, inspectable, and portable across projects. CVDMS tracks what data exists, how it has been labeled, which dataset versions were created from it, and where the resulting train/validation/test artifacts are stored.

CVDMS supports:

* single-label classification
* multi-label classification
* object detection
* semantic segmentation
* instance segmentation

## Main Workflows

### Upload Workflow

The upload workflow ingests imagery and labels into the canonical CVDMS catalog.

It validates manifests, normalizes supported label formats, computes image statistics, detects duplicate imagery, registers canonical images and labels, and writes structured metadata to Iceberg-backed tables.

See:

* [`README_upload.md`](README_upload.md)
* [`sample_walkthrough_upload.ipynb`](sample_walkthrough_upload.ipynb)

### Dataset Operations

Dataset operations create reproducible dataset versions from registered canonical imagery and labels.

The dataset API supports creating datasets, updating datasets through add/remove operations, retrieving dataset metadata, preserving source splits when needed, and deleting datasets.

See:

* [`README_datasets.md`](README_datasets.md)
* [`sample_walkthrough_datasets.ipynb`](sample_walkthrough_datasets.ipynb)

### Dataset Visualization

The visualization tool is a local Streamlit app for inspecting generated dataset versions.

It helps review class distributions, split balance, image-quality buckets, and other dataset summary artifacts before model training.

See:

* [`visualization_tool/README.md`](visualization_tool/README.md)

## Infrastructure

The CDK app is organized into separate stacks for logging, storage, upload processing, and dataset operations.

See:

* [`README_stacks.md`](README_stacks.md)

## Key Files and Folders

### `app.py`

Main CDK app entry point. It wires together the logging, storage, upload, and dataset stacks.

Deployment is done in two stages:

```bash
cdk deploy cvdmsv1-LoggingStack cvdmsv1-StorageStack --profile <profile-name>
cdk deploy cvdmsv1-UploadStack cvdmsv1-DatasetStack --profile <profile-name>
```

### `config.py`

Infrastructure configuration for Lambda memory, timeouts, AWS Batch settings, and related CDK deployment parameters.

### `cvdms_platform/`

Programmatic API for interacting with deployed CVDMS infrastructure.

Main API entry point:

* [`cvdms_platform/cvdms.py`](cvdms_platform/cvdms.py)

This provides the `CvdmsApp` interface used by the sample notebooks.

### `dataset_bootstrap/`

Generic sample-data bootstrap utilities for downloading selected datasets and creating manifests across supported task types.

Main entry point:

* [`dataset_bootstrap/main.py`](dataset_bootstrap/main.py)

### `additional_dataset_bootstraps/`

Dataset-specific bootstrap utilities that extend the generic bootstrap workflow.

Current example:

* [`additional_dataset_bootstraps/wheat_head_2021/README.md`](additional_dataset_bootstraps/wheat_head_2021/README.md)

### `stacks/`

CDK stack code and helper constructs used to define the AWS infrastructure.

### `workers/`

Lambda, AWS Batch, and shared helper code used by the upload and dataset workflows.

## Documentation

* [`README_upload.md`](README_upload.md) — upload formats, manifest normalization, and upload workflow
* [`README_datasets.md`](README_datasets.md) — dataset API, dataset versioning, split logic, and dataset artifacts
* [`README_stacks.md`](README_stacks.md) — AWS infrastructure stacks and workflow architecture
* [`visualization_tool/README.md`](visualization_tool/README.md) — local dataset visualization app
* [`additional_dataset_bootstraps/wheat_head_2021/README.md`](additional_dataset_bootstraps/wheat_head_2021/README.md) — Global Wheat Head Detection bootstrap utility

## Example Notebooks

* [`sample_walkthrough_upload.ipynb`](sample_walkthrough_upload.ipynb)
  Demonstrates how to upload a prepared manifest into CVDMS.

* [`sample_walkthrough_datasets.ipynb`](sample_walkthrough_datasets.ipynb)
  Demonstrates dataset operations such as create, update, get, and delete.

## Future Improvements

Possible future improvements include:

* Better duplicate detection using perceptual hashing or image-similarity methods.
* More robust duplicate handling across different file extensions, compression levels, resizing, and slight pixel changes.
* Additional dataset bootstrap utilities for public computer vision datasets.
* Expanded visualization summaries for dataset drift, class imbalance, and image-quality differences.
* Additional export helpers for model-specific training formats.

