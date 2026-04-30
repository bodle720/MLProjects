# cvdms-training-common

Shared PyTorch training utilities for model projects that consume dataset artifacts produced by **CVDMS**.

This package exists so individual training projects do not need to repeatedly implement the same CVDMS loading logic: reading `metadata.json`, finding train/val/test manifests, loading JSONL rows, loading images from S3, validating class maps, and adapting dataset rows into PyTorch-compatible datasets.

## Purpose

CVDMS produces versioned computer-vision datasets in S3. Each dataset version includes split-specific manifests and metadata describing the effective class set for that exact version.

Typical flow:

```text
CVDMS dataset version in S3
→ metadata.json + train/val/test JSONL manifests
→ cvdms_training_common
→ PyTorch Dataset/DataLoader
→ project-specific model training
```

This package is intentionally CVDMS-specific. It assumes the training project is consuming CVDMS dataset-version outputs, not arbitrary ML datasets.

## CVDMS input assumptions

A dataset version is expected to provide:

* `metadata/metadata.json`
* `manifests/train.jsonl`
* `manifests/val.jsonl`
* `manifests/test.jsonl`

The metadata file is expected to include:

* dataset ID
* dataset version
* label type
* effective classes for that version
* `class_to_idx`
* `idx_to_class`
* S3 URIs for the train, validation, and test manifests

For single-label classification, each manifest row is expected to include:

* `image_id`
* `source_ref`
* `split`
* `label_type`
* `label`

The `source_ref` field points to the image in S3, and the `label` must exist in the version-specific `class_to_idx` map.

## Installation

This package is intended to be installed in editable mode into each training project’s virtual environment.

Example:

```bash
cd MLProjects/Training_Projects/cvdms_eurosat_single_label_classifier
pip install -e ../cvdms_training_common
```

After installation, training projects can import shared CVDMS utilities normally:

```python
from cvdms_training_common.metadata import load_cvdms_metadata
from cvdms_training_common.datasets.single_label import CvdmsSingleLabelDataset
from cvdms_training_common.dataloaders.single_label import build_single_label_data_bundle
```

## Example usage

```python
from cvdms_training_common.dataloaders.single_label import build_single_label_data_bundle

metadata_uri = "s3://my-datasets-bucket/datasets/eurosat_demo/v1/metadata/metadata.json"

bundle = build_single_label_data_bundle(
    metadata_uri=metadata_uri,
    batch_size=32,
    num_workers=0,
    image_size=224,
)

train_loader = bundle.train_loader
val_loader = bundle.val_loader
test_loader = bundle.test_loader
metadata = bundle.metadata
```

A project-specific training script can then define its own model, transforms, loss function, optimizer, scheduler, and training loop behavior.

## Relationship to CVDMS

CVDMS is responsible for creating versioned datasets and writing the durable artifacts to S3.

`cvdms-training-common` is responsible for reading those artifacts and adapting them into reusable PyTorch-friendly components.

Individual training projects remain responsible for model design, experiment logic, evaluation, cloud training configuration, and deployment choices.