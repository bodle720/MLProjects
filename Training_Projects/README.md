# Training Projects

This folder contains computer vision training projects built around exported CVDMS dataset artifacts.

The projects focus on practical model training workflows: loading versioned dataset manifests, training models, logging metrics, generating diagnostics, selecting models, and preparing trained models for deployment.

## Projects

### Single-Label EuroSAT

[`projects/SingleLabel_EuroSAT/`](projects/SingleLabel_EuroSAT/)

A single-label image classification project using EuroSAT-style land-cover data exported from CVDMS.

This project demonstrates:

* PyTorch transfer learning
* ResNet-based classification
* Staged fine-tuning
* Cross-entropy loss with integer class targets
* Accuracy, precision, recall, F1, confusion matrices, and PR-curve diagnostics
* TensorBoard logging

### Multi-Label BigEarthNet v2

[`projects/MultiLabel_BigEarthNetv2/`](projects/MultiLabel_BigEarthNetv2/)

A multi-label image classification project using BigEarthNet v2-style land-cover data exported from CVDMS.

This project demonstrates:

* Multi-label classification with PyTorch
* Multi-hot targets and `BCEWithLogitsLoss`
* Sigmoid-based inference
* Per-class metrics and threshold tuning
* Macro average precision / mAP
* Multi-label diagnostic plots, including co-occurrence and false-association analysis

### Global Wheat Head Detection

[`projects/GlobalWheatHeadDetection/`](projects/GlobalWheatHeadDetection/)

An object detection project using Global Wheat Head Detection 2021 data exported from CVDMS.

This project demonstrates:

* CVDMS object-detection dataset exports
* Dataset caching and YOLO-format conversion
* YOLO training with Ultralytics
* MLflow experiment tracking
* Validation-based model selection
* Inference-configuration sweeps
* Deployment-oriented model packaging and planning

## Shared Utilities

### CVDMS Training Common

[`cvdms_training_common/`](cvdms_training_common/)

Shared Python utilities used across the training projects.

This package includes helpers for:

* Loading CVDMS metadata and manifests
* Reading images from S3 or local cached mirrors
* Building task-specific datasets and dataloaders
* Computing metrics and diagnostics
* Running training loops
* Generating dataset mosaics and visualizations

The package is intended to be installed locally in editable mode by individual projects.

## Connection to CVDMS

These projects are designed to consume dataset artifacts produced by the AWS-based Computer Vision Dataset Management System:

[`../AWS_Tools/Computer_Vision_DMS/cvdms_cdk/`](../AWS_Tools/Computer_Vision_DMS/cvdms_cdk/)

CVDMS handles dataset ingestion, validation, versioning, split creation, and manifest export. The training projects then use those exported artifacts for reproducible model training and evaluation.
