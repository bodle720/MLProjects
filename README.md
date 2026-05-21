# MLProjects

A repository for machine learning, computer vision, AWS data infrastructure, model training, model evaluation, and deployment-oriented workflows.

This repository is organized around two main areas:

1. **AWS tooling and data-platform infrastructure**
2. **Computer vision training projects built on reusable dataset artifacts**

The central theme is practical end-to-end ML engineering: building dataset infrastructure, exporting reusable dataset artifacts, training models from those artifacts, evaluating results, and preparing models for deployment.

## Highlights

* AWS CDK-based Computer Vision Dataset Management System (__CVDMS__) with Step Functions, Lambda, AWS Batch, S3, DynamoDB, Glue, and Athena.
* Reusable CVDMS dataset artifacts consumed by multiple downstream training projects.
* PyTorch image classification workflows for single-label and multi-label computer vision tasks.
* YOLO object detection workflow with MLflow tracking, model selection, and deployment-oriented packaging.

## Best Starting Points

* [`Computer_Vision_DMS`](AWS_Tools/Computer_Vision_DMS/cvdms_cdk/) — AWS/CDK data-platform work.
* [`GlobalWheatHeadDetection`](Training_Projects/projects/GlobalWheatHeadDetection/) — YOLO object detection, MLflow, and model-selection workflow.
* [`cvdms_training_common`](Training_Projects/cvdms_training_common/) — shared training utilities used across projects.

## Main Sections

### AWS Tools

[`AWS_Tools`](AWS_Tools/) contains AWS-focused projects and reusable cloud engineering examples. These projects cover services and patterns such as S3 workflows, Lambda packaging, AWS Batch, Docker/ECR-based workers, Step Functions, and CDK infrastructure.

The largest AWS project in this repository is the Computer Vision Dataset Management System (__CVDMS__):

* [`Computer_Vision_DMS`](AWS_Tools/Computer_Vision_DMS/cvdms_cdk/)

CVDMS is an AWS CDK-based system for ingesting, validating, organizing, versioning, and exporting computer vision datasets. It is designed to support multiple computer vision task types, including classification and object detection, and provides the dataset artifacts used by the training projects in this repository.

### Training Projects

[`Training_Projects`](Training_Projects/) contains computer vision training projects built around exported CVDMS dataset artifacts.

This section includes:

* [`cvdms_training_common`](Training_Projects/cvdms_training_common/)  
  Shared Python utilities for loading CVDMS metadata, manifests, images, dataloaders, metrics, training loops, and visualization helpers.

* [`SingleLabel_EuroSAT`](Training_Projects/projects/SingleLabel_EuroSAT/)  
  A single-label image classification project using PyTorch transfer learning.

* [`MultiLabel_BigEarthNetv2`](Training_Projects/projects/MultiLabel_BigEarthNetv2/)  
  A multi-label image classification project using multi-hot labels, BCE-with-logits loss, thresholded predictions, and multi-label diagnostics.

* [`GlobalWheatHeadDetection`](Training_Projects/projects/GlobalWheatHeadDetection/)  
  An object detection project using Global Wheat Head Detection data, CVDMS dataset exports, YOLO training, MLflow tracking, model selection, and deployment-oriented planning.
