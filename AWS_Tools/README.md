# AWS Tools

This folder contains AWS-focused projects and reusable cloud engineering examples.

The flagship project in this folder is the **Computer Vision Dataset Management System (CVDMS)**, an AWS CDK-based system for ingesting, validating, organizing, versioning, and exporting computer vision datasets.

## Projects

### Computer Vision Dataset Management System

[`Computer_Vision_DMS/cvdms_cdk/`](Computer_Vision_DMS/cvdms_cdk/)

CVDMS is the largest AWS project in this folder. It uses AWS CDK, Step Functions, Lambda, AWS Batch, S3, DynamoDB, Glue, Athena, and Apache Iceberg-backed dataset tables to manage computer vision datasets.

The system supports dataset ingestion, validation, deduplication, registration, dataset versioning, split generation, and export of train/validation/test artifacts for downstream model training.

CVDMS provides the dataset artifacts used by the training projects in [`../Training_Projects/`](../Training_Projects/).

### Lambda Functions for Image Processing

[`Lambda_Functions_for_Image_Processing/`](Lambda_Functions_for_Image_Processing/)

Examples of AWS Lambda functions for image processing workflows, including ZIP-based Lambda packaging, Docker-based Lambda packaging, dependency layers, S3 input/output, and image-processing result inspection.

### Parallel Computing with AWS Batch

[`Parallel_Computing_with_Batch/`](Parallel_Computing_with_Batch/)

A template for launching parallel workloads with AWS Batch. The project demonstrates job queues, job definitions, compute environments, S3-based inputs/outputs, and worker performance monitoring.

### Running Workers on EC2 using ECR

[`Running_Workers_on_EC2_using_ECR/`](Running_Workers_on_EC2_using_ECR/)

A workflow for building Dockerized Python workers, pushing them to Amazon ECR, launching EC2 instances, running containers through UserData and SSM, writing outputs to S3, and terminating instances automatically.

### S3 Functionalities

[`S3_Functionalities/`](S3_Functionalities/)

A collection of reusable S3 examples for uploading, downloading, reading, and writing objects such as JSON, CSV, and Parquet files from Python.

## Notes

Each project folder contains its own README with more detailed setup notes, implementation details, and usage examples.
