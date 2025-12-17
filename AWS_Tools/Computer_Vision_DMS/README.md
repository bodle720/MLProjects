# CVDMS — Computer Vision Dataset Management System [UNDER CONSTRUCTION]

CVDMS is a lightweight ML data platform for **organizing already-labeled imagery**, generating **repeatable training datasets**, and producing **training manifests** on demand.

It intentionally does **not** replace tools like **SageMaker Ground Truth** (labeling) or **SageMaker Studio** (interactive development). Instead, it focuses on the layer in between: **dataset management, quality gates, and operationally reliable workflows**.

---

## Why this exists

Training a model once is easy. The hard part is doing it **reliably and repeatedly** as data grows and changes.

This project helps with:
- **Reproducible dataset generation** via query/filter logic (not ad-hoc file copying)
- **Data quality validation** before expensive compute runs
- **Operational correctness** with explicit cleanup semantics on failure
- **Scalable processing** orchestrated with AWS Step Functions + AWS Batch
- **Consistent failure handling** routed to a single global DLQ for best-effort cleanup + job finalization

---

## What it does

At a high level:

1. A labeled upload lands in S3 (images + labels + a `job.json` manifest).
2. A kickoff step validates the upload and starts a Step Functions workflow.
3. The workflow runs staged processing (batching, validation, dedup, ingestion, etc.) using Batch jobs and Lambdas.
4. Any failure path emits a structured message to a **single global DLQ**, where a DLQ processor performs best-effort cleanup and marks the job failed.

---

## Core capabilities

### Dataset backbone
- Organizes uploads and associated metadata (job/user/event context).
- Supports building datasets through filtering/query logic (e.g., by label type, data source, time windows, job metadata).

### Upload workflow
- Validates structural correctness (e.g., image/label alignment, duplicate detection).
- Drives distributed compute stages via Step Functions + Batch.

### Failure handling that’s easy to reason about
- Failures are treated as first-class events.
- All errors route to **one global DLQ** with a consistent message shape:
  - `source` (where the failure originated, e.g. `stepfunctions`, `kickoff`, `lambda`)
  - `job_id`
  - `user`
  - `event_type`
  - `error`
- A DLQ processor performs best-effort cleanup (S3 prefix deletion, staging cleanup, job status updates, lock release).

---

## What it does *not* do

- **No labeling UI / workforce labeling** (Ground Truth is great at this).
- **No notebook IDE / experimentation environment** (Studio is great at this).
- **No attempt to be a full enterprise data catalog**.

CVDMS assumes your labels already exist and focuses on **organizing them**, **validating uploads**, and **turning curated subsets into manifests** suitable for training workflows.

---

## Architecture (AWS)

Typical building blocks include:
- **S3** for image/label storage and staging artifacts
- **DynamoDB** for job and metadata tracking
- **Step Functions** for orchestration
- **AWS Batch** for scalable, distributed processing
- **SQS** for event fanout and a **global DLQ**
- **Firehose + Glue + Athena** for centralized structured logging + querying

---

## Getting started (high level)

> This section is intentionally high-level; adapt to your environment and account setup.

1. Deploy infrastructure with AWS CDK.
2. Upload a job manifest and associated labeled data to the expected S3 prefix layout.
3. Kickoff triggers the workflow, validates inputs, and starts the Step Functions execution.
4. Query logs in Athena or inspect job status in DynamoDB.

---

## Repo notes

- The system is designed around **repeatable dataset generation** and **reliable workflow execution**.
- The emphasis is on production-minded behaviors: validation, orchestration, observability, and deterministic cleanup.

---

## Status

Active development. Expect changes as the workflow evolves (e.g., additional stages like enrichment, dataset manifest generation, and downstream training integrations).

---
