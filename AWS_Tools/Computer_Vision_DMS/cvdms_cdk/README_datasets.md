# Dataset Operations Overview

This document describes the current dataset functionality available
through the dataset API. The core implemented operations are:

- `get_dataset(...)`
- `submit_create_dataset(...)`
- `submit_update_dataset(...)`
- `submit_delete_dataset_all_versions(...)`

The dataset system is designed around **versioned dataset definitions**
built from canonical imagery and canonical labels already stored in the
platform. A dataset is not a mutable blob that is edited in place.
Instead, each create or update operation produces a **new dataset version**
with its own membership rows, manifest artifacts, metadata, and version
record. The top-level dataset record points to the latest version, while
historical versions remain queryable and auditable.

---

## High-Level Architecture

Dataset state is distributed across three layers:

### 1. Iceberg membership tables
These store the versioned membership rows for each dataset. Membership is
partitioned by `dataset_id` and `version`, and the table used depends on
the dataset label type.

There are five dataset membership shapes:

- **single-label**
  - `dataset_id`
  - `version`
  - `image_id`
  - `label`
  - `split`

- **multi-label**
  - `dataset_id`
  - `version`
  - `image_id`
  - `labels`
  - `split`

- **object-detection**
  - `dataset_id`
  - `version`
  - `image_id`
  - `bbox_annotation_ids`
  - `classes_present`
  - `split`

- **semantic-segmentation**
  - `dataset_id`
  - `version`
  - `image_id`
  - `semantic_mask_ids`
  - `classes_present`
  - `split`

- **instance-segmentation**
  - `dataset_id`
  - `version`
  - `image_id`
  - `instance_annotation_ids`
  - `classes_present`
  - `split`

For the three structured task types, `classes_present` stores the
**dataset-version-specific allowed-class subset** represented by
that membership row, not necessarily the full raw class set present
in the canonical artifact.

### 2. DynamoDB metadata
Two DynamoDB tables hold dataset metadata:

- a **dataset table** containing the dataset-level latest-version pointer and latest summary
- a **dataset versions table** containing one immutable row per dataset version

The version table is the authoritative provenance ledger. It records things like:

- version number
- label type
- operation (`create`, `add`, `remove`)
- split approach (`initial`, `maintain`, `rebalance`)
- split strategy name
- version description
- selection config
- counts by split
- artifact URIs

### 3. S3 version artifacts
Each dataset version writes a version-specific artifact bundle under
a versioned prefix. These artifacts include:

- the selection SQL used for that version
- the selection config JSON
- manifest files for all/train/val/test
- an enriched CSV view of the final split rows
- a metadata JSON summary

---

## Core Dataset Model

A dataset is defined by:

- a stable `dataset_id`
- an immutable `label_type`
- a version history
- a set of canonical image memberships for each version
- a split assignment for each version
- artifacts and metadata describing each version

Each dataset version is derived from either:

- an initial **create**
- an **update add**
- an **update remove**

The important conceptual rule is that a dataset evolves by
**versioning**, not by mutating a single stored definition in place.

---

# Supported Label Types

The dataset system currently supports five task types:

- `single-label`
- `multi-label`
- `object-detection`
- `semantic-segmentation`
- `instance-segmentation`

These differ in how candidates are selected, how membership rows
are stored, and how overlapping rows are handled during update operations.

---

# Selection Config

Both create and update use a `selection_config` to specify which canonical imagery/labels should be considered.

## Required key

- `allowed_classes`

## Optional keys

- `allowed_sources`
- `upload_date_range`
- `width_range`
- `height_range`
- `lighting_buckets`
- `blur_buckets`
- `contrast_buckets`
- `color_buckets`

## Meaning of fields

### `allowed_classes`
Required. A non-empty list of allowed class names. These are
normalized to lowercase and deduped.

This is the most important selection filter. Dataset selection is
always class-scoped.

### `allowed_sources`
Optional list of source names. If provided, only images whose
canonical imagery row has a matching `data_source` are eligible.

### `upload_date_range`
Optional 2-element list of ISO date strings: `[start, end]`. This
filters by canonical imagery upload date.

### `width_range` and `height_range`
Optional 2-element integer ranges. These filter by image width and height.

### `lighting_buckets`, `blur_buckets`, `contrast_buckets`, `color_buckets`
Optional lists of allowed quality-bucket values.

These filters operate on the canonical imagery profiling fields and are
useful for building datasets with controlled visual characteristics.

---

# `create_dataset(...)`

## Purpose

Creates a brand new dataset at version `1`.

## Arguments

- `dataset_id`: globally unique identifier for the dataset
- `label_type`: one of the five supported task types
- `description`: human-readable description of the dataset
- `selection_config`: selection rules used to resolve candidate imagery
- `split_strategy_name`: currently supports `stratified_v1`

## What happens during create

### 1. Inputs are validated
The system validates:

- dataset id format
- label type
- description
- selection config structure
- split strategy

The dataset id must be lowercase, use only letters/digits/hyphens, and be
globally unique.

### 2. Canonical candidates are resolved
The system queries canonical imagery plus canonical label structures and
returns **one candidate row per image** in a task-specific shape.

The candidate resolver also normalizes the rows into Python-native forms
and ensures they are ready for the split assignment stage.

### 3. Split assignment is performed
The resolved candidates are passed into the split strategy,
currently `stratified_v1`, which deterministically assigns each
row to `train`, `val`, or `test`.

### 4. Membership rows are written
The split rows are projected down into the minimal membership schema
for the dataset label type and inserted into the correct Iceberg
membership table under version `1`.

### 5. S3 artifacts are written
The system writes the version artifact bundle:

- selection SQL
- selection config JSON
- manifests
- enriched CSV
- metadata JSON

### 6. DynamoDB metadata is written
The dataset row is created and the first dataset version row is created
transactionally.

---

## Candidate resolution by task type during create

### Single-label
Only images that resolve to **exactly one distinct allowed class** are
retained.

Example:

- raw string labels on image: `["cat", "feline"]`
- `allowed_classes = ["cat", "dog"]`

After filtering to allowed classes, only `cat` remains, so the image is a
valid single-label candidate.

The resulting candidate row contains:

- `label`
- `classes_present = [label]`

### Multi-label
Any image with at least one allowed string label is retained.

The resulting row contains:

- `labels` = deduped allowed labels
- `classes_present` = same deduped allowed labels

### Object detection / semantic segmentation / instance segmentation
Any image whose linked structured label artifacts contain at least one
allowed class is retained.

The resulting row contains:

- the appropriate structured label id array (`bbox_annotation_ids`, `semantic_mask_ids`, or `instance_annotation_ids`)
- `classes_present` normalized to the allowed-class subset only

Important: for these three task types, the label ids are
**not trimmed or rewritten** at dataset creation time. The system
keeps the qualifying canonical label ids unchanged and only
normalizes `classes_present` to the dataset-relevant subset.
Downstream export or training-prep can later decide how to merge
or trim label content.

---

# `get_dataset(...)`

## Purpose

Returns the latest dataset information for a given dataset id.

If the dataset does not exist, it returns:

```python
{"exists": False}
````

Otherwise, it returns normalized information combining:

* top-level dataset state
* latest version metadata
* counts
* artifact pointers

## What it reads

This call is intentionally DDB-only. It reads:

1. the dataset row from the dataset table
2. the latest version row from the dataset versions table

It does not query Iceberg or S3 directly.

## What it returns conceptually

The result includes:

* whether the dataset exists
* dataset id
* label type
* created-at / created-by
* latest version number
* latest version description
* latest version split strategy
* last operation and split approach
* latest version counts
* artifact URIs for the latest version
* latest version selection config

This means callers can inspect the latest dataset definition and
provenance without re-querying membership tables directly.

---

# `update_dataset(...)`

## Purpose

Creates a new dataset version by applying either an `add` or `remove`
operation to the latest existing version.

This does **not** overwrite the old version. Instead, it produces a
new version number.

## Arguments

* `dataset_id`: target dataset
* `operation`: `"add"` or `"remove"`
* `selection_config`: describes which candidate imagery should be added or removed
* `split_approach`: `"maintain"` or `"rebalance"`
* `split_strategy_name`: required only when `split_approach="rebalance"`
* `description`: optional new description for the new version

## High-level update flow

### 1. Validate inputs

The update validator checks:

* dataset id format
* operation
* selection config
* split approach
* whether split strategy is correctly present or absent

Rules:

* `rebalance` requires a split strategy
* `maintain` must not provide one explicitly

### 2. Load current dataset state

The latest dataset metadata is loaded from DynamoDB.

If the dataset does not exist, update fails.

### 3. Determine update invariants

The system derives:

* `latest_version`
* `new_version = latest_version + 1`
* immutable `label_type`
* effective split strategy
* effective description

For `maintain`, the effective split strategy is pulled from the current
latest version metadata.

For `rebalance`, the caller must supply the split strategy explicitly.

### 4. Resolve selected imagery rows

The update selection config is resolved through the same candidate-resolution
logic used in create.

These selected rows represent the imagery to **add** or **remove**.

### 5. Resolve current membership rows

The current latest-version membership rows are loaded from the
appropriate Iceberg membership table.

The shape of these rows depends on task type:

* single-label rows persist `label`
* multi-label rows persist `labels`
* structured-task rows persist both the task payload ids and `classes_present`

### 6. Compute next-version rows

The system computes the next dataset-version membership rows by comparing
the selected imagery set to the current version’s membership set by
`image_id`.

This is where add/remove logic, label enrichment rules, and
maintain/rebalance behavior all happen.

### 7. Write membership rows

The final split rows for the new version are inserted into the
appropriate Iceberg membership table.

### 8. Write version artifacts

The version’s S3 artifacts are written.

### 9. Write DDB version metadata

The dataset row is advanced to the new version, and a new immutable
dataset-version row is written transactionally.

---

# Update Semantics

## Set logic is image-based

Update operations are applied strictly at the `image_id` level.

### `operation="add"`

The next version starts from the current version and includes all
selected images. Overlapping images may be merged depending on task type.

### `operation="remove"`

The next version removes any current membership row whose `image_id`
appears in the selected imagery set.

---

# `maintain` vs `rebalance`

## `maintain`

Preserve existing split assignments for rows that remain in the dataset.

Only **truly new images** receive new split assignments.

This means:

* retained rows keep their existing `train` / `val` / `test`
* overlapping rows that are enriched keep their existing split
* only new rows go through the split strategy

This is useful when you want to extend or trim a dataset without disrupting prior split assignments.

## `rebalance`

Recompute splits for the entire next-version image universe.

This means:

* all retained rows
* all enriched overlap rows
* all new rows

are recombined and passed through the split strategy again.

This is useful when the update changes dataset composition enough that
the original split balance should be reconsidered.

---

# Overlap Behavior on `add`

The most important update behavior is what happens when the selected imagery includes an `image_id` that already exists in the current dataset version.

This behavior depends on task type.

## Single-label

**Current row wins.** The selected row is ignored for overlap purposes.

Reason: one image can only contribute one effective scalar label in a
single-label dataset. If a later selection produces another string label
for the same image, that would create a contradiction. The system
therefore keeps the existing version’s single-label membership row
unchanged.

## Multi-label

**Labels are enriched.**

If the image already exists in the dataset and the selected row
includes new allowed string labels, the final row merges them.

Behavior:

* union current `labels` and selected `labels`
* dedupe
* preserve the image in the dataset
* keep the existing split in `maintain`
* rebalance the merged row in `rebalance`

This allows a multi-label dataset image to grow its label set over time.

## Object detection / semantic segmentation / instance segmentation

**Structured label ids are enriched.**

If the image already exists in the dataset and the selected row
introduces new structured label artifacts, the final row merges them.

Behavior:

* union current task-specific id array with selected task-specific id array
* union current `classes_present` with selected `classes_present`
* dedupe both
* preserve the image in the dataset
* keep the existing split in `maintain`
* rebalance the merged row in `rebalance`

This supports multiple structured label artifacts per image over time.

Examples include:

* additional object-detection box annotations
* additional segmentation masks
* additional instance annotations

The downstream training-prep/export layer is responsible for
interpreting those arrays for a particular model architecture.

---

# Split Strategy: `stratified_v1`

The current split strategy is a deterministic greedy stratified splitter.

## Input expectations

Each candidate row must include:

* `image_id`
* `dataset_label_type`
* `classes_present`

It also benefits from having:

* `sha256_hash`
* `data_source`
* `lighting_bucket`
* `blur_bucket`
* `contrast_bucket`
* `color_bucket`

The splitter preserves all existing row fields and simply adds `split`.

## Primary goals

The splitter tries to satisfy several objectives in priority order:

1. keep duplicate-content groups together
2. preserve class proportions across splits
3. roughly preserve target split sizes
4. secondarily balance source and image-condition buckets

## Leakage-awareness

Rows are grouped by:

* `sha256_hash` when available
* otherwise `image_id`

This means duplicate-content images will tend to stay in the same split, reducing leakage risk.

## Split ratios

The current fixed target ratios are:

* `train = 0.70`
* `val = 0.15`
* `test = 0.15`

## Strategy details

The splitter:

1. validates input rows
2. groups rows by leakage-aware group key
3. computes overall class/source/bucket frequencies
4. derives target counts for each split
5. orders groups by rarity and other deterministic tie-breakers
6. greedily assigns each group to the split that minimizes the scoring objective

The scoring function emphasizes:

* class balance most heavily
* total split size next
* then source and image-condition balancing
* with stable tiny tie-break bias for determinism

## Determinism

The splitter is intentionally deterministic. Given the same candidate rows, it will produce the same split assignments.

---

# Artifact Model

Each dataset version produces a full artifact bundle in S3.

## What is stored

### Selection SQL

The SQL artifact stored for a version means:

* for version `1`: the original full dataset-selection SQL
* for later versions: the **diff SQL** that selected the imagery to add or remove for that version

The system does **not** try to recompute a monolithic “full current
dataset SQL” for every update version. Instead, version history is
represented as:

* initial full definition at v1
* a sequence of changes for v2, v3, ...

To understand the full evolution of a dataset, you start from
version 1 and then follow the version-by-version updates.

### Selection config

The version’s selection config is stored as JSON.

### Manifest files

The version writes:

* `all`
* `train`
* `val`
* `test`

These are canonical dataset-export manifests in a simple task-aware shape.

### Enriched CSV

A CSV view of the final split rows is stored for profile/debug/inspection use.

### Metadata JSON

This contains a summary of split counts and distributions plus pointers to artifacts.

---

# DDB Provenance Model

The version history is also recorded in DynamoDB.

For each version, DynamoDB stores:

* version number
* operation
* split approach
* split strategy name
* version description
* selection config
* creator
* split counts
* total counts
* artifact URIs

This means the authoritative provenance for dataset evolution
lives in the version table, while S3 stores the heavier artifacts
themselves.

---

# How a Dataset Evolves Over Time

A dataset progresses through version history as follows:

## Version 1

Created by `create_dataset(...)`.

This version establishes:

* dataset id
* immutable label type
* original full selection SQL
* original split strategy
* original membership rows
* original artifacts
* original version metadata

## Versions 2+

Created by `update_dataset(...)`.

Each new version stores:

* the update operation (`add` or `remove`)
* the split approach (`maintain` or `rebalance`)
* the diff selection config
* the diff selection SQL
* the final newly materialized membership rows for that version
* the final newly materialized artifacts for that version

The latest dataset state is therefore not stored as a single mutable definition. It is the result of:

* version 1’s initial creation
* plus each versioned change thereafter

---

# Practical Guidance for Users

## When to use `create_dataset(...)`

Use create when defining a dataset for the first time from canonical assets.

Typical scenarios:

* building a fresh classification dataset
* defining an initial structured-label dataset from allowed classes and imagery filters
* creating the first version of a project-specific dataset

## When to use `update_dataset(..., operation="add")`

Use add when you want to extend the current dataset version.

Typical scenarios:

* add newly eligible images matching a selection
* enrich multi-label memberships with newly added labels
* enrich structured-task memberships with additional label artifacts
* expand the dataset to newly allowed classes

## When to use `update_dataset(..., operation="remove")`

Use remove when you want to subtract a selected set of images from the dataset.

Typical scenarios:

* remove a problematic subset
* remove a source
* remove a date range
* remove images from a certain profile bucket

## When to use `maintain`

Use maintain when you want minimal disruption to prior train/val/test assignment.

This is best when:

* adding a modest number of new images
* enriching labels on existing images
* removing a subset without wanting to reshuffle old splits

## When to use `rebalance`

Use rebalance when the update materially changes dataset
composition and you want splits recalculated globally.

This is best when:

* class composition changed significantly
* source mix changed significantly
* image-condition distribution changed significantly
* you want the entire latest version to reflect the best possible current split balance

---

# Summary

The dataset API currently supports four core operations:

* create a new versioned dataset
* delete a dataset and all its versions
* fetch the latest dataset metadata and artifact pointers
* create a new dataset version by updating the latest version through add/remove logic

The system is built around these principles:

* datasets are versioned, not mutated in place
* canonical imagery and labels are the source of truth
* each version materializes its own membership rows and artifacts
* split behavior is explicit and reproducible
* provenance is stored in DynamoDB
* artifacts are stored in S3
* membership is stored in Iceberg
* update history is incremental, with v1 as the starting point and later versions storing change-specific diff SQL/config

