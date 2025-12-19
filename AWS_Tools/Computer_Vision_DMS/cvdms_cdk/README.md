
# Welcome to your CDK Python project!

This is a blank project for CDK development with Python.

The `cdk.json` file tells the CDK Toolkit how to execute your app.

This project is set up like a standard Python project.  The initialization
process also creates a virtualenv within this project, stored under the `.venv`
directory.  To create the virtualenv it assumes that there is a `python3`
(or `python` for Windows) executable in your path with access to the `venv`
package. If for any reason the automatic creation of the virtualenv fails,
you can create the virtualenv manually.

To manually create a virtualenv on MacOS and Linux:

```
$ python -m venv .venv
```

After the init process completes and the virtualenv is created, you can use the following
step to activate your virtualenv.

```
$ source .venv/bin/activate
```

If you are a Windows platform, you would activate the virtualenv like this:

```
% .venv\Scripts\activate.bat
```

Once the virtualenv is activated, you can install the required dependencies.

```
$ pip install -r requirements.txt
```

At this point you can now synthesize the CloudFormation template for this code.

```
$ cdk synth
```

To add additional dependencies, for example other CDK libraries, just add
them to your `setup.py` file and rerun the `pip install -r requirements.txt`
command.

## Useful commands

 * `cdk ls`          list all stacks in the app
 * `cdk synth`       emits the synthesized CloudFormation template
 * `cdk deploy`      deploy this stack to your default AWS account/region
 * `cdk diff`        compare deployed stack with current state
 * `cdk docs`        open CDK documentation

# Why CVDMS Exists (and Why It’s Not Reinventing SageMaker)

CVDMS is **not** trying to replace Amazon SageMaker. SageMaker is excellent at running labeling jobs (Ground Truth) and training/deploying models.  
CVDMS focuses on what SageMaker typically does **not** provide out-of-the-box: a durable, canonical, queryable **image + label asset system** that remains consistent across sources, jobs, teams, and time.

In other words:

- **SageMaker = labeling + training orchestration**
- **CVDMS = canonical data management for imagery + labels + datasets**

---

## The Non-Negotiable Requirements CVDMS Solves

CVDMS exists because these requirements are common in real ML systems and expensive to ignore:

### ✅ “We must never pay to store duplicates long-term.”
SageMaker won’t stop the same image from being uploaded repeatedly across projects, buckets, teams, or labeling runs.  
CVDMS enforces **global deduplication** at the platform level, reducing:
- storage cost
- repeated labeling cost
- training skew from duplicated samples
- operational confusion (“which copy is the real one?”)

---

### ✅ “We need a single canonical ID for an image across sources/jobs.”
Without canonical identity, you can’t reliably answer:
- Is this image the same one we ingested last month?
- Which labels are attached to it?
- Which datasets/models used it?

CVDMS provides a stable identity layer (UUIDs + SHA mapping) so everything downstream has a consistent anchor:
- label lineage
- dataset membership
- auditability
- reproducibility

---

### ✅ “We must be able to reproduce a dataset exactly later.”
“Whatever is in this S3 prefix right now” is not a reproducible dataset definition.

CVDMS stores datasets as explicit, queryable definitions (Iceberg tables), including:
- exactly which images are included
- which label artifacts/versions are used
- splits and metadata

It then emits training manifests so the same dataset version can be regenerated and re-used deterministically.

---

### ✅ “We’ll have labels coming from multiple tools/vendors.”
Ground Truth is one source. Vendors are another. Internal tools are another.

Each source uses different schemas and artifact layouts. CVDMS acts as the normalization layer:
- strict accepted input formats
- consistent canonical storage layout
- consistent metadata and validation
- unified dataset exports regardless of label origin

SageMaker provides one labeling format; it doesn’t unify long-lived labeling data across multiple sources.

---

### ✅ “We may train outside SageMaker.”
Even if SageMaker is the main training platform today, many orgs also train on:
- AWS Batch / ECS
- EKS
- on-prem GPU clusters
- other clouds

If your data system is “SageMaker-only,” your data lifecycle is tied to one execution environment.
CVDMS keeps training as a **consumer** of data, not the owner of it.

---

## What CVDMS Does That SageMaker Typically Won’t Do Automatically

CVDMS provides a dedicated data management layer for imagery and labels:

- **Canonical storage layout** for images and label artifacts (consistent S3 structure + IDs)
- **Global deduplication + canonical identity** across uploads/jobs/sources
- **Append-only label attachment** (multiple label artifacts over time per image)
- **“classes_present” and other derived label metadata** for dataset selection and filtering
- **Dataset versioning** stored as data (Iceberg), not folder snapshots
- **Manifest generation** for 5 training task types:
  - single-label classification
  - multi-label classification
  - object detection (bounding boxes)
  - semantic segmentation
  - instance segmentation
- **Operational robustness** (Step Functions + Batch patterns, structured DLQ, cleanup semantics)

---

## “Not Reinventing the Wheel” (How CVDMS Complements SageMaker)

CVDMS intentionally does **not** rebuild SageMaker capabilities:

- **Not rebuilding Ground Truth** → integrates with it as a label source
- **Not rebuilding SageMaker training** → provides high-quality, reproducible training inputs
- **Not rebuilding Studio** → stays tool-agnostic and focused on data durability

CVDMS is the missing layer **below and beside** SageMaker:
- it makes your image assets coherent over time,
- makes labels portable across tools,
- and makes datasets reproducible across training environments.

---

## When CVDMS Is Worth It

If your goal is **“train one model quickly”**, SageMaker alone may be enough.

If your goal is **“build a scalable, long-lived image + label asset system”** that stays consistent across tools, teams, and time, CVDMS becomes a critical foundation—especially as your ML practice matures.

---

