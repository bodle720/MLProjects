# CVDMS Introduction

CVDMS (Computer Vision Data Management System) provides a **durable, canonical
data management layer for imagery and labels** used in machine learning
workflows. While platforms such as Amazon SageMaker excel at labeling orchestration
and model training, they generally do not provide a durable system for
managing image assets, labels, and datasets consistently across projects,
teams, and time.

CVDMS fills that gap by treating imagery and labels as **first-class, canonical
data assets**. It enforces global deduplication, stable canonical identities
for images, and reproducible dataset definitions so that teams can reliably
track which data exists, how it has been labeled, and where it has been used.
This ensures datasets can be regenerated exactly, labels from multiple tools
can be unified under a consistent schema, and training pipelines remain portable
across environments.

Beyond storage and lineage, CVDMS also emphasizes **dataset understanding and
transparency**. By computing dataset-level statistics, derived metadata, and visualization-friendly summaries, CVDMS
enables practitioners to explore and understand their training data more effectively. This improves
**data quality awareness, debugging, and model performance explainability**,
allowing teams to identify imbalance, distribution issues, or labeling
inconsistencies before they impact training results.

In short, CVDMS complements tools like SageMaker by providing the **data management
foundation beneath model training systems**—ensuring image assets are canonical,
datasets are reproducible, and the structure and characteristics of training data
remain understandable and auditable over time.

## Uploading Images and Adding Labels

The Upload flow combines:

- Step Functions orchestration
- AWS Batch workers
- Lambda ingestion stages
- Iceberg-backed tables
- shared infrastructure constructs


to create a pipeline that is:

- deterministic and idempotent
- atomic
- race-safe
- scalable


This architecture allows CVDMS to perform **large-scale
dataset ingestion while maintaining strict guarantees about
data integrity, duplicate detection, and label enrichment.**
See the documentation section below for a notebook demonstrating
how to upload images and labels using the CVDMS upload client.
The sample CSV and JSONL files in the `samples/` folder may contain
image label pairs with no labels or duplicate entries, simply for testing 
the capability of the code to detect such cases.

## Documentation

- **Infrastructure Stacks** – architecture and AWS resources  
  - [README_stacks.md](README_stacks.md)
- **Upload Workflow** – manifest formats and ingestion pipeline  
  - [README_upload.md](README_upload.md)
- **Upload Walkthrough** – sample notebook to show the upload steps 
  - [sample_walkthrough.ipynb](sample_walkthrough.ipynb)