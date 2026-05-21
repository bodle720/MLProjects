# Global Wheat Head Detection

This project is focused on **wheat head object detection** using the **Global Wheat Head Detection 2021** dataset.

Wheat head detection is a practical object-detection problem connected to agriculture, crop monitoring, phenotyping, and food-security applications. The model must adapt to a more specialized visual domain with dense small-object annotations and meaningful train/test distribution differences.

The goal is to build a complete train-to-deploy object detection workflow:

- ingest and organize the dataset through the CVDMS dataset pipeline
- explore dataset quality, split drift, and annotation characteristics
- convert CVDMS object-detection artifacts into YOLO format
- train and compare Ultralytics YOLO models
- track experiments with MLflow and TensorBoard
- select a best model checkpoint for deployment
- serve the selected model through a future FastAPI/Docker inference app

## Documentation

Detailed documentation is kept in `docs/`. The root README is intentionally short and serves as a table of contents for the project.

### Dataset Exploration

See here: [Initial Dataset Analysis](docs/README_initial_dataset.md)

Explains the dataset source, CVDMS dataset creation process, train/validation/test split structure, image-quality analysis, class/annotation characteristics, and the initial observations about split drift. This is the best place to start for understanding why the dataset is interesting and why the test split is expected to be challenging.

### Training Experiments

See here: [YOLO Training Experiments](docs/README_training_experiments.md)

Tracks the YOLO training workflow, speed tests, batch-size and dataloader-worker decisions, baseline runs, model-size comparisons, validation/test performance, and runtime tradeoffs.

## Repository Layout

```text
GlobalWheatHeadDetection/
├── README.md
├── requirements.txt
├── deployment/
├── docs/
│   ├── README_initial_dataset.md
│   └── README_training_experiments.md
└── training/
    ├── config.example.yaml
    ├── data/
    └── train_yolo/
```

## High-Level Workflow

The project follows this workflow:

```text
CVDMS dataset artifacts
        ↓
local dataset cache
        ↓
YOLO-format conversion
        ↓
YOLO training experiments
        ↓
MLflow/TensorBoard tracking
        ↓
best-checkpoint evaluation
        ↓
model selection
        ↓
FastAPI/Docker inference service
```

## Lower Level Workflow

This project follows a staged workflow from CVDMS dataset artifacts to a YOLO-formatted dataset, model training, validation-time model selection, and final MLflow registration. Some commands require project-specific arguments or configuration values; see the corresponding script files and `config.example.yaml` for details.

### 0. Place CVDMS manifests and metadata

Place the CVDMS dataset artifacts under:

```text
training/data/original/manifests/
```

This folder should contain the exported `metadata.json`, `train.jsonl`, `val.jsonl`, and `test.jsonl` files used as the source of truth for the training dataset.

### 1. Cache the CVDMS image data locally

```bash
python training/data/cache_dataset.py
```

This step reads the CVDMS manifests and mirrors the referenced S3 image/label data into the local project structure so training and visualization do not repeatedly load data from S3.

### 2. Generate dataset mosaics

```bash
python training/data/generate_mosaics.py
```

This creates visual mosaic sheets for quickly inspecting image quality, object density, annotations, and split-level dataset behavior before training.

### 3. Convert CVDMS artifacts to YOLO format

```bash
python -m training.data.convert_cvdms_to_yolo.main --overwrite
```

This converts the cached CVDMS object-detection artifacts into an Ultralytics YOLO-style dataset, including YOLO label files and a `dataset.yaml` file.

### 4. Train YOLO models

First configure `config.yaml` with the desired dataset path, model weights, run name, training parameters, and MLflow settings.

Start the MLflow tracking server:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Then launch training:

```bash
python -m training.train_yolo.main
```

Training logs metrics, artifacts, checkpoints, and evaluation results to MLflow for each candidate run.

### 5. Sweep validation-time inference settings

```bash
python training/train_yolo/model_selection/sweep_postprocess.py
```

This evaluates each candidate `best.pt` checkpoint across the configured validation-time inference grid, including image size, NMS IoU, and maximum detections, then records the best overall and lightweight candidates.

### 6. Select and register the final model

```bash
python training/train_yolo/model_selection/select_best_candidate.py
```

This loads the validation-selected candidate, evaluates it once on the held-out test split, records final metrics and metadata, packages the selected checkpoint with the pyfunc wrapper, and registers the model in MLflow with the `champion` alias.
