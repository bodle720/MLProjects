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