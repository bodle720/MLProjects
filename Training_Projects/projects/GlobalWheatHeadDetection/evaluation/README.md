# Evaluation Report Generator

This folder contains a standalone evaluation script for the selected Global Wheat Head Detection model.

The script loads the selected model from MLflow, runs YOLO evaluation over a full dataset split, saves a timestamped evaluation report, and optionally writes visual examples with ground-truth and predicted boxes overlaid.

## Prerequisites

Start the MLflow server before running evaluation:

```bash
mlflow server --host 0.0.0.0 --port 5000
```

The evaluator expects the selected model to be available from MLflow, for example:

```text
models:/GlobalWheatHeadDetector@champion
```

The script downloads/resolves the MLflow model artifact, finds the packaged YOLO `.pt` weights, loads them with Ultralytics, and then runs evaluation locally.

## Example command

From the project root:

```bash
python -m evaluation.main ^
  --data-yaml training/data/yolo/global-wheat-head-2021-v1/dataset.yaml ^
  --split test ^
  --mlflow-tracking-uri http://127.0.0.1:5000 ^
  --model-uri models:/GlobalWheatHeadDetector@champion ^
  --metric-conf 0.001 ^
  --visual-conf 0.25 ^
  --iou 0.8 ^
  --imgsz 640 ^
  --max-det 1000 ^
  --batch 4 ^
  --workers 0 ^
  --device 0 ^
  --visualize-sample 12 ^
  --visualize-strategy random
```

The script always evaluates the full selected split. `--visualize-sample` only controls how many images are saved for visual inspection.

## CPU vs GPU

Use GPU inference/evaluation with:

```bash
--device 0
```

This tells Ultralytics to use CUDA device `0`, assuming CUDA-enabled PyTorch is installed and a compatible GPU is available.

Use CPU evaluation with:

```bash
--device cpu
```

CPU mode is more portable but slower. GPU mode is preferred for full-split evaluation on this project because the local training environment already uses a GPU.

## Output folder

Each run creates a new timestamped folder under:

```text
evaluation/outputs/
```

Example:

```text
evaluation/outputs/eval_test_split_20260521_072005/
```

Expected output structure:

```text
evaluation/outputs/eval_test_split_YYYYMMDD_HHMMSS/
├── settings.json
├── mlflow_model_info.json
├── eval_summary.json
├── eval_summary.md
├── visualizations_summary.json
├── visualizations/
│   ├── 001_image_name.jpg
│   ├── 002_image_name.jpg
│   └── ...
├── mlflow_model_artifact/
└── yolo_eval/
```

The timestamped folder avoids overwriting previous evaluation runs.

## Saved settings

Each run saves the full evaluation configuration to:

```text
settings.json
```

This includes:

- dataset YAML path
- split
- MLflow tracking URI
- model URI
- metric confidence threshold
- visualization confidence threshold
- image size
- NMS IoU
- max detections
- batch size
- workers
- device
- visualization sample count
- visualization strategy
- matching IoU threshold

This makes each report reproducible and auditable.

## Visualization colors

Saved visualization images use the following colors:

| Color | Meaning |
|---|---|
| Red | Ground-truth box |
| Green | Predicted box that matched a ground-truth box |
| Blue | Predicted box that did not match a ground-truth box |

Ground-truth boxes are always drawn in red. A successful detection does not change the ground-truth box color; instead, the corresponding prediction is drawn in green. This means a correctly detected object may show both a red ground-truth box and a nearby or overlapping green prediction box.

A blue box is an unmatched prediction. In practical terms, blue boxes are possible false positives under the visualization matching rule.

## IoU settings

The evaluator uses three related but different IoU concepts:

| Concept | Setting / metric | What it controls |
|---|---|---|
| NMS IoU | `--iou` | YOLO post-processing; controls how overlapping predicted boxes are filtered |
| Metric IoU | `mAP50`, `mAP75`, `mAP50-95` | Official evaluation strictness for deciding whether predictions localize objects well |
| Visualization match IoU | `--match-iou-threshold` | Visualization-only rule for coloring predictions green or blue |

`--iou` is passed to Ultralytics YOLO during evaluation and prediction. It controls non-maximum suppression, so it affects which overlapping predicted boxes survive post-processing. Because it changes the final prediction set, it can indirectly affect reported precision, recall, and mAP.

The metric IoU thresholds are part of the official object-detection metrics. `mAP50` evaluates matches at IoU 0.50, `mAP75` evaluates matches at IoU 0.75, and `mAP50-95` averages performance across multiple IoU thresholds from 0.50 to 0.95. These metric thresholds are not the same thing as the `--iou` NMS setting.

`--match-iou-threshold` is only used by this evaluation script after predictions have already been generated. It decides whether a predicted box should be drawn green or blue in the saved visualization images. For example, with `--match-iou-threshold 0.5`, a predicted box is drawn green if it overlaps an unmatched ground-truth box of the same class with IoU of at least 0.5.

The green/red/blue overlays are therefore an interpretability aid, not a replacement for the official mAP metrics.

## Visualization strategies

`--visualize-strategy` controls which images are saved as visual examples. It does not change the full-split metrics.

| Strategy | Meaning |
|---|---|
| `random` | Selects a seeded random sample from the split |
| `first` | Selects the first N image paths from the split |
| `most_boxes` | Selects images with the most ground-truth boxes |
| `highest_conf` | Selects images with the highest-confidence predictions |

`random` is a good default for representative examples.

`most_boxes` is useful for dense wheat-head scenes.

`highest_conf` can produce visually strong examples, but it is slower because it must run predictions across the split to rank images.

## Why there are two confidence thresholds

The evaluator uses two confidence values:

```text
metric_conf
visual_conf
```

`metric_conf` is used for full-split YOLO evaluation. It is usually set very low, such as:

```bash
--metric-conf 0.001
```

This allows YOLO to compute mAP and precision/recall behavior from many candidate detections across confidence levels.

`visual_conf` is used only for saved example images. It should usually match deployment, for example:

```bash
--visual-conf 0.25
```

This makes the visual examples easier to read and closer to what a user would see from the deployed FastAPI app.

## Note

The visualizations are not identical to the reported metrics. The reported metrics come from full-split YOLO evaluation using `metric_conf`, usually `0.001`. The saved visualization images use `visual_conf`, usually `0.25`, to avoid clutter and show deployment-style predictions.

So the metrics answer:

```text
How well does the detector rank and localize objects across the full split?
```

The visualizations answer:

```text
What do the model predictions look like at the deployment confidence threshold?
```

The green/red/blue overlays are therefore illustrative, not the exact object set used internally for mAP50-95.