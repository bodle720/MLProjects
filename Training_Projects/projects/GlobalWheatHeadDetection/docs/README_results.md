# Final Results

This document summarizes the final model-selection result for the Global Wheat Head Detection project.

The selected model is a YOLO11s detector trained on CVDMS-exported Global Wheat Head Detection 2021 artifacts, selected by validation mAP50-95, evaluated once on the held-out test split, and registered for deployment through MLflow.

## Selected Model

| Field | Value |
|---|---|
| Selected training run | `baseline_003_yolo11s_e50_img640_b16_w4` |
| Model family | YOLO11s |
| Training epochs | 50 |
| Training image size | 640 |
| Training batch size | 16 |
| Training workers | 4 |
| Run ID | `0d5e89fd0d654608b8d57d9350583382` |
| Parameters | ~9.43M |
| Model file size | ~18.29 MB |
| Registered model | `GlobalWheatHeadDetector` |
| Registered alias | `champion` |
| Deployment URI | `models:/GlobalWheatHeadDetector@champion` |

The final deployment app loads this model from the MLflow Model Registry and serves it through a FastAPI/Docker inference API. See [`../deployment/README.md`](../deployment/README.md).

## Dataset Context

The model was trained on the CVDMS version of the Global Wheat Head Detection 2021 dataset.

| Split | Images | Avg. boxes / image |
|---|---:|---:|
| Train | 3,605 | 45.4 |
| Validation | 1,448 | 30.6 |
| Test | 1,334 | 50.5 |

The project preserves the official source splits. The test split is expected to be more challenging because the earlier dataset analysis showed meaningful split differences, and the test split also has a higher average object density than validation.

For dataset analysis, see [`README_initial_dataset.md`](README_initial_dataset.md).

## Model Selection Method

Model selection was performed on the validation split only.

The selection script swept inference-time settings across candidate MLflow runs and selected the candidate with the highest validation `box_map50_95`.

Sweep configuration:

| Setting | Values |
|---|---|
| Split | `val` |
| Selection metric | `box_map50_95` |
| Confidence threshold | `0.001` |
| Image sizes | `512`, `640`, `768` |
| NMS IoU values | `0.7`, `0.8`, `0.85`, `0.9` |
| Max detections | `300`, `500`, `1000` |
| Batch | `4` |
| Rectangular validation | `true` |
| Device | GPU `0` |

The winning validation configuration was:

| Field | Value |
|---|---:|
| Run | `baseline_003_yolo11s_e50_img640_b16_w4` |
| Image size | 640 |
| NMS IoU | 0.8 |
| Max detections | 1000 |
| Validation rank | 1 |
| Validation mAP50-95 | 0.528391 |

The held-out test split was not used for model selection. It was evaluated after selecting the final validation winner.

## Validation Sweep Result

The selected model achieved the following validation metrics under the selected inference configuration:

| Metric | Validation |
|---|---:|
| Precision | 0.924136 |
| Recall | 0.837216 |
| mAP50 | 0.920035 |
| mAP75 | 0.555195 |
| mAP50-95 | 0.528391 |

Validation evaluation speed from the sweep:

| Timing | ms / image |
|---|---:|
| Preprocess | 0.280 |
| Inference | 9.096 |
| Postprocess | 0.977 |
| Total eval pipeline | 10.353 |

These timings are YOLO evaluation timings from the validation sweep, not full FastAPI request latency.

## Final Held-Out Test Result

After validation-based selection, the selected model was evaluated once on the held-out test split.

| Metric | Test |
|---|---:|
| Precision | 0.805931 |
| Recall | 0.593790 |
| mAP50 | 0.684068 |
| mAP75 | 0.225171 |
| mAP50-95 | 0.308762 |

Test evaluation settings:

| Setting | Value |
|---|---:|
| Confidence threshold | 0.001 |
| Image size | 640 |
| NMS IoU | 0.8 |
| Max detections | 1000 |
| Batch | 4 |
| Device | GPU `0` |

Test evaluation speed:

| Timing | ms / image |
|---|---:|
| Preprocess | 0.256 |
| Inference | 5.692 |
| Postprocess | 1.019 |
| Total eval pipeline | 6.967 |

These timings are model-evaluation timings from Ultralytics/YOLO, not full API latency. API latency is measured separately by the FastAPI deployment service.

## Lightweight Comparison

A lightweight YOLO11n candidate was also tracked for comparison.

| Model | Run | Params | Size | Val mAP50 | Val mAP50-95 | Eval pipeline |
|---|---|---:|---:|---:|---:|---:|
| YOLO11s | `baseline_003_yolo11s_e50_img640_b16_w4` | ~9.43M | ~18.29 MB | 0.920035 | 0.528391 | 10.353 ms/img |
| YOLO11n | `baseline_001_yolo11n_e30_img640_b16_w4` | ~2.59M | ~5.22 MB | 0.896359 | 0.500922 | 8.809 ms/img |

The YOLO11n model is substantially smaller and somewhat faster, but YOLO11s produced the stronger validation mAP50-95 and was selected as the primary model.

## Training Curve Summary

The selected baseline 003 run trained for 50 epochs. The final epoch remained strong:

| Epoch | Val Precision | Val Recall | Val mAP50 | Val mAP50-95 |
|---:|---:|---:|---:|---:|
| 50 | 0.92299 | 0.86245 | 0.92874 | 0.51710 |

The best validation mAP50-95 during training occurred around epoch 42:

| Epoch | Val Precision | Val Recall | Val mAP50 | Val mAP50-95 |
|---:|---:|---:|---:|---:|
| 42 | 0.92205 | 0.86186 | 0.92892 | 0.52603 |

This is consistent with the selected checkpoint’s later validation sweep result of `0.528391` mAP50-95 under the selected inference configuration.

## Interpretation

The selected YOLO11s model performs well on the validation split, with high precision, strong recall, and validation mAP50 above 0.92. The final held-out test result is lower, especially for recall and mAP75/mAP50-95, which suggests the test split is meaningfully harder than validation.

That test gap is not unexpected for this dataset. Earlier dataset exploration showed split-level differences in image quality and distribution, and the test split has higher average object density than validation. Dense wheat-head scenes, overlapping heads, and small-object localization make stricter IoU metrics much harder than mAP50.

The model is still useful as a completed object-detection workflow: it was trained on versioned CVDMS artifacts, selected by validation metrics, evaluated on a held-out test split, registered in MLflow, and served through a Dockerized FastAPI app. The result shows a realistic train-to-deploy pipeline rather than only a training notebook.

## Deployment Handoff

The selected model was registered as:

```text
models:/GlobalWheatHeadDetector@champion
```

The deployment app loads this MLflow champion model at startup and exposes an image-upload prediction endpoint.

See:

```text
deployment/README.md
```

The deployment response includes structured detections and request-level timing fields, including upload/validation time, model inference time, parsing time, and total request latency.

## Future Additions

### Prediction Examples

Add several held-out test images with predicted boxes drawn over the original images.

Planned examples:

| Example | Notes |
|---|---|
| Example 1 | Typical successful detection |
| Example 2 | Dense wheat-head scene |
| Example 3 | Harder/failure case with missed or merged heads |

Suggested folder:

```text
docs/images/results/
```

### Training Plots

Add a small number of plots from `results.csv`, such as:

- validation mAP50-95 over epochs
- validation box/class/DFL loss over epochs
- train vs. validation loss if useful

Suggested files:

```text
docs/images/results/baseline_003_val_map50_95.png
docs/images/results/baseline_003_val_losses.png
```

### API Latency Results

Add latency measurements from the deployed FastAPI app after running repeated `/predict` requests through Docker.

Suggested summary:

| Mode | Requests | Median latency | Mean latency | Notes |
|---|---:|---:|---:|---|
| Docker CPU | TBD | TBD | TBD | Local laptop Docker run |
| Local GPU | TBD | TBD | TBD | Optional, non-Docker or GPU-enabled Docker |

The API writes request-level timing fields that can be used to create latency histograms or summary tables.