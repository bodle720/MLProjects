# Production Readiness Future Work

This document outlines possible future work for moving the Global Wheat Head Detection project from a deployment-ready portfolio prototype toward a more production-minded agricultural vision system.

The current Project 3 system demonstrates the full machine learning lifecycle: CVDMS dataset ingestion, YOLO training, MLflow model selection, held-out evaluation, FastAPI serving, Docker deployment, and latency benchmarking. However, the selected model is best understood as a deployed prototype, not a production-grade wheat-head counting system.

Production readiness would require more than another training run. It would require clearer acceptance criteria, deeper failure analysis, stronger data quality checks, deployment-specific operating-point selection, and monitoring/retraining infrastructure.

## Current Status

The current system can:

- train and evaluate a YOLO wheat-head detector,
- select a model using validation metrics,
- register the selected model as an MLflow `champion`,
- serve predictions through FastAPI and Docker,
- report deployed API latency measurements,
- generate evaluation summaries and visual diagnostics.

The current system is not yet production-grade because dense wheat-head scenes still show missed heads, duplicate predictions, imperfect localization, and a meaningful validation-to-test performance gap.

## What "Production Grade" Would Mean

For this project, production grade would mean the system is reliable enough for a specific agricultural use case, not merely that it runs in Docker.

Possible production targets could include:

| Use Case | Production Requirement |
|---|---|
| Research visual aid | Predictions help researchers inspect images, with human review allowed |
| Automated image-level counting | Predicted wheat-head counts are accurate enough for downstream analysis |
| Commercial phenotyping pipeline | The system is robust across farms, cameras, seasons, cultivars, lighting, blur, and field conditions |

Before claiming production readiness, the project would need explicit acceptance criteria such as maximum count error, minimum recall on dense scenes, acceptable false-positive rate, latency requirements, and validation on representative field data.

## Future Work Areas

### 1. Label QA and Dataset Audit

Label QA means checking the quality and consistency of the ground-truth annotations. For this project, that means reviewing whether wheat heads are consistently boxed, whether dense overlapping heads are labeled correctly, whether partially visible heads are handled consistently, and whether duplicate or missing annotations exist.

This matters because object detection metrics depend heavily on annotation quality. A model can look worse than it really is if the ground-truth boxes are inconsistent, especially at stricter localization metrics such as mAP75 and mAP50-95.

Possible work:

- sample and review labels across train/val/test splits,
- inspect dense and high-error images,
- flag missing or duplicate ground-truth boxes,
- check tiny, unusually large, or suspicious boxes,
- compare annotation patterns across image sources or conditions.

### 2. Failure-Case Mining

Failure-case mining means using evaluation outputs to find the images where the model performs worst. Instead of only looking at aggregate mAP, the goal is to identify recurring failure modes that can guide model and data improvements.

For this project, useful failure groups might include images with many missed wheat heads, many unmatched predictions, high duplicate-prediction counts, high count error, blur, poor lighting, or dense overlap.

Possible work:

- rank images by missed detections,
- rank images by unmatched predictions,
- compute predicted count vs. ground-truth count error,
- generate a failure-case image gallery,
- summarize common visual patterns in poor-performing examples.

### 3. Deployment-Threshold Sweep

A deployment-threshold sweep tests different inference settings for the actual deployed API behavior. This is different from selecting a model by validation mAP.

The evaluation configuration may use a very low confidence threshold so that official metrics can consider many candidate detections. A deployed API usually needs a cleaner operating point that balances recall, false positives, duplicate boxes, visual quality, and latency.

Settings to sweep could include:

| Setting | Meaning |
|---|---|
| Confidence threshold | Minimum confidence required to return a detection |
| NMS IoU threshold | Overlap threshold used to suppress duplicate predictions |
| Max detections | Maximum number of detections returned per image |
| Image size | Input size used during inference |

Possible metrics:

- precision and recall at deployment settings,
- duplicate prediction rate,
- missed-head rate,
- predicted count error,
- average and p95 latency,
- qualitative visual cleanliness.

### 4. Tiled or SAHI-Style Inference

Standard YOLO inference resizes the full image to a fixed input size. For dense small-object detection, this can make individual wheat heads very small in the model input.

Tiled inference addresses this by splitting an image into smaller overlapping tiles, running detection on each tile, and merging the results. This can improve small-object recall because each wheat head appears larger relative to the tile.

Possible work:

- compare standard YOLO inference against tiled inference,
- test different tile sizes and overlap ratios,
- measure mAP, recall, duplicate rate, count error, and latency,
- evaluate whether tiled inference improves dense-scene performance enough to justify the added runtime cost.

### 5. Larger or Newer Detector Experiments

A larger detector may improve recall and localization, but model size alone is unlikely to solve all production-readiness issues. The current bottleneck likely includes small objects, dense overlap, image resizing, label ambiguity, and domain shift.

Possible model experiments:

- YOLO11m / YOLO11l / YOLO11x,
- newer Ultralytics detectors,
- RT-DETR-style detectors,
- ensemble approaches,
- models better suited to dense small-object detection.

Any larger-model experiment should be evaluated against both accuracy and deployment cost, including model size, inference latency, and hardware requirements.

### 6. Postprocessing Beyond Default NMS

NMS, or non-maximum suppression, removes overlapping predictions that appear to refer to the same object. In dense wheat-head scenes, default NMS can be difficult to tune because true wheat heads may overlap visually while duplicate predictions may also overlap.

Future postprocessing experiments could explore whether duplicate predictions can be reduced without harming recall.

Possible work:

- tune NMS IoU thresholds,
- test Soft-NMS,
- test Weighted Boxes Fusion,
- apply tile-aware box merging,
- develop wheat-head-specific duplicate suppression rules.

This would mainly target duplicate or visually messy predictions. It would not fully solve missed detections in difficult dense scenes.

### 7. Count-Focused Evaluation

If the production goal is wheat-head counting, then object detection mAP may not be the only metric that matters. A detector can have imperfect boxes but still produce useful counts, or it can have visually plausible boxes but poor count accuracy.

Future work should evaluate counting behavior directly.

Possible metrics:

- absolute count error per image,
- percentage count error,
- mean absolute error,
- count error by density bucket,
- count error by lighting/contrast/blur bucket,
- correlation between predicted and true counts.

This would clarify whether the system is useful for counting, not just whether it scores well as an object detector.

### 8. Alternative Task Formulations

Bounding-box detection may not be the best formulation if the real production goal is accurate wheat-head counts in dense scenes.

Alternative approaches could include:

| Approach | Why It Might Help |
|---|---|
| Instance segmentation | More detailed object boundaries than boxes |
| Keypoint / center detection | Focuses on wheat-head centers rather than full boxes |
| Density-map counting | Estimates object density/counts without requiring perfect boxes |
| Hybrid detection + count calibration | Uses detections but corrects systematic count bias |

These alternatives would require additional research and possibly different labels, but they may be more appropriate for production-grade counting.

### 9. Monitoring and Inference Logging

A production system needs visibility into how it behaves after deployment. The current FastAPI app already reports request-level latency, which is a good start. A production-minded version would save structured inference logs and analyze them over time.

Useful logged fields could include:

- timestamp,
- model URI and version,
- confidence / IoU / image-size settings,
- image dimensions,
- detection count,
- latency,
- error status,
- request source or batch identifier,
- summary statistics for prediction confidence.

These logs could be used to monitor latency, detect unusual inputs, track detection-count drift, and identify images that should be reviewed or added to future training data.

### 10. Drift and Data Monitoring

Data drift occurs when production images differ from the data used to train and validate the model. For agricultural imagery, drift could come from new fields, cameras, countries, cultivars, growth stages, lighting conditions, blur, or seasonal changes.

Future work could compare production images against training/evaluation distributions using image-quality metrics already present in the CVDMS workflow.

Possible monitoring slices:

- lighting bucket,
- contrast bucket,
- blur bucket,
- color bucket,
- image size,
- detection count,
- confidence distribution,
- source/site/camera if available.

This would help decide when the model should be audited, retrained, or rolled back.

### 11. Human Review and Active Learning

A production agricultural vision system would likely need a human review workflow for uncertain or high-impact cases. Rather than treating every prediction as final, the system could flag images for review when confidence is low, count error is suspected, or input conditions differ from training data.

Human-reviewed examples could become targeted training data for future model versions.

Possible work:

- define uncertainty or anomaly rules,
- flag images with unusually high/low detection counts,
- flag images with many low-confidence detections,
- build a review queue,
- feed corrected examples back into CVDMS as a new dataset version.

### 12. Retraining and Model Rollback

Production systems need a safe way to improve models without breaking deployed behavior. MLflow model aliases already provide a useful foundation for this.

Future work could formalize:

- model candidate registration,
- validation gates before promotion,
- champion/challenger comparison,
- rollback to the previous champion,
- retraining from new CVDMS dataset versions,
- test reports attached to each model version.

This would make the MLflow registry part of a more realistic production lifecycle.

### 13. More Realistic MLflow Infrastructure

The current project uses MLflow locally, which is appropriate for a portfolio prototype. A more production-like setup would use a shared tracking server, a database backend, and a remote artifact store.

Possible future setup:

- MLflow tracking server,
- PostgreSQL backend store,
- S3 or MinIO artifact store,
- model registry aliases,
- Docker Compose or cloud deployment,
- promotion scripts that register and alias selected models.

This would better reflect how teams manage model metadata, artifacts, and deployment candidates in a shared environment.

### 14. Acceptance Tests and Production Gates

Before a model is promoted for production use, it should pass predefined acceptance tests. These tests should reflect the actual use case rather than only aggregate mAP.

Possible gates:

- minimum recall on dense scenes,
- maximum count error,
- maximum duplicate prediction rate,
- minimum performance on difficult lighting/contrast buckets,
- maximum p95 latency,
- successful API health check,
- successful rollback test,
- documented known failure modes.

These gates would make model promotion more disciplined and easier to explain.

## Suggested Expansion Project Scope

A strong expansion could focus on production hardening rather than simply training another model.

Recommended scope:

1. Build a label QA and failure-case audit workflow.
2. Run a deployment-threshold sweep.
3. Compare standard inference against tiled/SAHI-style inference.
4. Add structured inference logging and latency/detection monitoring.
5. Evaluate count error and dense-scene performance.
6. Document what improved, what did not, and what would still be needed for true production readiness.

## Summary

Project 3 demonstrates a complete train-to-deploy workflow. Project 4 would investigate what it takes to make that deployed system reliable.

The main production-readiness gap is not one missing hyperparameter. It is the broader engineering and validation work needed for dense agricultural small-object detection:

- better annotation QA,
- targeted failure-mode data,
- deployment-specific thresholding,
- tiled or higher-resolution inference,
- count-focused evaluation,
- monitoring and drift detection,
- retraining and rollback processes,
- explicit production acceptance criteria.