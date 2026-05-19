# YOLO Training Experiments

This document tracks the YOLO training experiments for the Global Wheat Head Detection project. The goal is to compare model configurations, training settings, validation/test performance, and practical runtime tradeoffs while keeping the workflow reproducible.

The current training pipeline uses Ultralytics YOLO with MLflow and TensorBoard logging. Each run saves standard YOLO outputs, `best.pt`, `last.pt`, a config snapshot, a training summary, and explicit post-training evaluations of the best checkpoint on both validation and test splits.

## Local Training Context

Training is currently being run on a local laptop GPU:

```text
NVIDIA GeForce RTX 4060 Laptop GPU
8 GB VRAM
```

Runtime and thermal behavior should be interpreted as local development observations rather than controlled hardware benchmarks. The same training configuration may perform differently on desktop GPUs, cloud GPUs, or machines with different cooling and storage setups.

Early speed tests showed that dataloader settings mattered significantly. In particular, increasing dataloader workers improved GPU utilization substantially compared with the initial low-worker setup. The first serious baseline configuration settled on:

```text
batch = 16
workers = 4
image size = 640
```

This provided a good practical balance between throughput, memory use, and stability on the local RTX 4060 Laptop GPU.

## Speed Tuning Summary

| Run                                  |   Model | Epochs | Image Size | Batch | Workers | Purpose                | Outcome                                                              |
| ------------------------------------ | ------: | -----: | ---------: | ----: | ------: | ---------------------- | -------------------------------------------------------------------- |
| Initial smoke run                    | YOLO11n |      5 |        640 |     8 |       0 | Verify training worked | Training worked, but GPU utilization was poor                        |
| `speed_001_yolo11n_e5_img640_b8_w4`  | YOLO11n |      5 |        640 |     8 |       4 | Test worker increase   | Better utilization than workers=0                                    |
| `speed_002_yolo11n_e5_img640_b16_w4` | YOLO11n |      5 |        640 |    16 |       4 | Test larger batch      | Best practical speed/stability tradeoff                              |
| `speed_003_yolo11n_e5_img640_b16_w8` | YOLO11n |      5 |        640 |    16 |       8 | Test more workers      | Slight training-phase improvement, but not enough end-to-end benefit |


`workers=8` was not chosen because it did not provide enough end-to-end improvement to justify the extra burstiness. Dataset caching was also not enabled yet because local image access was already fast enough after batch/worker tuning.

## Baseline Runs

The first baseline used YOLO11n as a lightweight starting point. It trained cleanly and established the initial validation/test performance gap. The next baseline moved to YOLO11s to test whether a larger model could improve performance on the harder test split without becoming impractical on the local GPU.

| Run                                           |   Model | Epochs | Batch | Workers | Val Precision | Val Recall | Val mAP50 | Val mAP50-95 | Test Precision | Test Recall | Test mAP50 | Test mAP50-95 |   Runtime |
| --------------------------------------------- | ------: | -----: | ----: | ------: | ------------: | ---------: | --------: | -----------: | -------------: | ----------: | ---------: | ------------: | --------: |
| `baseline_001_yolo11n_e30_img640_b16_w4`      | YOLO11n |     30 |    16 |       4 |        ~0.912 |     ~0.838 |    ~0.909 |       ~0.499 |         ~0.784 |      ~0.597 |     ~0.659 |        ~0.280 | ~31.7 min |
| `baseline_002_yolo11s_e30_img640_b16_w4`      | YOLO11s |     30 |    16 |       4 |         0.916 |      0.864 |     0.928 |        0.517 |          0.805 |       0.642 |      0.699 |         0.319 |   ~40 min |
| `baseline_003_yolo11s_e50_img640_b16_w4`      | YOLO11s |     50 |    16 |       4 |         0.922 |      0.862 |     0.929 |        0.526 |          0.818 |       0.638 |      0.706 |         0.309 |   ~60 min |
| `baseline_004_aug1_yolo11s_e30_img640_b16_w4` | YOLO11s |     30 |    16 |       4 |         0.918 |      0.858 |     0.925 |        0.522 |          0.809 |       0.634 |      0.702 |         0.303 |   ~38 min |

## Interpretation

- **YOLO11n to YOLO11s:** Improved both validation and test performance.

  ```text
  test mAP50-95: ~0.280 -> 0.319
  test recall:   ~0.597 -> 0.642
  ```
  
    This suggests that the larger YOLO11s model is learning useful additional structure rather than only improving on the validation split. The YOLO11s run costs more runtime and thermal load.