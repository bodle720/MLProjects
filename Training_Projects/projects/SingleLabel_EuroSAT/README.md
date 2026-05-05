# CVDMS EuroSAT Single-Label Classifier

This project is a small baseline implementation showing how to train a PyTorch model from dataset artifacts produced by **CVDMS**.

It consumes a CVDMS dataset version from S3, including:

- `metadata.json`
- train/validation/test JSONL manifests
- S3 image references
- version-specific class mappings

The goal is not to build a complex model, but to demonstrate the basic path from a versioned CVDMS dataset into a reproducible PyTorch training workflow.

## What this project demonstrates

This project trains a single-label image classifier using a pretrained ResNet18 model with staged transfer learning:

1. Train the classifier head only.
2. Unfreeze later backbone layers for light fine-tuning.
3. Unfreeze additional backbone layers for a final fine-tuning phase.

Each phase has its own configured learning rates, trainable layers, and maximum epoch count. Early stopping can shorten a phase, but the phase schedule remains explicit and controlled.

## Outputs

Training writes artifacts under `outputs/`, including:

- model checkpoints
- class map
- CVDMS training metadata
- model architecture summaries
- evaluation summary JSON
- TensorBoard logs

TensorBoard includes standard training diagnostics such as:

- train/validation loss
- train/validation accuracy
- train/validation precision, recall, and F1
- learning rates by parameter group
- trainable parameter counts
- validation confusion matrices
- one-vs-rest precision-recall curves

## Purpose

This project serves as a compact proof of concept for using CVDMS dataset outputs in a real PyTorch training workflow.

It is intended as a baseline example, not a final modeling benchmark. The main value is demonstrating that CVDMS can produce versioned dataset artifacts that are directly usable for downstream model training, evaluation, and experiment inspection.