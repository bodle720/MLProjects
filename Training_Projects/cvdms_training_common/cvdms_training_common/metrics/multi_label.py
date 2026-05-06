"""
Multi-label classification metric helpers for CVDMS/PyTorch training projects.

This module is intentionally verbose because multi-label metrics are easy to
misread. It assumes the standard PyTorch multi-label setup:

    logits shape:        [batch_size, num_classes]
    targets shape:       [batch_size, num_classes]
    targets values:      multi-hot binary labels, 0 or 1
    probabilities:       sigmoid(logits)
    predictions:         probabilities >= threshold
    training loss style: BCEWithLogitsLoss with float targets

The model should output raw logits. Do not add sigmoid to the model head before
BCEWithLogitsLoss. Sigmoid is applied here only for metrics, thresholded
predictions, precision-recall curves, AP/mAP, and probability diagnostics.

Metric interpretation notes
---------------------------
Multi-label classification is not the same as single-label multiclass
classification. Each class is an independent yes/no decision for every image.
For C classes and N images, there are N*C binary decisions.

Thresholded metrics use a fixed probability threshold, often 0.5:

    prediction[class_j] = sigmoid(logit[class_j]) >= threshold

From those thresholded predictions, each class gets its own binary confusion
counts:

    TP: target=1 and prediction=1
    FP: target=0 and prediction=1
    TN: target=0 and prediction=0
    FN: target=1 and prediction=0

Important aggregate metrics:

    hamming_accuracy:
        Fraction of all individual label decisions that are correct.
        This can be high when most labels are absent, so it should not be used
        alone for imbalanced multi-label datasets.

    subset_accuracy / exact-match accuracy:
        Fraction of samples where the entire predicted label set exactly
        matches the entire true label set. This is very strict and often low
        for realistic multi-label tasks.

    micro precision/recall/F1:
        Aggregates TP/FP/FN over all classes first, then computes the metric.
        Common classes dominate this score.

    macro precision/recall/F1:
        Computes the metric per class, then averages classes equally. This is
        better for seeing whether rare classes are being ignored.

    weighted precision/recall/F1:
        Computes the metric per class, then averages using true class support.
        This sits between macro and micro behavior.

Threshold-free ranking metrics:

    average_precision (AP):
        Per-class area-like summary of the precision-recall curve. It measures
        whether true examples for that class are ranked above false examples.
        AP does not require selecting one fixed threshold.

    macro_average_precision / mAP:
        Mean of per-class AP values across classes with at least one positive
        example. In this project, user-facing reports should call this mAP.

    micro_average_precision / micro-AP:
        AP after flattening all class decisions into one long binary problem.
        Common classes and common decisions influence it more than rare classes.

Probability heatmaps:

    conditional prediction probability heatmap:
        Row i selects samples where true class i is present. Cell (i, j) is the
        average predicted probability for class j on those samples. This is not
        a confusion matrix: high off-diagonal values may reflect legitimate
        label co-occurrence.

    false-association probability heatmap:
        Off-diagonal cell (i, j) selects samples where true class i is present
        and true class j is absent, then averages predicted probability for j.
        This better highlights false association. The diagonal remains the
        average probability for the true class when it is present.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from sklearn.metrics import average_precision_score, precision_recall_curve

_EPS = 1e-12
_TP_COL = 0
_FP_COL = 1
_TN_COL = 2
_FN_COL = 3

@dataclass(frozen=True)
class MultiLabelSummaryMetrics:
    """
    Aggregate metrics derived from per-class multi-label confusion counts.

    The confusion-count convention is [TP, FP, TN, FN] per class.

    All metrics in this dataclass are threshold-dependent except the metadata
    fields. They depend on the probability threshold used to convert sigmoid
    outputs into binary predictions.
    """

    threshold: float
    hamming_accuracy: float
    hamming_loss: float
    precision_micro: float
    recall_micro: float
    f1_micro: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    subset_accuracy: float | None
    total_examples: int | None
    total_label_decisions: int

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "threshold": self.threshold,
            "hamming_accuracy": self.hamming_accuracy,
            "hamming_loss": self.hamming_loss,
            "precision_micro": self.precision_micro,
            "recall_micro": self.recall_micro,
            "f1_micro": self.f1_micro,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "precision_weighted": self.precision_weighted,
            "recall_weighted": self.recall_weighted,
            "f1_weighted": self.f1_weighted,
            "subset_accuracy": self.subset_accuracy,
            "total_examples": self.total_examples,
            "total_label_decisions": self.total_label_decisions,
        }

@dataclass(frozen=True)
class PrecisionRecallClassSummary:
    """
    Precision-recall and AP summary for one class.

    average_precision:
        Per-class AP. This is threshold-free and measures ranking quality.

    best_f1 / best_threshold:
        Best F1 observed while sweeping that class's PR thresholds. This is a
        diagnostic only. It does not mean the model was trained with that
        threshold, and it can overfit if chosen directly from test data.
    """

    class_idx: int
    class_name: str
    average_precision: float | None
    best_f1: float | None
    best_precision: float | None
    best_recall: float | None
    best_threshold: float | None
    positive_count: int

    def to_dict(self) -> dict[str, float | int | str | None]:
        return {
            "class_idx": self.class_idx,
            "class_name": self.class_name,
            "average_precision": self.average_precision,
            "ap": self.average_precision,
            "best_f1": self.best_f1,
            "best_precision": self.best_precision,
            "best_recall": self.best_recall,
            "best_threshold": self.best_threshold,
            "positive_count": self.positive_count,
            "support": self.positive_count,
        }

@dataclass(frozen=True)
class PrecisionRecallSummary:
    """
    Precision-recall and AP/mAP summary for one evaluation pass.

    macro_average_precision is the project mAP value: mean per-class AP across
    classes that have at least one positive target in this evaluation split.
    micro_average_precision is micro-AP: AP after flattening all class decisions.
    """

    macro_average_precision: float | None
    micro_average_precision: float | None
    per_class: dict[str, PrecisionRecallClassSummary]

    @property
    def map(self) -> float | None:
        """Alias for macro average precision. User-facing name: mAP."""
        return self.macro_average_precision

    @property
    def micro_ap(self) -> float | None:
        """Alias for micro average precision. User-facing name: micro-AP."""
        return self.micro_average_precision

    def to_dict(self) -> dict[str, Any]:
        return {
            "macro_average_precision": self.macro_average_precision,
            "map": self.macro_average_precision,
            "micro_average_precision": self.micro_average_precision,
            "micro_ap": self.micro_average_precision,
            "per_class": {
                class_name: summary.to_dict()
                for class_name, summary in self.per_class.items()
            },
        }

@dataclass(frozen=True)
class MultiLabelClassificationMetrics:
    """
    Aggregate multi-label metrics for one evaluation pass.

    This object stores the thresholded confusion counts and enough metadata to
    produce aggregate threshold-dependent summaries. AP/mAP requires stored
    probabilities and targets, so it is computed by precision_recall_summary().
    """

    loss: float
    threshold: float
    total_examples: int
    confusion_counts: torch.Tensor
    exact_match_count: int | None = None

    @property
    def num_classes(self) -> int:
        return int(self.confusion_counts.shape[0])

    def summary(self) -> MultiLabelSummaryMetrics:
        return summary_metrics_from_confusion_counts(
            self.confusion_counts,
            threshold=self.threshold,
            total_examples=self.total_examples,
            exact_match_count=self.exact_match_count,
        )


def probabilities_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert raw multi-label logits to probabilities with sigmoid.

    Use this for metrics and diagnostics only. BCEWithLogitsLoss expects raw
    logits and applies the stable sigmoid/BCE calculation internally.
    """
    if logits.ndim != 2:
        raise ValueError(
            f"Expected logits with shape [batch_size, num_classes], got {tuple(logits.shape)}"
        )
    if not torch.is_floating_point(logits):
        raise TypeError(f"logits must be floating point, got dtype={logits.dtype}")
    if not bool(torch.isfinite(logits.detach()).all()):
        raise ValueError("logits must contain only finite values")
    return torch.sigmoid(logits)


def predicted_labels_from_probabilities(
    probabilities: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Convert sigmoid probabilities to binary multi-label predictions.

    A prediction is positive where probability >= threshold. The threshold is a
    modeling/reporting choice, not part of the network itself.
    """
    _validate_threshold(threshold)
    if probabilities.ndim != 2:
        raise ValueError(
            "Expected probabilities with shape [batch_size, num_classes], "
            f"got {tuple(probabilities.shape)}"
        )
    if not torch.is_floating_point(probabilities):
        raise TypeError(
            f"probabilities must be floating point, got dtype={probabilities.dtype}"
        )
    if not bool(torch.isfinite(probabilities.detach()).all()):
        raise ValueError("probabilities must contain only finite values")
    if bool((probabilities.detach() < 0.0).any()) or bool((probabilities.detach() > 1.0).any()):
        raise ValueError("probabilities must contain values in [0, 1]")
    return probabilities >= threshold


def predicted_labels_from_logits(
    logits: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Convert raw multi-label logits to binary predictions using sigmoid + threshold.
    """
    return predicted_labels_from_probabilities(
        probabilities_from_logits(logits),
        threshold=threshold,
    )


@torch.no_grad()
def exact_match_count_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> int:
    """
    Count samples where the entire predicted label set exactly matches the target set.

    This is the numerator for subset accuracy. It is strict: one missing label
    or one extra label makes the entire sample incorrect.
    """
    _validate_logits_and_targets(logits, targets)
    preds = predicted_labels_from_logits(logits, threshold=threshold)
    targets_bool = targets.to(dtype=torch.bool)
    return int((preds == targets_bool).all(dim=1).sum().item())


@torch.no_grad()
def exact_match_accuracy_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> float:
    """
    Compute subset accuracy, also called exact-match accuracy.

    Formula:
        exact matched samples / total samples

    Interpretation:
        Useful as a strict whole-label-set metric, but usually harsh for
        multi-label remote-sensing datasets where partial correctness matters.
    """
    _validate_logits_and_targets(logits, targets)
    total = int(targets.shape[0])
    if total == 0:
        raise ValueError("Cannot compute exact-match accuracy for an empty target tensor")
    return exact_match_count_from_logits(logits, targets, threshold=threshold) / total


def new_confusion_counts(
    *,
    num_classes: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """
    Create empty per-class multi-label confusion counts.

    Returned tensor shape is [num_classes, 4], with each row:
        [true_positive, false_positive, true_negative, false_negative]
    """
    if isinstance(num_classes, bool) or not isinstance(num_classes, int):
        raise TypeError(f"num_classes must be an int, got {type(num_classes).__name__}")
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}")
    return torch.zeros((num_classes, 4), dtype=torch.long, device=device)


@torch.no_grad()
def update_confusion_counts_from_logits(
    confusion_counts: torch.Tensor,
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Update per-class multi-label confusion counts in-place from raw logits.

    This is typically called batch-by-batch during evaluation. It converts
    logits to sigmoid probabilities, thresholds them, and adds TP/FP/TN/FN
    counts for each class.
    """
    _validate_confusion_counts(confusion_counts)
    _validate_logits_and_targets(logits, targets)
    num_classes = int(confusion_counts.shape[0])
    if logits.shape[1] != num_classes:
        raise ValueError(
            f"logits num_classes={logits.shape[1]} does not match "
            f"confusion_counts num_classes={num_classes}"
        )
    preds = predicted_labels_from_logits(logits, threshold=threshold)
    return update_confusion_counts_from_predictions(
        confusion_counts,
        predictions=preds,
        targets=targets,
    )


@torch.no_grad()
def update_confusion_counts_from_predictions(
    confusion_counts: torch.Tensor,
    *,
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Update per-class multi-label confusion counts in-place from binary predictions.

    predictions must already be boolean with shape [batch_size, num_classes].
    targets must be binary 0/1 with the same shape.
    """
    _validate_confusion_counts(confusion_counts)
    _validate_predictions_and_targets(predictions, targets)
    num_classes = int(confusion_counts.shape[0])
    if predictions.shape[1] != num_classes:
        raise ValueError(
            f"predictions num_classes={predictions.shape[1]} does not match "
            f"confusion_counts num_classes={num_classes}"
        )
    preds = predictions.detach().cpu().to(dtype=torch.bool)
    targets_bool = targets.detach().cpu().to(dtype=torch.bool)
    true_positive = (preds & targets_bool).sum(dim=0).to(torch.long)
    false_positive = (preds & ~targets_bool).sum(dim=0).to(torch.long)
    true_negative = (~preds & ~targets_bool).sum(dim=0).to(torch.long)
    false_negative = (~preds & targets_bool).sum(dim=0).to(torch.long)
    counts_cpu = confusion_counts.detach().cpu()
    counts_cpu[:, _TP_COL] += true_positive
    counts_cpu[:, _FP_COL] += false_positive
    counts_cpu[:, _TN_COL] += true_negative
    counts_cpu[:, _FN_COL] += false_negative
    confusion_counts.copy_(counts_cpu.to(confusion_counts.device))
    return confusion_counts


def summary_metrics_from_confusion_counts(
    confusion_counts: torch.Tensor,
    *,
    threshold: float = 0.5,
    total_examples: int | None = None,
    exact_match_count: int | None = None,
) -> MultiLabelSummaryMetrics:
    """
    Compute aggregate threshold-dependent multi-label metrics.

    Calculations:
        per-class precision = TP / (TP + FP)
        per-class recall    = TP / (TP + FN)
        per-class F1        = 2PR / (P + R)
        micro scores        = compute P/R/F1 after summing TP/FP/FN over classes
        macro scores        = mean of per-class scores over classes with support > 0
        weighted scores     = support-weighted mean of per-class scores
        hamming accuracy    = (TP + TN) / all label decisions
        hamming loss        = (FP + FN) / all label decisions
        subset accuracy     = exact_match_count / total_examples, if provided
    """
    _validate_confusion_counts(confusion_counts)
    _validate_threshold(threshold)
    if total_examples is not None:
        _validate_nonnegative_int(total_examples, "total_examples")
    if exact_match_count is not None:
        _validate_nonnegative_int(exact_match_count, "exact_match_count")
    if exact_match_count is not None and total_examples is None:
        raise ValueError("total_examples is required when exact_match_count is provided")
    if exact_match_count is not None and total_examples is not None and exact_match_count > total_examples:
        raise ValueError(
            f"exact_match_count={exact_match_count} cannot exceed total_examples={total_examples}"
        )

    counts = confusion_counts.to("cpu").to(torch.float64)
    true_positive = counts[:, _TP_COL]
    false_positive = counts[:, _FP_COL]
    true_negative = counts[:, _TN_COL]
    false_negative = counts[:, _FN_COL]
    support = true_positive + false_negative
    predicted = true_positive + false_positive
    total_label_decisions = int(counts.sum().item())
    if total_label_decisions == 0:
        raise ValueError("Cannot compute metrics from empty confusion counts")

    precision = _safe_divide(true_positive, predicted)
    recall = _safe_divide(true_positive, support)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    valid_support = support > 0
    if not bool(valid_support.any()):
        raise ValueError("Cannot compute metrics because every class has zero support")

    total_tp = true_positive.sum()
    total_fp = false_positive.sum()
    total_tn = true_negative.sum()
    total_fn = false_negative.sum()
    precision_micro = _safe_divide(total_tp, total_tp + total_fp)
    recall_micro = _safe_divide(total_tp, total_tp + total_fn)
    f1_micro = _safe_divide(
        2.0 * precision_micro * recall_micro,
        precision_micro + recall_micro,
    )
    weights = support / support.sum()
    hamming_accuracy = (total_tp + total_tn) / counts.sum()
    hamming_loss = (total_fp + total_fn) / counts.sum()
    subset_accuracy = None
    if exact_match_count is not None and total_examples is not None:
        subset_accuracy = exact_match_count / total_examples if total_examples > 0 else None

    return MultiLabelSummaryMetrics(
        threshold=threshold,
        hamming_accuracy=float(hamming_accuracy.item()),
        hamming_loss=float(hamming_loss.item()),
        precision_micro=float(precision_micro.item()),
        recall_micro=float(recall_micro.item()),
        f1_micro=float(f1_micro.item()),
        precision_macro=float(precision[valid_support].mean().item()),
        recall_macro=float(recall[valid_support].mean().item()),
        f1_macro=float(f1[valid_support].mean().item()),
        precision_weighted=float((precision * weights).sum().item()),
        recall_weighted=float((recall * weights).sum().item()),
        f1_weighted=float((f1 * weights).sum().item()),
        subset_accuracy=subset_accuracy,
        total_examples=total_examples,
        total_label_decisions=total_label_decisions,
    )


def per_class_metrics_from_confusion_counts(
    confusion_counts: torch.Tensor,
) -> dict[int, dict[str, float | int | None]]:
    """
    Compute per-class threshold-dependent metrics keyed by class index.

    Each class is treated as its own binary classifier. This table is the most
    direct way to see which classes are being missed, over-predicted, or handled
    well at the current threshold.
    """
    _validate_confusion_counts(confusion_counts)
    counts = confusion_counts.to("cpu").to(torch.float64)
    true_positive = counts[:, _TP_COL]
    false_positive = counts[:, _FP_COL]
    true_negative = counts[:, _TN_COL]
    false_negative = counts[:, _FN_COL]
    support = true_positive + false_negative
    predicted = true_positive + false_positive
    precision = _safe_divide(true_positive, predicted)
    recall = _safe_divide(true_positive, support)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    out: dict[int, dict[str, float | int | None]] = {}

    for class_idx in range(counts.shape[0]):
        tp = int(true_positive[class_idx].item())
        fp = int(false_positive[class_idx].item())
        tn = int(true_negative[class_idx].item())
        fn = int(false_negative[class_idx].item())
        support_i = tp + fn
        predicted_i = tp + fp
        total_i = tp + fp + tn + fn
        precision_i = None if predicted_i == 0 else float(precision[class_idx].item())
        recall_i = None if support_i == 0 else float(recall[class_idx].item())
        f1_i = None if support_i == 0 else float(f1[class_idx].item())
        accuracy_i = None if total_i == 0 else (tp + tn) / total_i
        specificity_i = None if (tn + fp) == 0 else tn / (tn + fp)
        out[class_idx] = {
            "precision": precision_i,
            "recall": recall_i,
            "f1": f1_i,
            "accuracy": accuracy_i,
            "specificity": specificity_i,
            "support": support_i,
            "predicted_positive": predicted_i,
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
        }

    return out


def per_class_metrics_with_names(
    confusion_counts: torch.Tensor,
    *,
    idx_to_class: Mapping[int | str, str],
) -> dict[str, dict[str, float | int | None]]:
    """
    Compute per-class threshold-dependent metrics keyed by class name.
    """
    per_idx = per_class_metrics_from_confusion_counts(confusion_counts)
    named: dict[str, dict[str, float | int | None]] = {}
    for class_idx, metrics in per_idx.items():
        named[_class_name(idx_to_class, class_idx)] = metrics
    return named


def per_class_metrics_table(
    confusion_counts: torch.Tensor,
    *,
    idx_to_class: Mapping[int | str, str],
    pr_summary: PrecisionRecallSummary | None = None,
) -> list[dict[str, float | int | str | None]]:
    """
    Build a flat CSV/JSON-friendly per-class metric table.

    Columns include thresholded counts/metrics plus AP and best-F1 threshold
    diagnostics when a PrecisionRecallSummary is provided.
    """
    per_idx = per_class_metrics_from_confusion_counts(confusion_counts)
    rows: list[dict[str, float | int | str | None]] = []

    for class_idx, metrics in per_idx.items():
        class_name = _class_name(idx_to_class, class_idx)
        pr_item = pr_summary.per_class.get(class_name) if pr_summary is not None else None
        row: dict[str, float | int | str | None] = {
            "class_idx": class_idx,
            "class_name": class_name,
            "support": metrics["support"],
            "predicted_positive": metrics["predicted_positive"],
            "true_positive": metrics["true_positive"],
            "false_positive": metrics["false_positive"],
            "true_negative": metrics["true_negative"],
            "false_negative": metrics["false_negative"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "accuracy": metrics["accuracy"],
            "specificity": metrics["specificity"],
            "average_precision": None,
            "ap": None,
            "best_f1": None,
            "best_precision": None,
            "best_recall": None,
            "best_threshold": None,
        }
        if pr_item is not None:
            row.update(
                {
                    "average_precision": pr_item.average_precision,
                    "ap": pr_item.average_precision,
                    "best_f1": pr_item.best_f1,
                    "best_precision": pr_item.best_precision,
                    "best_recall": pr_item.best_recall,
                    "best_threshold": pr_item.best_threshold,
                }
            )
        rows.append(row)

    return rows


def write_metric_rows_csv(
    rows: Sequence[Mapping[str, Any]],
    output_path: str | Path,
) -> None:
    """
    Write a list of flat metric dictionaries to CSV.

    None values are written as blank cells. This is useful for the per-class
    metrics table and threshold sweep table.
    """
    if not rows:
        raise ValueError("rows cannot be empty")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_clean(row.get(key)) for key in fieldnames})


def write_json(data: Any, output_path: str | Path) -> None:
    """
    Write JSON with NumPy/Torch scalar cleanup.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_clean(data), f, indent=2, sort_keys=True)


def confusion_counts_to_nested_list(confusion_counts: torch.Tensor) -> list[list[int]]:
    """
    Convert confusion counts to a JSON-serializable nested list.

    Each row is [TP, FP, TN, FN] for one class.
    """
    _validate_confusion_counts(confusion_counts)
    return [[int(value) for value in row] for row in confusion_counts.to("cpu").tolist()]


def precision_recall_summary(
    *,
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    idx_to_class: Mapping[int | str, str],
) -> PrecisionRecallSummary:
    """
    Compute precision-recall summaries from sigmoid probabilities and targets.

    Per-class AP measures ranking quality for each class independently. mAP is
    the mean of those per-class AP values across classes with positive examples.
    micro-AP flattens all class decisions before computing AP.
    """
    probs_np, targets_np = _validate_pr_inputs(probabilities, targets)
    num_classes = probs_np.shape[1]
    class_names = _class_names_in_index_order(idx_to_class=idx_to_class, num_classes=num_classes)
    per_class: dict[str, PrecisionRecallClassSummary] = {}
    average_precisions: list[float] = []

    for class_idx, class_name in enumerate(class_names):
        labels_binary = targets_np[:, class_idx]
        scores = probs_np[:, class_idx]
        positive_count = int(labels_binary.sum())
        if positive_count == 0:
            per_class[class_name] = PrecisionRecallClassSummary(
                class_idx=class_idx,
                class_name=class_name,
                average_precision=None,
                best_f1=None,
                best_precision=None,
                best_recall=None,
                best_threshold=None,
                positive_count=0,
            )
            continue
        precision, recall, thresholds = precision_recall_curve(labels_binary, scores)
        average_precision = float(average_precision_score(labels_binary, scores))
        best = _best_f1_point(precision=precision, recall=recall, thresholds=thresholds)
        average_precisions.append(average_precision)
        per_class[class_name] = PrecisionRecallClassSummary(
            class_idx=class_idx,
            class_name=class_name,
            average_precision=average_precision,
            best_f1=best["best_f1"],
            best_precision=best["best_precision"],
            best_recall=best["best_recall"],
            best_threshold=best["best_threshold"],
            positive_count=positive_count,
        )

    macro_ap = float(np.mean(average_precisions)) if average_precisions else None
    micro_ap = _micro_average_precision(y_true=targets_np, probabilities=probs_np)
    return PrecisionRecallSummary(
        macro_average_precision=macro_ap,
        micro_average_precision=micro_ap,
        per_class=per_class,
    )


def logits_to_probabilities_and_targets(
    *,
    logits_batches: list[torch.Tensor],
    target_batches: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert collected logits/targets batches into sigmoid probabilities and targets.

    Evaluation loops usually collect raw logits batch-by-batch. This helper
    concatenates them on CPU, validates shapes, and applies sigmoid once.
    """
    if not logits_batches:
        raise ValueError("logits_batches cannot be empty")
    if not target_batches:
        raise ValueError("target_batches cannot be empty")
    logits = torch.cat([batch.detach().cpu() for batch in logits_batches], dim=0)
    targets = torch.cat([batch.detach().cpu() for batch in target_batches], dim=0)
    _validate_logits_and_targets(logits, targets)
    return probabilities_from_logits(logits), targets


def threshold_sweep_summary(
    *,
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    thresholds: Sequence[float] | None = None,
) -> list[dict[str, float | int | None]]:
    """
    Evaluate threshold-dependent metrics over a list of thresholds.

    Use this on validation data to inspect whether 0.5 is too conservative or
    too permissive. Do not choose thresholds from test data if the test set is
    meant to remain an unbiased final evaluation.
    """
    probs_np, targets_np = _validate_pr_inputs(probabilities, targets)
    if thresholds is None:
        thresholds = tuple(round(x, 2) for x in np.linspace(0.05, 0.95, 19))
    out: list[dict[str, float | int | None]] = []
    probs_t = torch.from_numpy(probs_np)
    targets_t = torch.from_numpy(targets_np).to(torch.float32)

    for threshold in thresholds:
        _validate_threshold(float(threshold))
        preds = predicted_labels_from_probabilities(probs_t, threshold=float(threshold))
        counts = new_confusion_counts(num_classes=probs_t.shape[1])
        counts = update_confusion_counts_from_predictions(
            counts,
            predictions=preds,
            targets=targets_t,
        )
        exact_matches = int((preds == targets_t.to(dtype=torch.bool)).all(dim=1).sum().item())
        summary = summary_metrics_from_confusion_counts(
            counts,
            threshold=float(threshold),
            total_examples=int(targets_t.shape[0]),
            exact_match_count=exact_matches,
        )
        out.append(summary.to_dict())

    return out


def conditional_prediction_probability_matrix(
    *,
    probabilities: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the conditional prediction probability heatmap matrix.

    Row i selects samples where true class i is present. Cell (i, j) is:

        mean predicted probability for class j among samples with true class i

    This is a co-occurrence/association diagnostic, not a confusion matrix. A
    high off-diagonal can be correct if classes frequently occur together.

    Returns:
        matrix: shape [num_classes, num_classes], NaN for unsupported rows
        row_support: number of samples used for each row
    """
    probs_np, targets_np = _validate_pr_inputs(probabilities, targets)
    num_classes = probs_np.shape[1]
    matrix = np.full((num_classes, num_classes), np.nan, dtype=np.float64)
    row_support = np.zeros(num_classes, dtype=np.int64)

    for i in range(num_classes):
        mask = targets_np[:, i] == 1
        row_support[i] = int(mask.sum())
        if row_support[i] > 0:
            matrix[i, :] = probs_np[mask, :].mean(axis=0)

    return matrix, row_support


def false_association_probability_matrix(
    *,
    probabilities: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a false-association probability heatmap matrix.

    For off-diagonal cell (i, j), select samples where true class i is present
    and true class j is absent, then average predicted probability for class j.
    This asks: "when class i is truly present, how strongly does the model
    hallucinate class j when j is not actually labeled?"

    Diagonal cell (i, i) remains the mean predicted probability for class i
    among samples where true class i is present.

    Returns:
        matrix: shape [num_classes, num_classes], NaN where no samples qualify
        cell_support: number of samples used for each cell
    """
    probs_np, targets_np = _validate_pr_inputs(probabilities, targets)
    num_classes = probs_np.shape[1]
    matrix = np.full((num_classes, num_classes), np.nan, dtype=np.float64)
    cell_support = np.zeros((num_classes, num_classes), dtype=np.int64)

    for i in range(num_classes):
        true_i = targets_np[:, i] == 1
        for j in range(num_classes):
            if i == j:
                mask = true_i
            else:
                mask = true_i & (targets_np[:, j] == 0)
            cell_support[i, j] = int(mask.sum())
            if cell_support[i, j] > 0:
                matrix[i, j] = float(probs_np[mask, j].mean())

    return matrix, cell_support


def thresholded_true_predicted_cooccurrence_matrix(
    *,
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    normalize_rows: bool = False,
) -> np.ndarray:
    """
    Count or row-normalize true-label vs predicted-label co-occurrences.

    Row i means true class i is present. Column j means class j was predicted
    positive at the chosen threshold. Cell (i, j) counts how often both were
    true for the same sample. If normalize_rows=True, each row is divided by
    support of true class i.

    This is not a pure error matrix: off-diagonal values can reflect legitimate
    multi-label co-occurrence.
    """
    _validate_threshold(threshold)
    probs_np, targets_np = _validate_pr_inputs(probabilities, targets)
    preds_np = (probs_np >= threshold).astype(np.int64)
    matrix = targets_np.astype(np.int64).T @ preds_np
    if normalize_rows:
        support = targets_np.sum(axis=0).astype(np.float64)
        matrix = _safe_row_normalize(matrix.astype(np.float64), support)
    return matrix


def missed_vs_extra_label_matrix(
    *,
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    normalize_rows: bool = False,
) -> np.ndarray:
    """
    Count or row-normalize missed-label vs extra-label pairings.

    Row i means true class i was missed: target_i=1 and prediction_i=0.
    Column j means class j was an extra false positive: target_j=0 and
    prediction_j=1. Cell (i, j) counts how often both happened on the same
    sample.

    This is closer to an error-confusion diagnostic than the co-occurrence
    matrix because both axes are explicitly errors.
    """
    _validate_threshold(threshold)
    probs_np, targets_np = _validate_pr_inputs(probabilities, targets)
    preds_np = (probs_np >= threshold).astype(np.int64)
    missed = ((targets_np == 1) & (preds_np == 0)).astype(np.int64)
    extra = ((targets_np == 0) & (preds_np == 1)).astype(np.int64)
    matrix = missed.T @ extra
    if normalize_rows:
        missed_support = missed.sum(axis=0).astype(np.float64)
        matrix = _safe_row_normalize(matrix.astype(np.float64), missed_support)
    return matrix


def matrix_to_nested_list(matrix: np.ndarray, *, none_for_nan: bool = True) -> list[list[float | int | None]]:
    """
    Convert a matrix to a JSON-friendly nested list.
    """
    arr = np.asarray(matrix)
    out: list[list[float | int | None]] = []
    for row in arr.tolist():
        clean_row: list[float | int | None] = []
        for value in row:
            if none_for_nan and isinstance(value, float) and math.isnan(value):
                clean_row.append(None)
            elif isinstance(value, (np.integer,)):
                clean_row.append(int(value))
            elif isinstance(value, (np.floating,)):
                clean_row.append(float(value))
            else:
                clean_row.append(value)
        out.append(clean_row)
    return out


def make_precision_recall_figure(
    *,
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    idx_to_class: Mapping[int | str, str],
    title_prefix: str = "Precision-Recall Curves",
    annotate_best_f1: bool = False,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """
    Build one combined multi-label precision-recall curve figure.

    This is compact but can become visually busy with many classes. For Project
    2, prefer this as a quick overview and use make_precision_recall_small_multiples_figure
    or make_per_class_precision_recall_figures for class-by-class inspection.
    """
    probs_np, targets_np = _validate_pr_inputs(probabilities, targets)
    num_classes = probs_np.shape[1]
    class_names = _class_names_in_index_order(idx_to_class=idx_to_class, num_classes=num_classes)
    summary = precision_recall_summary(
        probabilities=probabilities,
        targets=targets,
        idx_to_class=idx_to_class,
    )
    if figsize is None:
        figsize = (10.0, 7.0)
    fig, ax = plt.subplots(figsize=figsize)

    for class_idx, class_name in enumerate(class_names):
        labels_binary = targets_np[:, class_idx]
        scores = probs_np[:, class_idx]
        if int(labels_binary.sum()) == 0:
            continue
        precision, recall, _ = precision_recall_curve(labels_binary, scores)
        class_summary = summary.per_class[class_name]
        ap = class_summary.average_precision
        label = f"{class_name} (AP={ap:.3f})" if ap is not None else class_name
        ax.plot(recall, precision, linewidth=1.6, label=label)
        if annotate_best_f1 and class_summary.best_recall is not None:
            ax.scatter(
                [class_summary.best_recall],
                [class_summary.best_precision],
                marker="*",
                s=90,
                zorder=3,
            )

    ax.set_title(
        f"{title_prefix}\n"
        f"Macro AP / mAP={_fmt_optional(summary.macro_average_precision)}, "
        f"Micro AP={_fmt_optional(summary.micro_average_precision)}"
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    return fig


def make_precision_recall_small_multiples_figure(
    *,
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    idx_to_class: Mapping[int | str, str],
    title_prefix: str = "Per-Class Precision-Recall Curves",
    annotate_best_f1: bool = False,
    max_cols: int = 4,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """
    Build a small-multiples PR figure with one subplot per class.

    This is usually easier to read than overlaying all classes on one plot.
    """
    probs_np, targets_np = _validate_pr_inputs(probabilities, targets)
    num_classes = probs_np.shape[1]
    class_names = _class_names_in_index_order(idx_to_class=idx_to_class, num_classes=num_classes)
    summary = precision_recall_summary(
        probabilities=probabilities,
        targets=targets,
        idx_to_class=idx_to_class,
    )
    ncols = min(max_cols, num_classes)
    nrows = int(math.ceil(num_classes / ncols))
    if figsize is None:
        figsize = (4.0 * ncols, 3.2 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)

    for class_idx, class_name in enumerate(class_names):
        ax = axes[class_idx // ncols][class_idx % ncols]
        labels_binary = targets_np[:, class_idx]
        scores = probs_np[:, class_idx]
        class_summary = summary.per_class[class_name]
        if int(labels_binary.sum()) > 0:
            precision, recall, _ = precision_recall_curve(labels_binary, scores)
            ax.plot(recall, precision, linewidth=1.8)
            if annotate_best_f1 and class_summary.best_recall is not None:
                ax.scatter(
                    [class_summary.best_recall],
                    [class_summary.best_precision],
                    marker="*",
                    s=80,
                    zorder=3,
                )
        ax.set_title(
            f"{class_name}\nAP={_fmt_optional(class_summary.average_precision)}, "
            f"support={class_summary.positive_count}",
            fontsize=9,
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.grid(True, alpha=0.3)

    for empty_idx in range(num_classes, nrows * ncols):
        axes[empty_idx // ncols][empty_idx % ncols].axis("off")

    fig.suptitle(
        f"{title_prefix} | mAP={_fmt_optional(summary.macro_average_precision)}, "
        f"micro-AP={_fmt_optional(summary.micro_average_precision)}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return fig


def make_per_class_precision_recall_figures(
    *,
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    idx_to_class: Mapping[int | str, str],
    title_prefix: str = "Precision-Recall Curve",
    annotate_best_f1: bool = True,
    figsize: tuple[float, float] = (7.0, 5.0),
) -> dict[str, Figure]:
    """
    Build one precision-recall figure per class.

    This is useful for saving individual readable PR plots to disk. Remember to
    close figures after saving if creating many plots in a long process.
    """
    probs_np, targets_np = _validate_pr_inputs(probabilities, targets)
    num_classes = probs_np.shape[1]
    class_names = _class_names_in_index_order(idx_to_class=idx_to_class, num_classes=num_classes)
    summary = precision_recall_summary(
        probabilities=probabilities,
        targets=targets,
        idx_to_class=idx_to_class,
    )
    figures: dict[str, Figure] = {}

    for class_idx, class_name in enumerate(class_names):
        labels_binary = targets_np[:, class_idx]
        scores = probs_np[:, class_idx]
        class_summary = summary.per_class[class_name]
        fig, ax = plt.subplots(figsize=figsize)
        if int(labels_binary.sum()) > 0:
            precision, recall, _ = precision_recall_curve(labels_binary, scores)
            ax.plot(recall, precision, linewidth=2.0)
            if annotate_best_f1 and class_summary.best_recall is not None:
                ax.scatter(
                    [class_summary.best_recall],
                    [class_summary.best_precision],
                    marker="*",
                    s=100,
                    zorder=3,
                    label=f"Best F1={_fmt_optional(class_summary.best_f1)}",
                )
                ax.legend(loc="lower left")
        ax.set_title(
            f"{title_prefix}: {class_name}\n"
            f"AP={_fmt_optional(class_summary.average_precision)}, "
            f"support={class_summary.positive_count}"
        )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        figures[class_name] = fig

    return figures


def make_per_class_metric_bar_figure(
    *,
    rows: Sequence[Mapping[str, Any]],
    metric_key: str,
    title: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """
    Build a bar chart for a per-class metric such as AP, F1, precision, or recall.
    """
    if not rows:
        raise ValueError("rows cannot be empty")
    class_names = [str(row["class_name"]) for row in rows]
    values = [np.nan if row.get(metric_key) is None else float(row[metric_key]) for row in rows]
    if figsize is None:
        figsize = (max(10.0, 0.55 * len(rows)), 5.5)
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(np.arange(len(rows)), values)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0 if _metric_is_unit_interval(metric_key) else max(1.0, np.nanmax(values)))
    ax.set_ylabel(ylabel or metric_key)
    ax.set_title(title or f"Per-Class {metric_key}")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def make_binary_confusion_grid_figure(
    *,
    confusion_counts: torch.Tensor,
    idx_to_class: Mapping[int | str, str],
    title_prefix: str = "Per-Class Binary Confusion Matrices",
    max_cols: int = 4,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """
    Build one 2x2 binary confusion matrix per class.

    Each mini-matrix uses:
        rows:    true negative class / true positive class
        columns: predicted negative / predicted positive

    Layout per class:
        [[TN, FP],
         [FN, TP]]
    """
    _validate_confusion_counts(confusion_counts)
    counts = confusion_counts.detach().cpu().to(torch.long)
    num_classes = int(counts.shape[0])
    class_names = _class_names_in_index_order(idx_to_class=idx_to_class, num_classes=num_classes)
    ncols = min(max_cols, num_classes)
    nrows = int(math.ceil(num_classes / ncols))
    if figsize is None:
        figsize = (4.0 * ncols, 3.4 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)

    for class_idx, class_name in enumerate(class_names):
        ax = axes[class_idx // ncols][class_idx % ncols]
        tp = int(counts[class_idx, _TP_COL].item())
        fp = int(counts[class_idx, _FP_COL].item())
        tn = int(counts[class_idx, _TN_COL].item())
        fn = int(counts[class_idx, _FN_COL].item())
        matrix = np.array([[tn, fp], [fn, tp]], dtype=np.int64)
        ax.imshow(matrix)
        ax.set_title(class_name, fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["True 0", "True 1"])
        for row_idx in range(2):
            for col_idx in range(2):
                ax.text(col_idx, row_idx, str(matrix[row_idx, col_idx]), ha="center", va="center")

    for empty_idx in range(num_classes, nrows * ncols):
        axes[empty_idx // ncols][empty_idx % ncols].axis("off")

    fig.suptitle(title_prefix, fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return fig


def make_conditional_prediction_probability_heatmap_figure(
    *,
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    idx_to_class: Mapping[int | str, str],
    title: str = "Conditional Prediction Probability Heatmap",
    annotate: bool = True,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """
    Plot the conditional prediction probability heatmap.

    Row i: true class i is present.
    Cell (i, j): average predicted probability for class j.
    """
    matrix, _ = conditional_prediction_probability_matrix(
        probabilities=probabilities,
        targets=targets,
    )
    return make_matrix_heatmap_figure(
        matrix=matrix,
        idx_to_class=idx_to_class,
        title=title,
        xlabel="Model output class probability",
        ylabel="True class present",
        colorbar_label="Mean predicted probability",
        annotate=annotate,
        value_format=".2f",
        figsize=figsize,
    )


def make_false_association_probability_heatmap_figure(
    *,
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    idx_to_class: Mapping[int | str, str],
    title: str = "False-Association Probability Heatmap",
    annotate: bool = True,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """
    Plot the false-association probability heatmap.

    Off-diagonal cell (i, j): mean p(j) where true i is present and true j is absent.
    Diagonal cell (i, i): mean p(i) where true i is present.
    """
    matrix, _ = false_association_probability_matrix(
        probabilities=probabilities,
        targets=targets,
    )
    return make_matrix_heatmap_figure(
        matrix=matrix,
        idx_to_class=idx_to_class,
        title=title,
        xlabel="Model output class probability",
        ylabel="True class present",
        colorbar_label="Mean predicted probability",
        annotate=annotate,
        value_format=".2f",
        figsize=figsize,
    )

def make_thresholded_cooccurrence_heatmap_figure(
    *,
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    idx_to_class: Mapping[int | str, str],
    threshold: float = 0.5,
    normalize_rows: bool = True,
    title: str | None = None,
    annotate: bool = True,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """
    Plot true-label vs predicted-label co-occurrence at a fixed threshold.
    """
    matrix = thresholded_true_predicted_cooccurrence_matrix(
        probabilities=probabilities,
        targets=targets,
        threshold=threshold,
        normalize_rows=normalize_rows,
    )
    return make_matrix_heatmap_figure(
        matrix=matrix,
        idx_to_class=idx_to_class,
        title=title or f"True-Label vs Predicted-Label Co-occurrence (threshold={threshold:.2f})",
        xlabel="Predicted positive class",
        ylabel="True class present",
        colorbar_label="Row-normalized rate" if normalize_rows else "Count",
        annotate=annotate,
        value_format=".2f" if normalize_rows else ".0f",
        figsize=figsize,
    )

def make_missed_vs_extra_heatmap_figure(
    *,
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    idx_to_class: Mapping[int | str, str],
    threshold: float = 0.5,
    normalize_rows: bool = True,
    title: str | None = None,
    annotate: bool = True,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """
    Plot missed true labels vs extra false-positive labels at a fixed threshold.
    """
    matrix = missed_vs_extra_label_matrix(
        probabilities=probabilities,
        targets=targets,
        threshold=threshold,
        normalize_rows=normalize_rows,
    )
    return make_matrix_heatmap_figure(
        matrix=matrix,
        idx_to_class=idx_to_class,
        title=title or f"Missed Labels vs Extra Labels (threshold={threshold:.2f})",
        xlabel="Extra false-positive class",
        ylabel="Missed true class",
        colorbar_label="Row-normalized rate" if normalize_rows else "Count",
        annotate=annotate,
        value_format=".2f" if normalize_rows else ".0f",
        figsize=figsize,
    )

def make_matrix_heatmap_figure(
    *,
    matrix: np.ndarray,
    idx_to_class: Mapping[int | str, str],
    title: str,
    xlabel: str,
    ylabel: str,
    colorbar_label: str,
    annotate: bool = True,
    value_format: str = ".2f",
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """
    Build a generic class-by-class heatmap figure.
    """
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"matrix must be square 2D, got shape={arr.shape}")
    num_classes = arr.shape[0]
    class_names = _class_names_in_index_order(idx_to_class=idx_to_class, num_classes=num_classes)
    if figsize is None:
        figsize = (max(8.0, 0.55 * num_classes), max(6.8, 0.50 * num_classes))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(arr)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    if annotate and num_classes <= 25:
        for row_idx in range(num_classes):
            for col_idx in range(num_classes):
                value = arr[row_idx, col_idx]
                text = "n/a" if np.isnan(value) else format(value, value_format)
                ax.text(col_idx, row_idx, text, ha="center", va="center", fontsize=7)

    fig.tight_layout()
    return fig

def save_figures(
    figures: Mapping[str, Figure],
    output_dir: str | Path,
    *,
    prefix: str = "",
    dpi: int = 150,
    close: bool = True,
) -> dict[str, str]:
    """
    Save named matplotlib figures as PNG files.

    Returns a mapping from figure name to output path string.
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    for name, fig in figures.items():
        safe_name = _safe_filename(name)
        filename = f"{prefix}{safe_name}.png" if prefix else f"{safe_name}.png"
        out_path = path / filename
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        saved[name] = str(out_path)
        if close:
            plt.close(fig)
    return saved

def _best_f1_point(
    *,
    precision: np.ndarray,
    recall: np.ndarray,
    thresholds: np.ndarray,
) -> dict[str, float | None]:
    if thresholds.size == 0:
        return {
            "best_f1": None,
            "best_precision": None,
            "best_recall": None,
            "best_threshold": None,
        }
    precision_aligned = precision[:-1]
    recall_aligned = recall[:-1]
    f1 = (2.0 * precision_aligned * recall_aligned) / (
        precision_aligned + recall_aligned + _EPS
    )
    best_idx = int(np.nanargmax(f1))
    return {
        "best_f1": float(f1[best_idx]),
        "best_precision": float(precision_aligned[best_idx]),
        "best_recall": float(recall_aligned[best_idx]),
        "best_threshold": float(thresholds[best_idx]),
    }

def _micro_average_precision(
    *,
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float | None:
    if int(y_true.sum()) == 0:
        return None
    return float(average_precision_score(y_true.ravel(), probabilities.ravel()))

def _validate_pr_inputs(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    if probabilities.ndim != 2:
        raise ValueError(
            "Expected probabilities with shape [num_examples, num_classes], "
            f"got {tuple(probabilities.shape)}"
        )
    if targets.ndim != 2:
        raise ValueError(
            f"Expected targets with shape [num_examples, num_classes], got {tuple(targets.shape)}"
        )
    if probabilities.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: probabilities={tuple(probabilities.shape)}, "
            f"targets={tuple(targets.shape)}"
        )
    if probabilities.shape[0] == 0:
        raise ValueError("probabilities/targets cannot be empty")
    if probabilities.shape[1] < 2:
        raise ValueError(
            f"Expected at least 2 classes, got num_classes={probabilities.shape[1]}"
        )
    if not torch.is_floating_point(probabilities):
        raise TypeError(
            f"probabilities must be floating point, got dtype={probabilities.dtype}"
        )
    _validate_binary_target_tensor(targets, "targets")
    probs_cpu = probabilities.detach().cpu()
    targets_cpu = targets.detach().cpu().to(torch.float32)
    if not bool(torch.isfinite(probs_cpu).all()):
        raise ValueError("probabilities must contain only finite values")
    if bool((probs_cpu < 0.0).any()) or bool((probs_cpu > 1.0).any()):
        raise ValueError("probabilities must contain values in [0, 1]")
    return probs_cpu.numpy(), targets_cpu.numpy().astype(np.int32)

def _validate_logits_and_targets(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    if logits.ndim != 2:
        raise ValueError(
            f"Expected logits with shape [batch_size, num_classes], got {tuple(logits.shape)}"
        )
    if targets.ndim != 2:
        raise ValueError(
            f"Expected targets with shape [batch_size, num_classes], got {tuple(targets.shape)}"
        )
    if logits.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: logits={tuple(logits.shape)}, targets={tuple(targets.shape)}"
        )
    if logits.shape[0] == 0:
        raise ValueError("logits/targets cannot be empty")
    if logits.shape[1] < 2:
        raise ValueError(f"Expected at least 2 classes, got num_classes={logits.shape[1]}")
    if not torch.is_floating_point(logits):
        raise TypeError(f"logits must be floating point, got dtype={logits.dtype}")
    if not bool(torch.isfinite(logits.detach()).all()):
        raise ValueError("logits must contain only finite values")
    _validate_binary_target_tensor(targets, "targets")

def _validate_predictions_and_targets(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    if predictions.ndim != 2:
        raise ValueError(
            f"Expected predictions with shape [batch_size, num_classes], "
            f"got {tuple(predictions.shape)}"
        )
    if targets.ndim != 2:
        raise ValueError(
            f"Expected targets with shape [batch_size, num_classes], got {tuple(targets.shape)}"
        )
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: predictions={tuple(predictions.shape)}, "
            f"targets={tuple(targets.shape)}"
        )
    if predictions.shape[0] == 0:
        raise ValueError("predictions/targets cannot be empty")
    if predictions.shape[1] < 2:
        raise ValueError(
            f"Expected at least 2 classes, got num_classes={predictions.shape[1]}"
        )
    if predictions.dtype != torch.bool:
        raise TypeError(f"predictions must have dtype=torch.bool, got {predictions.dtype}")
    _validate_binary_target_tensor(targets, "targets")

def _validate_binary_target_tensor(targets: torch.Tensor, name: str) -> None:
    if not (
        torch.is_floating_point(targets)
        or targets.dtype in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.bool,
        }
    ):
        raise TypeError(f"{name} must contain binary numeric labels, got dtype={targets.dtype}")
    targets_cpu = targets.detach().cpu()
    if targets_cpu.numel() == 0:
        raise ValueError(f"{name} cannot be empty")
    if torch.is_floating_point(targets_cpu) and not bool(torch.isfinite(targets_cpu).all()):
        raise ValueError(f"{name} must contain only finite values")
    valid = (targets_cpu == 0) | (targets_cpu == 1)
    if not bool(valid.all()):
        raise ValueError(f"{name} must contain only binary values 0/1")

def _validate_confusion_counts(confusion_counts: torch.Tensor) -> None:
    if confusion_counts.ndim != 2:
        raise ValueError(
            f"confusion_counts must be 2D, got shape={tuple(confusion_counts.shape)}"
        )
    if confusion_counts.shape[1] != 4:
        raise ValueError(
            "confusion_counts must have shape [num_classes, 4], got "
            f"shape={tuple(confusion_counts.shape)}"
        )
    if confusion_counts.shape[0] < 2:
        raise ValueError(
            f"confusion_counts must have at least 2 classes, got {confusion_counts.shape[0]}"
        )
    if confusion_counts.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError(f"confusion_counts must have integer dtype, got {confusion_counts.dtype}")
    if bool((confusion_counts.detach().cpu() < 0).any()):
        raise ValueError("confusion_counts cannot contain negative counts")

def _validate_threshold(threshold: float) -> None:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError(f"threshold must be numeric, got {type(threshold).__name__}")
    if not np.isfinite(float(threshold)):
        raise ValueError(f"threshold must be finite, got {threshold}")
    if float(threshold) < 0.0 or float(threshold) > 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

def _validate_nonnegative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")

def _fmt_optional(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"

def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return torch.where(
        denominator.abs() > _EPS,
        numerator / denominator.clamp_min(_EPS),
        torch.zeros_like(numerator, dtype=torch.float64),
    )

def _safe_row_normalize(matrix: np.ndarray, row_denominator: np.ndarray) -> np.ndarray:
    out = np.full_like(matrix, np.nan, dtype=np.float64)
    for row_idx in range(matrix.shape[0]):
        denom = float(row_denominator[row_idx])
        if denom > 0:
            out[row_idx, :] = matrix[row_idx, :] / denom
    return out

def _class_names_in_index_order(
    *,
    idx_to_class: Mapping[int | str, str],
    num_classes: int,
) -> list[str]:
    return [_class_name(idx_to_class, idx) for idx in range(num_classes)]

def _class_name(idx_to_class: Mapping[int | str, str], class_idx: int) -> str:
    if class_idx in idx_to_class:
        return str(idx_to_class[class_idx])
    str_idx = str(class_idx)
    if str_idx in idx_to_class:
        return str(idx_to_class[str_idx])
    raise ValueError(f"idx_to_class is missing class index {class_idx}")

def _metric_is_unit_interval(metric_key: str) -> bool:
    return metric_key in {
        "precision",
        "recall",
        "f1",
        "accuracy",
        "specificity",
        "average_precision",
        "ap",
        "best_f1",
        "best_precision",
        "best_recall",
    }

def _safe_filename(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    safe = "_".join(part for part in safe.split("_") if part)
    return safe[:120] or "figure"

def _json_clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_clean(value.tolist())
    if isinstance(value, torch.Tensor):
        return _json_clean(value.detach().cpu().tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    return value