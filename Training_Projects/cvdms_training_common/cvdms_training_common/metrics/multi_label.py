"""
Multi-label classification metric helpers for CVDMS/PyTorch training projects.

These helpers assume standard multi-label classification:

    logits shape:   [batch_size, num_classes]
    targets shape:  [batch_size, num_classes]
    probabilities:  sigmoid(logits)
    predictions:    probabilities >= threshold
    loss style:     BCEWithLogitsLoss-style float targets

The model should output raw logits. Do not apply sigmoid before passing logits
to BCEWithLogitsLoss. Sigmoid is applied here for metrics, PR curves, AP/mAP,
and threshold-based predictions.
"""

from dataclasses import dataclass
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

    Confusion-count convention:
        column 0 = true positives
        column 1 = false positives
        column 2 = true negatives
        column 3 = false negatives
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
    Precision-recall summary for one multi-label class.
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
            "best_f1": self.best_f1,
            "best_precision": self.best_precision,
            "best_recall": self.best_recall,
            "best_threshold": self.best_threshold,
            "positive_count": self.positive_count,
        }

@dataclass(frozen=True)
class PrecisionRecallSummary:
    """
    Precision-recall and AP/mAP summary for one multi-label evaluation pass.
    """

    macro_average_precision: float | None
    micro_average_precision: float | None
    per_class: dict[str, PrecisionRecallClassSummary]

    def to_dict(self) -> dict[str, Any]:
        return {
            "macro_average_precision": self.macro_average_precision,
            "micro_average_precision": self.micro_average_precision,
            "per_class": {
                class_name: summary.to_dict()
                for class_name, summary in self.per_class.items()
            },
        }

@dataclass(frozen=True)
class MultiLabelClassificationMetrics:
    """
    Aggregate multi-label metrics for one evaluation pass.
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
    """
    if logits.ndim != 2:
        raise ValueError(
            f"Expected logits with shape [batch_size, num_classes], got {tuple(logits.shape)}"
        )

    if not torch.is_floating_point(logits):
        raise TypeError(f"logits must be floating point, got dtype={logits.dtype}")

    return torch.sigmoid(logits)

def predicted_labels_from_probabilities(
    probabilities: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Convert sigmoid probabilities to binary multi-label predictions.
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

    return probabilities >= threshold

def predicted_labels_from_logits(
    logits: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Convert raw multi-label logits to binary predictions.
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
    Count samples where the entire predicted label set exactly matches the target label set.
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

    A sample is counted correct only if every class prediction matches exactly.
    This metric is strict for multi-label classification.
    """
    _validate_logits_and_targets(logits, targets)

    total = int(targets.shape[0])
    if total == 0:
        raise ValueError("Cannot compute exact-match accuracy for an empty target tensor")

    return exact_match_count_from_logits(
        logits,
        targets,
        threshold=threshold,
    ) / total

def new_confusion_counts(
    *,
    num_classes: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """
    Create empty per-class multi-label confusion counts.

    Returned tensor shape is [num_classes, 4]:

        column 0 = true positives
        column 1 = false positives
        column 2 = true negatives
        column 3 = false negatives
    """
    if isinstance(num_classes, bool) or not isinstance(num_classes, int):
        raise TypeError(f"num_classes must be an int, got {type(num_classes).__name__}")

    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}")

    return torch.zeros(
        (num_classes, 4),
        dtype=torch.long,
        device=device,
    )

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
    """
    _validate_confusion_counts(confusion_counts)
    _validate_logits_and_targets(logits, targets)

    num_classes = int(confusion_counts.shape[0])
    if logits.shape[1] != num_classes:
        raise ValueError(
            f"logits num_classes={logits.shape[1]} does not match "
            f"confusion_counts num_classes={num_classes}"
        )

    preds = predicted_labels_from_logits(logits, threshold=threshold).detach().cpu()
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
    Compute micro/macro/weighted precision, recall, F1, hamming accuracy, and hamming loss.

    Macro averages treat each class equally.
    Weighted averages weight each class by ground-truth positive support.
    Micro averages aggregate TP/FP/FN across all classes.
    """
    _validate_confusion_counts(confusion_counts)
    _validate_threshold(threshold)

    if total_examples is not None:
        _validate_nonnegative_int(total_examples, "total_examples")

    if exact_match_count is not None:
        _validate_nonnegative_int(exact_match_count, "exact_match_count")

    if exact_match_count is not None and total_examples is None:
        raise ValueError("total_examples is required when exact_match_count is provided")

    if (
        exact_match_count is not None
        and total_examples is not None
        and exact_match_count > total_examples
    ):
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
    Compute per-class precision, recall, F1, accuracy, support, and confusion counts.
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

        out[class_idx] = {
            "precision": precision_i,
            "recall": recall_i,
            "f1": f1_i,
            "accuracy": accuracy_i,
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
    idx_to_class: dict[int, str],
) -> dict[str, dict[str, float | int | None]]:
    """
    Compute per-class metrics keyed by class name.
    """
    per_idx = per_class_metrics_from_confusion_counts(confusion_counts)

    named: dict[str, dict[str, float | int | None]] = {}
    for class_idx, metrics in per_idx.items():
        class_name = idx_to_class.get(class_idx)
        if class_name is None:
            raise ValueError(f"idx_to_class is missing class index {class_idx}")
        named[class_name] = metrics

    return named

def confusion_counts_to_nested_list(confusion_counts: torch.Tensor) -> list[list[int]]:
    """
    Convert confusion counts to a JSON-serializable nested list.

    Each row is [TP, FP, TN, FN] for one class.
    """
    _validate_confusion_counts(confusion_counts)

    return [
        [int(value) for value in row]
        for row in confusion_counts.to("cpu").tolist()
    ]

def precision_recall_summary(
    *,
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    idx_to_class: dict[int, str],
) -> PrecisionRecallSummary:
    """
    Compute precision-recall summaries from sigmoid probabilities and multi-hot targets.

    Args:
        probabilities:
            Tensor shaped [num_examples, num_classes], usually sigmoid(logits).
        targets:
            Tensor shaped [num_examples, num_classes] with binary multi-hot labels.
        idx_to_class:
            Mapping from class index to class name.

    Returns:
        PrecisionRecallSummary containing macro/micro average precision and
        per-class best-F1 threshold information.
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
        best = _best_f1_point(
            precision=precision,
            recall=recall,
            thresholds=thresholds,
        )

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

def make_precision_recall_figure(
    *,
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    idx_to_class: dict[int, str],
    title_prefix: str = "Precision-Recall Curves",
    annotate_best_f1: bool = False,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """
    Build a TensorBoard-ready multi-label precision-recall curve figure.

    For multi-label classifiers, these curves are threshold-free ranking
    diagnostics. Thresholded precision/recall/F1 is computed separately using
    a selected probability threshold.
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
        ax.plot(recall, precision, linewidth=1.8, label=label)

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
        f"Macro AP={_fmt_optional(summary.macro_average_precision)}, "
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

def logits_to_probabilities_and_targets(
    *,
    logits_batches: list[torch.Tensor],
    target_batches: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert collected logits/targets batches into sigmoid probabilities and targets.

    This helper is useful when an evaluation loop collects logits batch-by-batch.
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
    thresholds: list[float] | tuple[float, ...] | None = None,
) -> list[dict[str, float | int | None]]:
    """
    Evaluate threshold-dependent metrics over a list of thresholds.

    This is useful for choosing a global threshold after inspecting validation
    performance. It does not modify model training.
    """
    probs_np, targets_np = _validate_pr_inputs(probabilities, targets)

    if thresholds is None:
        thresholds = tuple(round(x, 2) for x in np.linspace(0.05, 0.95, 19))

    out: list[dict[str, float | int | None]] = []

    probs_t = torch.from_numpy(probs_np)
    targets_t = torch.from_numpy(targets_np).to(torch.float32)

    for threshold in thresholds:
        _validate_threshold(float(threshold))
        preds = predicted_labels_from_probabilities(
            probs_t,
            threshold=float(threshold),
        )

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

        item = summary.to_dict()
        out.append(item)

    return out

@torch.no_grad()
def update_confusion_counts_from_predictions(
    confusion_counts: torch.Tensor,
    *,
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Update per-class multi-label confusion counts in-place from binary predictions.
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
        raise TypeError(
            f"{name} must contain binary numeric labels, got dtype={targets.dtype}"
        )

    targets_cpu = targets.detach().cpu()

    if targets_cpu.numel() == 0:
        raise ValueError(f"{name} cannot be empty")

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
        raise TypeError(
            f"confusion_counts must have integer dtype, got {confusion_counts.dtype}"
        )

def _validate_threshold(threshold: float) -> None:
    if isinstance(threshold, bool) or not isinstance(threshold, int | float):
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

def _safe_divide(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
) -> torch.Tensor:
    return torch.where(
        denominator.abs() > _EPS,
        numerator / denominator.clamp_min(_EPS),
        torch.zeros_like(numerator, dtype=torch.float64),
    )

def _class_names_in_index_order(
    *,
    idx_to_class: dict[int, str],
    num_classes: int,
) -> list[str]:
    names: list[str] = []

    for idx in range(num_classes):
        class_name = idx_to_class.get(idx)
        if class_name is None:
            raise ValueError(f"idx_to_class is missing class index {idx}")
        names.append(class_name)

    return names