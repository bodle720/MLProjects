"""
Single-label classification metric helpers for CVDMS/PyTorch training projects.

These helpers assume standard single-label, multi-class classification:

    logits shape:  [batch_size, num_classes]
    targets shape: [batch_size]
    prediction:    argmax(logits, dim=1)
    loss style:    CrossEntropyLoss-style integer targets

The model should output raw logits. Do not apply softmax before passing logits
to the loss. For precision-recall curves, logits are converted to probabilities
with softmax and analyzed one-vs-rest.
"""

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from sklearn.metrics import average_precision_score, precision_recall_curve

_EPS = 1e-12

@dataclass(frozen=True)
class SingleLabelSummaryMetrics:
    """
    Aggregate metrics derived from a confusion matrix.

    The confusion matrix convention is:
        rows    = ground truth classes
        columns = predicted classes
    """

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    total_examples: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "accuracy": self.accuracy,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "precision_weighted": self.precision_weighted,
            "recall_weighted": self.recall_weighted,
            "f1_weighted": self.f1_weighted,
            "total_examples": self.total_examples,
        }

@dataclass(frozen=True)
class PrecisionRecallClassSummary:
    """
    One-vs-rest precision-recall summary for one class.
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
    Precision-recall summary for a single-label multi-class evaluation pass.
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
class ClassificationMetrics:
    """
    Aggregate classification metrics for one evaluation pass.
    """

    loss: float
    accuracy: float
    total_examples: int
    confusion_matrix: torch.Tensor

    @property
    def num_classes(self) -> int:
        return int(self.confusion_matrix.shape[0])

    def per_class_accuracy(self) -> dict[int, float | None]:
        return per_class_accuracy_from_confusion_matrix(self.confusion_matrix)

    def summary(self) -> SingleLabelSummaryMetrics:
        return summary_metrics_from_confusion_matrix(self.confusion_matrix)

def predicted_classes_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert raw classification logits to predicted class indices.
    """
    if logits.ndim != 2:
        raise ValueError(
            f"Expected logits with shape [batch_size, num_classes], got {tuple(logits.shape)}"
        )

    return logits.argmax(dim=1)

def probabilities_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert raw logits to class probabilities with softmax.

    This is intended for analysis/visualization such as PR curves, not for
    CrossEntropyLoss.
    """
    if logits.ndim != 2:
        raise ValueError(
            f"Expected logits with shape [batch_size, num_classes], got {tuple(logits.shape)}"
        )

    if not torch.is_floating_point(logits):
        raise TypeError(f"logits must be floating point, got dtype={logits.dtype}")

    return torch.softmax(logits, dim=1)

@torch.no_grad()
def count_correct_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> int:
    """
    Count correct predictions from raw logits and integer class targets.
    """
    _validate_logits_and_targets(logits, targets)

    preds = predicted_classes_from_logits(logits)
    return int((preds == targets).sum().item())

@torch.no_grad()
def accuracy_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Compute batch accuracy from raw logits and integer class targets.
    """
    _validate_logits_and_targets(logits, targets)

    total = int(targets.numel())
    if total == 0:
        raise ValueError("Cannot compute accuracy for an empty target tensor")

    correct = count_correct_from_logits(logits, targets)
    return correct / total

def new_confusion_matrix(
    *,
    num_classes: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """
    Create an empty confusion matrix.

    Rows are ground truth classes.
    Columns are predicted classes.
    """
    if isinstance(num_classes, bool) or not isinstance(num_classes, int):
        raise TypeError(f"num_classes must be an int, got {type(num_classes).__name__}")

    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}")

    return torch.zeros(
        (num_classes, num_classes),
        dtype=torch.long,
        device=device,
    )

@torch.no_grad()
def update_confusion_matrix_from_logits(
    confusion_matrix: torch.Tensor,
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Update a confusion matrix in-place using raw logits and integer targets.

    Rows are ground truth classes.
    Columns are predicted classes.
    """
    _validate_confusion_matrix(confusion_matrix)
    _validate_logits_and_targets(logits, targets)

    num_classes = int(confusion_matrix.shape[0])
    preds = predicted_classes_from_logits(logits).to("cpu")
    targets_cpu = targets.to("cpu")

    if targets_cpu.numel() == 0:
        return confusion_matrix

    min_target = int(targets_cpu.min().item())
    max_target = int(targets_cpu.max().item())
    min_pred = int(preds.min().item())
    max_pred = int(preds.max().item())

    if min_target < 0 or max_target >= num_classes:
        raise ValueError(
            f"Target class indices out of range for num_classes={num_classes}: "
            f"min={min_target}, max={max_target}"
        )

    if min_pred < 0 or max_pred >= num_classes:
        raise ValueError(
            f"Predicted class indices out of range for num_classes={num_classes}: "
            f"min={min_pred}, max={max_pred}"
        )

    cm_cpu = confusion_matrix.to("cpu")

    for target, pred in zip(targets_cpu.view(-1), preds.view(-1)):
        cm_cpu[int(target), int(pred)] += 1

    confusion_matrix.copy_(cm_cpu.to(confusion_matrix.device))
    return confusion_matrix

def summary_metrics_from_confusion_matrix(
    confusion_matrix: torch.Tensor,
) -> SingleLabelSummaryMetrics:
    """
    Compute accuracy, macro precision/recall/F1, and weighted precision/recall/F1.

    Macro averages treat each class equally.
    Weighted averages weight each class by its ground-truth support.
    """
    _validate_confusion_matrix(confusion_matrix)

    cm = confusion_matrix.to("cpu").to(torch.float64)
    total = int(cm.sum().item())

    if total == 0:
        raise ValueError("Cannot compute metrics from an empty confusion matrix")

    true_positive = torch.diag(cm)
    support = cm.sum(dim=1)
    predicted = cm.sum(dim=0)

    precision = _safe_divide(true_positive, predicted)
    recall = _safe_divide(true_positive, support)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)

    valid_support = support > 0
    if not bool(valid_support.any()):
        raise ValueError("Cannot compute metrics because every class has zero support")

    weights = support / support.sum()
    accuracy = float(true_positive.sum().item() / total)

    return SingleLabelSummaryMetrics(
        accuracy=accuracy,
        precision_macro=float(precision[valid_support].mean().item()),
        recall_macro=float(recall[valid_support].mean().item()),
        f1_macro=float(f1[valid_support].mean().item()),
        precision_weighted=float((precision * weights).sum().item()),
        recall_weighted=float((recall * weights).sum().item()),
        f1_weighted=float((f1 * weights).sum().item()),
        total_examples=total,
    )

def per_class_metrics_from_confusion_matrix(
    confusion_matrix: torch.Tensor,
) -> dict[int, dict[str, float | int | None]]:
    """
    Compute per-class precision, recall, F1, accuracy, and support.
    """
    _validate_confusion_matrix(confusion_matrix)

    cm = confusion_matrix.to("cpu").to(torch.float64)
    true_positive = torch.diag(cm)
    support = cm.sum(dim=1)
    predicted = cm.sum(dim=0)

    precision = _safe_divide(true_positive, predicted)
    recall = _safe_divide(true_positive, support)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)

    out: dict[int, dict[str, float | int | None]] = {}

    for class_idx in range(cm.shape[0]):
        support_i = int(support[class_idx].item())

        if support_i == 0:
            out[class_idx] = {
                "precision": _none_if_no_predictions(float(predicted[class_idx].item())),
                "recall": None,
                "f1": None,
                "accuracy": None,
                "support": 0,
            }
            continue

        correct_i = float(true_positive[class_idx].item())
        out[class_idx] = {
            "precision": float(precision[class_idx].item()),
            "recall": float(recall[class_idx].item()),
            "f1": float(f1[class_idx].item()),
            "accuracy": correct_i / support_i,
            "support": support_i,
        }

    return out

def per_class_metrics_with_names(
    confusion_matrix: torch.Tensor,
    *,
    idx_to_class: dict[int, str],
) -> dict[str, dict[str, float | int | None]]:
    """
    Compute per-class metrics keyed by class name.
    """
    per_idx = per_class_metrics_from_confusion_matrix(confusion_matrix)

    named: dict[str, dict[str, float | int | None]] = {}
    for class_idx, metrics in per_idx.items():
        class_name = idx_to_class.get(class_idx)
        if class_name is None:
            raise ValueError(f"idx_to_class is missing class index {class_idx}")
        named[class_name] = metrics

    return named

def per_class_accuracy_from_confusion_matrix(
    confusion_matrix: torch.Tensor,
) -> dict[int, float | None]:
    """
    Compute per-class accuracy from a confusion matrix.
    """
    per_class = per_class_metrics_from_confusion_matrix(confusion_matrix)
    return {
        class_idx: metrics["accuracy"] if isinstance(metrics["accuracy"], float) else None
        for class_idx, metrics in per_class.items()
    }

def per_class_accuracy_with_names(
    confusion_matrix: torch.Tensor,
    *,
    idx_to_class: dict[int, str],
) -> dict[str, float | None]:
    """
    Compute per-class accuracy keyed by class name.
    """
    per_idx = per_class_accuracy_from_confusion_matrix(confusion_matrix)

    named: dict[str, float | None] = {}
    for class_idx, acc in per_idx.items():
        class_name = idx_to_class.get(class_idx)
        if class_name is None:
            raise ValueError(f"idx_to_class is missing class index {class_idx}")
        named[class_name] = acc

    return named

def confusion_matrix_to_nested_list(confusion_matrix: torch.Tensor) -> list[list[int]]:
    """
    Convert a confusion matrix tensor to a JSON-serializable nested list.
    """
    _validate_confusion_matrix(confusion_matrix)

    return [
        [int(value) for value in row]
        for row in confusion_matrix.to("cpu").tolist()
    ]

def make_confusion_matrix_figure(
    confusion_matrix: torch.Tensor,
    *,
    idx_to_class: dict[int, str],
    title_prefix: str = "Confusion Matrix",
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """
    Build a TensorBoard-ready matplotlib confusion-matrix figure.

    Rows are ground truth classes.
    Columns are predicted classes.

    Diagonal cells are annotated as:
        correct_count / row_total

    Off-diagonal cells are annotated with the raw count.
    """
    _validate_confusion_matrix(confusion_matrix)

    cm = confusion_matrix.to("cpu").to(torch.long)
    num_classes = int(cm.shape[0])
    class_names = _class_names_in_index_order(idx_to_class=idx_to_class, num_classes=num_classes)
    summary = summary_metrics_from_confusion_matrix(cm)

    if figsize is None:
        side = max(6.0, min(16.0, 0.75 * num_classes + 3.0))
        figsize = (side, side)

    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(cm.numpy(), interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(
        f"{title_prefix}\n"
        f"Accuracy={summary.accuracy:.4f}, "
        f"Macro F1={summary.f1_macro:.4f}"
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground truth class")

    tick_positions = list(range(num_classes))
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(class_names, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(class_names)

    max_count = int(cm.max().item()) if cm.numel() else 0
    threshold = max_count / 2.0 if max_count > 0 else 0.0

    for row_idx in range(num_classes):
        row_total = int(cm[row_idx, :].sum().item())

        for col_idx in range(num_classes):
            count = int(cm[row_idx, col_idx].item())
            text_color = "white" if count > threshold else "black"

            if row_idx == col_idx:
                label = f"{count}/{row_total}" if row_total > 0 else f"{count}/0"
            else:
                label = str(count)

            ax.text(
                col_idx,
                row_idx,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )

    ax.set_ylim(num_classes - 0.5, -0.5)
    fig.tight_layout()
    return fig

def precision_recall_summary(
    *,
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    idx_to_class: dict[int, str],
) -> PrecisionRecallSummary:
    """
    Compute one-vs-rest precision-recall summaries from probabilities and targets.

    Args:
        probabilities:
            Tensor shaped [num_examples, num_classes], usually softmax(logits).
        targets:
            Tensor shaped [num_examples] with integer class IDs.
        idx_to_class:
            Mapping from class index to class name.

    Returns:
        PrecisionRecallSummary containing macro/micro average precision and
        per-class best-F1 threshold information.
    """
    probs_np, targets_np = _validate_pr_inputs(probabilities, targets)
    num_classes = probs_np.shape[1]
    class_names = _class_names_in_index_order(idx_to_class=idx_to_class, num_classes=num_classes)
    y_true = _one_hot_targets(targets_np, num_classes)

    per_class: dict[str, PrecisionRecallClassSummary] = {}
    average_precisions: list[float] = []

    for class_idx, class_name in enumerate(class_names):
        labels_binary = y_true[:, class_idx]
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
    micro_ap = _micro_average_precision(y_true=y_true, probabilities=probs_np)

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
    Build a TensorBoard-ready one-vs-rest precision-recall curve figure.

    For single-label argmax classifiers, these curves are diagnostic ranking
    plots. They do not define the prediction rule, which remains argmax over
    class logits/probabilities.

    If annotate_best_f1=True, a star marks the threshold that maximizes one-vs-rest
    F1 for each class. This is mainly useful for threshold-based analysis, not for
    the default argmax classifier.
    """
    probs_np, targets_np = _validate_pr_inputs(probabilities, targets)
    num_classes = probs_np.shape[1]
    class_names = _class_names_in_index_order(idx_to_class=idx_to_class, num_classes=num_classes)
    y_true = _one_hot_targets(targets_np, num_classes)
    summary = precision_recall_summary(
        probabilities=probabilities,
        targets=targets,
        idx_to_class=idx_to_class,
    )

    if figsize is None:
        figsize = (10.0, 7.0)

    fig, ax = plt.subplots(figsize=figsize)

    for class_idx, class_name in enumerate(class_names):
        labels_binary = y_true[:, class_idx]
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
    Convert collected logits/targets batches into probabilities and targets.

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

    if targets.ndim != 1:
        raise ValueError(
            f"Expected targets with shape [num_examples], got {tuple(targets.shape)}"
        )

    if probabilities.shape[0] != targets.shape[0]:
        raise ValueError(
            f"Example count mismatch: probabilities={probabilities.shape[0]}, "
            f"targets={targets.shape[0]}"
        )

    if not torch.is_floating_point(probabilities):
        raise TypeError(
            f"probabilities must be floating point, got dtype={probabilities.dtype}"
        )

    targets_cpu = targets.detach().cpu()
    probs_cpu = probabilities.detach().cpu()

    if targets_cpu.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError(f"targets must contain integer class IDs, got dtype={targets.dtype}")

    num_classes = int(probs_cpu.shape[1])

    if targets_cpu.numel() == 0:
        raise ValueError("targets cannot be empty")

    min_target = int(targets_cpu.min().item())
    max_target = int(targets_cpu.max().item())

    if min_target < 0 or max_target >= num_classes:
        raise ValueError(
            f"Target class indices out of range for num_classes={num_classes}: "
            f"min={min_target}, max={max_target}"
        )

    return probs_cpu.numpy(), targets_cpu.numpy().astype(np.int64)

def _one_hot_targets(
    targets: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    y_true = np.zeros((targets.shape[0], num_classes), dtype=np.int32)
    y_true[np.arange(targets.shape[0]), targets] = 1
    return y_true

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

def _none_if_no_predictions(predicted_count: float) -> float | None:
    if predicted_count <= 0:
        return None
    return 0.0

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

def _validate_logits_and_targets(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    if logits.ndim != 2:
        raise ValueError(
            f"Expected logits with shape [batch_size, num_classes], got {tuple(logits.shape)}"
        )

    if targets.ndim != 1:
        raise ValueError(
            f"Expected targets with shape [batch_size], got {tuple(targets.shape)}"
        )

    if logits.shape[0] != targets.shape[0]:
        raise ValueError(
            f"Batch size mismatch: logits batch={logits.shape[0]}, "
            f"targets batch={targets.shape[0]}"
        )

    if not torch.is_floating_point(logits):
        raise TypeError(f"logits must be floating point, got dtype={logits.dtype}")

    if targets.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError(f"targets must contain integer class IDs, got dtype={targets.dtype}")

def _validate_confusion_matrix(confusion_matrix: torch.Tensor) -> None:
    if confusion_matrix.ndim != 2:
        raise ValueError(
            f"confusion_matrix must be 2D, got shape={tuple(confusion_matrix.shape)}"
        )

    if confusion_matrix.shape[0] != confusion_matrix.shape[1]:
        raise ValueError(
            "confusion_matrix must be square, got "
            f"shape={tuple(confusion_matrix.shape)}"
        )

    if confusion_matrix.shape[0] < 2:
        raise ValueError(
            f"confusion_matrix must have at least 2 classes, got {confusion_matrix.shape[0]}"
        )

    if confusion_matrix.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError(
            f"confusion_matrix must have integer dtype, got {confusion_matrix.dtype}"
        )