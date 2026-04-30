"""
Reusable single-label classification training loops for CVDMS/PyTorch projects.

These helpers are intended for standard single-label, multi-class classifiers:

    images -> model(images) -> logits [batch_size, num_classes]
    targets -> integer class IDs [batch_size]

They intentionally do not define model architectures. Individual projects should
build their own model, loss function, optimizer, scheduler, and transforms.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:  # pragma: no cover - older PyTorch fallback
    LRScheduler = Any  # type: ignore

from cvdms_training_common.checkpoints import (
    BestCheckpointTracker,
    save_checkpoint,
    save_class_map,
    save_cvdms_training_metadata,
    save_json_artifact,
)
from cvdms_training_common.metadata import CvdmsDatasetMetadata
from cvdms_training_common.metrics.single_label import (
    confusion_matrix_to_nested_list,
    logits_to_probabilities_and_targets,
    make_confusion_matrix_figure,
    make_precision_recall_figure,
    new_confusion_matrix,
    per_class_metrics_with_names,
    precision_recall_summary,
    summary_metrics_from_confusion_matrix,
    update_confusion_matrix_from_logits,
)

@dataclass(frozen=True)
class EpochResult:
    """
    Metrics from one train/eval epoch.
    """

    loss: float
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    total_examples: int
    elapsed_seconds: float
    confusion_matrix: torch.Tensor
    probabilities: torch.Tensor | None = None
    targets: torch.Tensor | None = None

    def to_summary(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "precision_weighted": self.precision_weighted,
            "recall_weighted": self.recall_weighted,
            "f1_weighted": self.f1_weighted,
            "total_examples": self.total_examples,
            "elapsed_seconds": self.elapsed_seconds,
            "confusion_matrix": confusion_matrix_to_nested_list(self.confusion_matrix),
        }

@dataclass
class FitResult:
    """
    Final structured result from fit_classifier().
    """

    history: dict[str, list[float]] = field(default_factory=dict)
    best_checkpoint: dict[str, Any] = field(default_factory=dict)
    test_metrics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "history": self.history,
            "best_checkpoint": self.best_checkpoint,
            "test_metrics": self.test_metrics,
        }

def train_one_epoch(
    *,
    model: nn.Module,
    dataloader,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    num_classes: int,
) -> EpochResult:
    """
    Train a classifier for one epoch.

    Uses:
        - model.train()
        - optimizer.zero_grad(set_to_none=True)
        - raw logits passed to loss function
        - argmax over logits for accuracy
        - sample-weighted average loss/accuracy
        - confusion-matrix-derived precision/recall/F1
    """
    model.train()
    start_time = time.perf_counter()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    confusion_matrix = new_confusion_matrix(num_classes=num_classes, device="cpu")

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = loss_fn(logits, targets)

        loss.backward()
        optimizer.step()

        batch_size = int(targets.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_examples += batch_size

        update_confusion_matrix_from_logits(
            confusion_matrix,
            logits.detach().cpu(),
            targets.detach().cpu(),
        )

    if total_examples == 0:
        raise ValueError("Training dataloader produced zero examples")

    elapsed = time.perf_counter() - start_time
    summary = summary_metrics_from_confusion_matrix(confusion_matrix)

    return EpochResult(
        loss=total_loss / total_examples,
        accuracy=total_correct / total_examples,
        precision_macro=summary.precision_macro,
        recall_macro=summary.recall_macro,
        f1_macro=summary.f1_macro,
        precision_weighted=summary.precision_weighted,
        recall_weighted=summary.recall_weighted,
        f1_weighted=summary.f1_weighted,
        total_examples=total_examples,
        elapsed_seconds=elapsed,
        confusion_matrix=confusion_matrix,
    )

@torch.inference_mode()
def evaluate_classifier(
    *,
    model: nn.Module,
    dataloader,
    loss_fn: nn.Module,
    device: torch.device,
    num_classes: int,
    collect_pr_data: bool = False,
) -> EpochResult:
    """
    Evaluate a classifier on validation or test data.

    Uses:
        - model.eval()
        - torch.inference_mode()
        - sample-weighted average loss/accuracy
        - confusion-matrix-derived precision/recall/F1

    If collect_pr_data=True, also collects probabilities and targets for
    diagnostic one-vs-rest precision-recall plots.
    """
    model.eval()
    start_time = time.perf_counter()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    confusion_matrix = new_confusion_matrix(num_classes=num_classes, device="cpu")
    logits_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # no need to apply softmax first, only need that if want probabilities:
        # argmax(logits) is equivalent to argmax(softmax(logits))
        logits = model(images)
        loss = loss_fn(logits, targets)

        batch_size = int(targets.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_examples += batch_size

        logits_cpu = logits.detach().cpu()
        targets_cpu = targets.detach().cpu()

        update_confusion_matrix_from_logits(
            confusion_matrix,
            logits_cpu,
            targets_cpu,
        )

        if collect_pr_data:
            logits_batches.append(logits_cpu)
            target_batches.append(targets_cpu)

    if total_examples == 0:
        raise ValueError("Evaluation dataloader produced zero examples")

    probabilities = None
    collected_targets = None
    if collect_pr_data:
        probabilities, collected_targets = logits_to_probabilities_and_targets(
            logits_batches=logits_batches,
            target_batches=target_batches,
        )

    elapsed = time.perf_counter() - start_time
    summary = summary_metrics_from_confusion_matrix(confusion_matrix)

    return EpochResult(
        loss=total_loss / total_examples,
        accuracy=total_correct / total_examples,
        precision_macro=summary.precision_macro,
        recall_macro=summary.recall_macro,
        f1_macro=summary.f1_macro,
        precision_weighted=summary.precision_weighted,
        recall_weighted=summary.recall_weighted,
        f1_weighted=summary.f1_weighted,
        total_examples=total_examples,
        elapsed_seconds=elapsed,
        confusion_matrix=confusion_matrix,
        probabilities=probabilities,
        targets=collected_targets,
    )

def fit_classifier(
    *,
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    cvdms_metadata: CvdmsDatasetMetadata,
    model_name: str,
    output_dir: str | Path,
    epochs: int,
    device: torch.device | str | None = None,
    metadata_uri: str | None = None,
    scheduler: LRScheduler | None = None,
    scheduler_step_on: str = "epoch",
    tensorboard_dir: str | Path | None = None,
    hyperparameters: dict[str, Any] | None = None,
    extra_checkpoint_metadata: dict[str, Any] | None = None,
    best_metric_name: str = "val_accuracy",
    best_metric_mode: str = "max",
    log_train_confusion_matrix: bool = False,
    log_val_confusion_matrix: bool = True,
    log_val_precision_recall_curve: bool = True,
    log_test_figures: bool = True,
    print_fn: Callable[[str], None] | None = print,
    train_sampler: Any | None = None,
    is_main_process: bool = True,
) -> FitResult:
    """
    Fit a standard classifier using train/val/test loaders.

    This function:
        - trains for N epochs
        - evaluates on validation each epoch
        - logs TensorBoard scalar timelines for loss, accuracy, precision, recall, F1, and LR
        - optionally logs validation confusion matrix images per epoch
        - optionally logs validation one-vs-rest PR curve images per epoch
        - saves last_checkpoint.pt every epoch
        - saves best_checkpoint.pt using the selected validation metric
        - evaluates on test once at the end
        - saves evaluation_summary.json
        - saves CVDMS training metadata and class map artifacts

    Args:
        scheduler_step_on:
            "epoch" means scheduler.step() after each epoch.
            "val_loss" means scheduler.step(val_loss), useful for ReduceLROnPlateau.
            "none" disables scheduler stepping even if scheduler is provided.

    This version is “DDP-safe” for logging/checkpointing, but it does not yet all-reduce metrics across ranks. So in a
    true multi-process DDP run, each rank computes metrics on its local shard. Since only rank 0 logs/saves, the saved
    metrics would reflect rank 0’s shard unless we later add distributed metric aggregation. That is fine for now; before
    real DDP training, we should add all-reduce support for loss totals, correct counts, total examples, and confusion matrices.
    """
    _validate_epochs(epochs)

    resolved_device = _resolve_device(device)
    model.to(resolved_device)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    hparams = dict(hyperparameters or {})
    extra = dict(extra_checkpoint_metadata or {})

    if is_main_process:
        if metadata_uri is not None:
            save_cvdms_training_metadata(
                path=output_path / "cvdms_training_metadata.json",
                cvdms_metadata=cvdms_metadata,
                metadata_uri=metadata_uri,
                model_name=model_name,
                hyperparameters=hparams,
                extra=extra,
            )

        save_class_map(
            path=output_path / "class_map.json",
            cvdms_metadata=cvdms_metadata,
        )

    writer = (
        SummaryWriter(log_dir=str(tensorboard_dir))
        if tensorboard_dir and is_main_process
        else None
    )

    history = _new_history()

    best_tracker = BestCheckpointTracker(
        output_dir=output_path,
        filename="best_checkpoint.pt",
        metric_name=best_metric_name,
        mode=best_metric_mode,
    )

    try:
        for epoch in range(1, epochs + 1):
            if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)

            train_result = train_one_epoch(
                model=model,
                dataloader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=resolved_device,
                num_classes=cvdms_metadata.num_classes,
            )

            val_result = evaluate_classifier(
                model=model,
                dataloader=val_loader,
                loss_fn=loss_fn,
                device=resolved_device,
                num_classes=cvdms_metadata.num_classes,
                collect_pr_data=log_val_precision_recall_curve,
            )

            _append_history(
                history=history,
                split="train",
                result=train_result,
            )
            _append_history(
                history=history,
                split="val",
                result=val_result,
            )

            _log_epoch_to_tensorboard(
                writer=writer,
                epoch=epoch,
                train_result=train_result,
                val_result=val_result,
                optimizer=optimizer,
                cvdms_metadata=cvdms_metadata,
                log_train_confusion_matrix=log_train_confusion_matrix,
                log_val_confusion_matrix=log_val_confusion_matrix,
                log_val_precision_recall_curve=log_val_precision_recall_curve,
            )

            if scheduler is not None:
                _step_scheduler(
                    scheduler=scheduler,
                    scheduler_step_on=scheduler_step_on,
                    val_loss=val_result.loss,
                )

            if is_main_process:
                save_checkpoint(
                    path=output_path / "last_checkpoint.pt",
                    model=model,
                    epoch=epoch,
                    cvdms_metadata=cvdms_metadata,
                    model_name=model_name,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metric_name="val_accuracy",
                    metric_value=val_result.accuracy,
                    hyperparameters=hparams,
                    extra=extra,
                )

                best_metric_value = _select_best_metric_value(
                    metric_name=best_metric_name,
                    train_result=train_result,
                    val_result=val_result,
                )

                is_new_best = best_tracker.maybe_save(
                    metric_value=best_metric_value,
                    model=model,
                    epoch=epoch,
                    cvdms_metadata=cvdms_metadata,
                    model_name=model_name,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    hyperparameters=hparams,
                    extra=extra,
                )

                if print_fn is not None:
                    marker = " *best*" if is_new_best else ""
                    print_fn(_format_epoch_log(epoch, epochs, train_result, val_result) + marker)

        test_result = evaluate_classifier(
            model=model,
            dataloader=test_loader,
            loss_fn=loss_fn,
            device=resolved_device,
            num_classes=cvdms_metadata.num_classes,
            collect_pr_data=True,
        )

        test_summary = _build_result_summary(
            result=test_result,
            cvdms_metadata=cvdms_metadata,
            include_precision_recall=True,
        )

        if writer is not None:
            _log_test_to_tensorboard(
                writer=writer,
                test_result=test_result,
                cvdms_metadata=cvdms_metadata,
                log_test_figures=log_test_figures,
            )

        if is_main_process:
            save_json_artifact(
                path=output_path / "evaluation_summary.json",
                payload={
                    "history": history,
                    "best_checkpoint": best_tracker.summary(),
                    "test_metrics": test_summary,
                },
            )

            if print_fn is not None:
                print_fn(
                    " | ".join(
                        [
                            "final_test",
                            f"test_loss={test_result.loss:.4f}",
                            f"test_acc={test_result.accuracy:.4f}",
                            f"test_precision={test_result.precision_macro:.4f}",
                            f"test_recall={test_result.recall_macro:.4f}",
                            f"test_f1={test_result.f1_macro:.4f}",
                        ]
                    )
                )

        return FitResult(
            history=history,
            best_checkpoint=best_tracker.summary(),
            test_metrics=test_summary,
        )

    finally:
        if writer is not None:
            writer.flush()
            writer.close()

def _new_history() -> dict[str, list[float]]:
    return {
        "train_loss": [],
        "train_accuracy": [],
        "train_precision_macro": [],
        "train_recall_macro": [],
        "train_f1_macro": [],
        "train_precision_weighted": [],
        "train_recall_weighted": [],
        "train_f1_weighted": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision_macro": [],
        "val_recall_macro": [],
        "val_f1_macro": [],
        "val_precision_weighted": [],
        "val_recall_weighted": [],
        "val_f1_weighted": [],
    }

def _append_history(
    *,
    history: dict[str, list[float]],
    split: str,
    result: EpochResult,
) -> None:
    history[f"{split}_loss"].append(result.loss)
    history[f"{split}_accuracy"].append(result.accuracy)
    history[f"{split}_precision_macro"].append(result.precision_macro)
    history[f"{split}_recall_macro"].append(result.recall_macro)
    history[f"{split}_f1_macro"].append(result.f1_macro)
    history[f"{split}_precision_weighted"].append(result.precision_weighted)
    history[f"{split}_recall_weighted"].append(result.recall_weighted)
    history[f"{split}_f1_weighted"].append(result.f1_weighted)

def _build_result_summary(
    *,
    result: EpochResult,
    cvdms_metadata: CvdmsDatasetMetadata,
    include_precision_recall: bool,
) -> dict[str, Any]:
    summary = {
        "loss": result.loss,
        "accuracy": result.accuracy,
        "precision_macro": result.precision_macro,
        "recall_macro": result.recall_macro,
        "f1_macro": result.f1_macro,
        "precision_weighted": result.precision_weighted,
        "recall_weighted": result.recall_weighted,
        "f1_weighted": result.f1_weighted,
        "total_examples": result.total_examples,
        "elapsed_seconds": result.elapsed_seconds,
        "confusion_matrix": confusion_matrix_to_nested_list(result.confusion_matrix),
        "per_class_metrics": per_class_metrics_with_names(
            result.confusion_matrix,
            idx_to_class=cvdms_metadata.idx_to_class,
        ),
    }

    if include_precision_recall and result.probabilities is not None and result.targets is not None:
        summary["precision_recall"] = precision_recall_summary(
            probabilities=result.probabilities,
            targets=result.targets,
            idx_to_class=cvdms_metadata.idx_to_class,
        ).to_dict()

    return summary

def _log_epoch_to_tensorboard(
    *,
    writer: SummaryWriter | None,
    epoch: int,
    train_result: EpochResult,
    val_result: EpochResult,
    optimizer: Optimizer,
    cvdms_metadata: CvdmsDatasetMetadata,
    log_train_confusion_matrix: bool,
    log_val_confusion_matrix: bool,
    log_val_precision_recall_curve: bool,
) -> None:
    if writer is None:
        return

    _log_scalar_metrics(writer=writer, split="train", result=train_result, step=epoch)
    _log_scalar_metrics(writer=writer, split="val", result=val_result, step=epoch)

    lr = _get_current_learning_rate(optimizer)
    if lr is not None:
        writer.add_scalar("learning_rate", lr, epoch)

    if log_train_confusion_matrix:
        _log_confusion_matrix_figure(
            writer=writer,
            tag="confusion_matrix/train",
            confusion_matrix=train_result.confusion_matrix,
            cvdms_metadata=cvdms_metadata,
            step=epoch,
            title_prefix=f"Train Confusion Matrix - Epoch {epoch}",
        )

    if log_val_confusion_matrix:
        _log_confusion_matrix_figure(
            writer=writer,
            tag="confusion_matrix/val",
            confusion_matrix=val_result.confusion_matrix,
            cvdms_metadata=cvdms_metadata,
            step=epoch,
            title_prefix=f"Validation Confusion Matrix - Epoch {epoch}",
        )

    if log_val_precision_recall_curve and val_result.probabilities is not None:
        _log_precision_recall_figure(
            writer=writer,
            tag="precision_recall/val",
            result=val_result,
            cvdms_metadata=cvdms_metadata,
            step=epoch,
            title_prefix=f"Validation One-vs-Rest PR Curves - Epoch {epoch}",
        )

def _log_test_to_tensorboard(
    *,
    writer: SummaryWriter,
    test_result: EpochResult,
    cvdms_metadata: CvdmsDatasetMetadata,
    log_test_figures: bool,
) -> None:
    _log_scalar_metrics(writer=writer, split="test", result=test_result, step=0)

    if not log_test_figures:
        return

    _log_confusion_matrix_figure(
        writer=writer,
        tag="confusion_matrix/test",
        confusion_matrix=test_result.confusion_matrix,
        cvdms_metadata=cvdms_metadata,
        step=0,
        title_prefix="Test Confusion Matrix",
    )

    if test_result.probabilities is not None:
        _log_precision_recall_figure(
            writer=writer,
            tag="precision_recall/test",
            result=test_result,
            cvdms_metadata=cvdms_metadata,
            step=0,
            title_prefix="Test One-vs-Rest PR Curves",
        )

def _log_scalar_metrics(
    *,
    writer: SummaryWriter,
    split: str,
    result: EpochResult,
    step: int,
) -> None:
    writer.add_scalar(f"loss/{split}", result.loss, step)
    writer.add_scalar(f"accuracy/{split}", result.accuracy, step)
    writer.add_scalar(f"precision_macro/{split}", result.precision_macro, step)
    writer.add_scalar(f"recall_macro/{split}", result.recall_macro, step)
    writer.add_scalar(f"f1_macro/{split}", result.f1_macro, step)
    writer.add_scalar(f"precision_weighted/{split}", result.precision_weighted, step)
    writer.add_scalar(f"recall_weighted/{split}", result.recall_weighted, step)
    writer.add_scalar(f"f1_weighted/{split}", result.f1_weighted, step)

def _log_confusion_matrix_figure(
    *,
    writer: SummaryWriter,
    tag: str,
    confusion_matrix: torch.Tensor,
    cvdms_metadata: CvdmsDatasetMetadata,
    step: int,
    title_prefix: str,
) -> None:
    fig = make_confusion_matrix_figure(
        confusion_matrix,
        idx_to_class=cvdms_metadata.idx_to_class,
        title_prefix=title_prefix,
    )

    try:
        writer.add_figure(tag, fig, step)
    finally:
        plt.close(fig)

def _log_precision_recall_figure(
    *,
    writer: SummaryWriter,
    tag: str,
    result: EpochResult,
    cvdms_metadata: CvdmsDatasetMetadata,
    step: int,
    title_prefix: str,
) -> None:
    if result.probabilities is None or result.targets is None:
        return

    fig = make_precision_recall_figure(
        probabilities=result.probabilities,
        targets=result.targets,
        idx_to_class=cvdms_metadata.idx_to_class,
        title_prefix=title_prefix,
        annotate_best_f1=False,
    )

    try:
        writer.add_figure(tag, fig, step)
    finally:
        plt.close(fig)

def _format_epoch_log(
    epoch: int,
    epochs: int,
    train_result: EpochResult,
    val_result: EpochResult,
) -> str:
    return " | ".join(
        [
            f"epoch={epoch}/{epochs}",
            f"train_loss={train_result.loss:.4f}",
            f"train_acc={train_result.accuracy:.4f}",
            f"train_f1={train_result.f1_macro:.4f}",
            f"val_loss={val_result.loss:.4f}",
            f"val_acc={val_result.accuracy:.4f}",
            f"val_f1={val_result.f1_macro:.4f}",
            f"time={train_result.elapsed_seconds + val_result.elapsed_seconds:.1f}s",
        ]
    )

def _get_current_learning_rate(optimizer: Optimizer) -> float | None:
    if not optimizer.param_groups:
        return None

    lr = optimizer.param_groups[0].get("lr")
    if lr is None:
        return None

    return float(lr)

def _step_scheduler(
    *,
    scheduler: LRScheduler,
    scheduler_step_on: str,
    val_loss: float,
) -> None:
    if scheduler_step_on == "none":
        return

    if scheduler_step_on == "epoch":
        scheduler.step()
        return

    if scheduler_step_on == "val_loss":
        scheduler.step(val_loss)
        return

    raise ValueError(
        "scheduler_step_on must be one of {'epoch', 'val_loss', 'none'}, "
        f"got {scheduler_step_on!r}"
    )

def _select_best_metric_value(
    *,
    metric_name: str,
    train_result: EpochResult,
    val_result: EpochResult,
) -> float:
    supported = {
        "val_accuracy": val_result.accuracy,
        "val_loss": val_result.loss,
        "val_precision_macro": val_result.precision_macro,
        "val_recall_macro": val_result.recall_macro,
        "val_f1_macro": val_result.f1_macro,
        "val_precision_weighted": val_result.precision_weighted,
        "val_recall_weighted": val_result.recall_weighted,
        "val_f1_weighted": val_result.f1_weighted,
        "train_accuracy": train_result.accuracy,
        "train_loss": train_result.loss,
        "train_precision_macro": train_result.precision_macro,
        "train_recall_macro": train_result.recall_macro,
        "train_f1_macro": train_result.f1_macro,
        "train_precision_weighted": train_result.precision_weighted,
        "train_recall_weighted": train_result.recall_weighted,
        "train_f1_weighted": train_result.f1_weighted,
    }

    if metric_name not in supported:
        raise ValueError(
            "Unsupported best_metric_name. Expected one of "
            f"{sorted(supported)}, got {metric_name!r}"
        )

    return supported[metric_name]

def _resolve_device(device: torch.device | str | None) -> torch.device:
    if device is not None:
        return torch.device(device)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _validate_epochs(epochs: int) -> None:
    if isinstance(epochs, bool) or not isinstance(epochs, int):
        raise TypeError(f"epochs must be an int, got {type(epochs).__name__}")

    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")