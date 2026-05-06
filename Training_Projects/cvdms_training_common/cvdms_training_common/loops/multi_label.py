"""
Reusable multi-label classification training loops for CVDMS/PyTorch projects.

These helpers are intended for standard multi-label classifiers:

    images -> model(images) -> logits [batch_size, num_classes]
    targets -> multi-hot float targets [batch_size, num_classes]

They intentionally do not define model architectures. Individual projects should
build their own model, loss function, optimizer, scheduler, and transforms.

For multi-label classification:
    - Use raw logits with BCEWithLogitsLoss.
    - Do not apply sigmoid before the loss.
    - Metrics apply sigmoid(logits), then threshold probabilities.
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
from cvdms_training_common.metrics.multi_label import (
    confusion_counts_to_nested_list,
    exact_match_count_from_logits,
    logits_to_probabilities_and_targets,
    make_precision_recall_figure,
    new_confusion_counts,
    per_class_metrics_with_names,
    precision_recall_summary,
    summary_metrics_from_confusion_counts,
    threshold_sweep_summary,
    update_confusion_counts_from_logits,
)

@dataclass(frozen=True)
class EpochResult:
    """
    Metrics from one train/eval epoch.
    """

    loss: float
    threshold: float
    hamming_accuracy: float
    hamming_loss: float
    subset_accuracy: float | None
    precision_micro: float
    recall_micro: float
    f1_micro: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    total_examples: int
    exact_match_count: int
    elapsed_seconds: float
    confusion_counts: torch.Tensor
    probabilities: torch.Tensor | None = None
    targets: torch.Tensor | None = None
    macro_average_precision: float | None = None
    micro_average_precision: float | None = None

    @property
    def accuracy(self) -> float:
        """
        Backward-friendly alias.

        For multi-label classification this means hamming accuracy, not
        single-label argmax accuracy.
        """
        return self.hamming_accuracy

    def to_summary(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "threshold": self.threshold,
            "hamming_accuracy": self.hamming_accuracy,
            "hamming_loss": self.hamming_loss,
            "subset_accuracy": self.subset_accuracy,
            "precision_micro": self.precision_micro,
            "recall_micro": self.recall_micro,
            "f1_micro": self.f1_micro,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "precision_weighted": self.precision_weighted,
            "recall_weighted": self.recall_weighted,
            "f1_weighted": self.f1_weighted,
            "macro_average_precision": self.macro_average_precision,
            "micro_average_precision": self.micro_average_precision,
            "total_examples": self.total_examples,
            "exact_match_count": self.exact_match_count,
            "elapsed_seconds": self.elapsed_seconds,
            "confusion_counts": confusion_counts_to_nested_list(self.confusion_counts),
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
    threshold: float = 0.5,
) -> EpochResult:
    """
    Train a multi-label classifier for one epoch.

    Uses:
        - model.train()
        - optimizer.zero_grad(set_to_none=True)
        - raw logits passed to BCEWithLogitsLoss-style loss function
        - sigmoid + threshold for metrics only
        - sample-weighted average loss
        - per-class TP/FP/TN/FN-derived precision/recall/F1
    """
    model.train()
    start_time = time.perf_counter()

    total_loss = 0.0
    total_examples = 0
    exact_match_count = 0
    confusion_counts = new_confusion_counts(num_classes=num_classes, device="cpu")

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = loss_fn(logits, targets)

        loss.backward()
        optimizer.step()

        batch_size = int(targets.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

        logits_cpu = logits.detach().cpu()
        targets_cpu = targets.detach().cpu()

        exact_match_count += exact_match_count_from_logits(
            logits_cpu,
            targets_cpu,
            threshold=threshold,
        )

        update_confusion_counts_from_logits(
            confusion_counts,
            logits_cpu,
            targets_cpu,
            threshold=threshold,
        )

    if total_examples == 0:
        raise ValueError("Training dataloader produced zero examples")

    elapsed = time.perf_counter() - start_time
    summary = summary_metrics_from_confusion_counts(
        confusion_counts,
        threshold=threshold,
        total_examples=total_examples,
        exact_match_count=exact_match_count,
    )

    return EpochResult(
        loss=total_loss / total_examples,
        threshold=threshold,
        hamming_accuracy=summary.hamming_accuracy,
        hamming_loss=summary.hamming_loss,
        subset_accuracy=summary.subset_accuracy,
        precision_micro=summary.precision_micro,
        recall_micro=summary.recall_micro,
        f1_micro=summary.f1_micro,
        precision_macro=summary.precision_macro,
        recall_macro=summary.recall_macro,
        f1_macro=summary.f1_macro,
        precision_weighted=summary.precision_weighted,
        recall_weighted=summary.recall_weighted,
        f1_weighted=summary.f1_weighted,
        total_examples=total_examples,
        exact_match_count=exact_match_count,
        elapsed_seconds=elapsed,
        confusion_counts=confusion_counts,
    )

@torch.inference_mode()
def evaluate_classifier(
    *,
    model: nn.Module,
    dataloader,
    loss_fn: nn.Module,
    device: torch.device,
    num_classes: int,
    threshold: float = 0.5,
    collect_pr_data: bool = False,
    idx_to_class: dict[int, str] | None = None,
) -> EpochResult:
    """
    Evaluate a multi-label classifier on validation or test data.

    Uses:
        - model.eval()
        - torch.inference_mode()
        - sample-weighted average loss
        - sigmoid + threshold for threshold-dependent metrics
        - optional sigmoid probabilities and targets for AP/mAP and PR curves

    If collect_pr_data=True, idx_to_class is required so AP/mAP summaries can be
    attached to the returned EpochResult.
    """
    if collect_pr_data and idx_to_class is None:
        raise ValueError("idx_to_class is required when collect_pr_data=True")

    model.eval()
    start_time = time.perf_counter()

    total_loss = 0.0
    total_examples = 0
    exact_match_count = 0
    confusion_counts = new_confusion_counts(num_classes=num_classes, device="cpu")
    logits_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).float()

        logits = model(images)
        loss = loss_fn(logits, targets)

        batch_size = int(targets.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

        logits_cpu = logits.detach().cpu()
        targets_cpu = targets.detach().cpu()

        exact_match_count += exact_match_count_from_logits(
            logits_cpu,
            targets_cpu,
            threshold=threshold,
        )

        update_confusion_counts_from_logits(
            confusion_counts,
            logits_cpu,
            targets_cpu,
            threshold=threshold,
        )

        if collect_pr_data:
            logits_batches.append(logits_cpu)
            target_batches.append(targets_cpu)

    if total_examples == 0:
        raise ValueError("Evaluation dataloader produced zero examples")

    probabilities = None
    collected_targets = None
    macro_average_precision = None
    micro_average_precision = None

    if collect_pr_data:
        probabilities, collected_targets = logits_to_probabilities_and_targets(
            logits_batches=logits_batches,
            target_batches=target_batches,
        )
        pr_summary = precision_recall_summary(
            probabilities=probabilities,
            targets=collected_targets,
            idx_to_class=idx_to_class or {},
        )
        macro_average_precision = pr_summary.macro_average_precision
        micro_average_precision = pr_summary.micro_average_precision

    elapsed = time.perf_counter() - start_time
    summary = summary_metrics_from_confusion_counts(
        confusion_counts,
        threshold=threshold,
        total_examples=total_examples,
        exact_match_count=exact_match_count,
    )

    return EpochResult(
        loss=total_loss / total_examples,
        threshold=threshold,
        hamming_accuracy=summary.hamming_accuracy,
        hamming_loss=summary.hamming_loss,
        subset_accuracy=summary.subset_accuracy,
        precision_micro=summary.precision_micro,
        recall_micro=summary.recall_micro,
        f1_micro=summary.f1_micro,
        precision_macro=summary.precision_macro,
        recall_macro=summary.recall_macro,
        f1_macro=summary.f1_macro,
        precision_weighted=summary.precision_weighted,
        recall_weighted=summary.recall_weighted,
        f1_weighted=summary.f1_weighted,
        total_examples=total_examples,
        exact_match_count=exact_match_count,
        elapsed_seconds=elapsed,
        confusion_counts=confusion_counts,
        probabilities=probabilities,
        targets=collected_targets,
        macro_average_precision=macro_average_precision,
        micro_average_precision=micro_average_precision,
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
    threshold: float = 0.5,
    device: torch.device | str | None = None,
    metadata_uri: str | None = None,
    scheduler: LRScheduler | None = None,
    scheduler_step_on: str = "epoch",
    tensorboard_dir: str | Path | None = None,
    hyperparameters: dict[str, Any] | None = None,
    extra_checkpoint_metadata: dict[str, Any] | None = None,
    best_metric_name: str = "val_f1_macro",
    best_metric_mode: str = "max",
    log_val_precision_recall_curve: bool = True,
    log_test_figures: bool = True,
    log_threshold_sweep: bool = True,
    print_fn: Callable[[str], None] | None = print,
    train_sampler: Any | None = None,
    is_main_process: bool = True,
) -> FitResult:
    """
    Fit a standard multi-label classifier using train/val/test loaders.

    This function:
        - trains for N epochs
        - evaluates on validation each epoch
        - logs TensorBoard scalar timelines for loss, hamming accuracy, precision, recall, F1, AP/mAP, and LR
        - optionally logs validation multi-label PR curve images per epoch
        - saves last_checkpoint.pt every epoch
        - saves best_checkpoint.pt using the selected validation metric
        - evaluates on test once at the end
        - saves evaluation_summary.json
        - saves CVDMS training metadata and class map artifacts

    Args:
        threshold:
            Global sigmoid probability threshold used for threshold-dependent
            metrics such as precision, recall, F1, hamming accuracy, and subset
            accuracy.
        scheduler_step_on:
            "epoch" means scheduler.step() after each epoch.
            "val_loss" means scheduler.step(val_loss), useful for ReduceLROnPlateau.
            "none" disables scheduler stepping even if scheduler is provided.

    This version is “DDP-safe” for logging/checkpointing, but it does not yet
    all-reduce metrics across ranks. So in a true multi-process DDP run, each
    rank computes metrics on its local shard. Since only rank 0 logs/saves, the
    saved metrics would reflect rank 0’s shard unless distributed metric
    aggregation is added later.
    """
    _validate_epochs(epochs)

    resolved_device = _resolve_device(device)
    model.to(resolved_device)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    hparams = dict(hyperparameters or {})
    hparams.setdefault("threshold", threshold)
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

    val_needs_pr_data = (
        log_val_precision_recall_curve
        or _best_metric_requires_pr_data(best_metric_name)
        or log_threshold_sweep
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
                threshold=threshold,
            )

            val_result = evaluate_classifier(
                model=model,
                dataloader=val_loader,
                loss_fn=loss_fn,
                device=resolved_device,
                num_classes=cvdms_metadata.num_classes,
                threshold=threshold,
                collect_pr_data=val_needs_pr_data,
                idx_to_class=cvdms_metadata.idx_to_class,
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
                log_val_precision_recall_curve=log_val_precision_recall_curve,
                log_threshold_sweep=log_threshold_sweep,
            )

            if scheduler is not None:
                _step_scheduler(
                    scheduler=scheduler,
                    scheduler_step_on=scheduler_step_on,
                    val_loss=val_result.loss,
                )

            if is_main_process:
                best_metric_value = _select_best_metric_value(
                    metric_name=best_metric_name,
                    train_result=train_result,
                    val_result=val_result,
                )

                save_checkpoint(
                    path=output_path / "last_checkpoint.pt",
                    model=model,
                    epoch=epoch,
                    cvdms_metadata=cvdms_metadata,
                    model_name=model_name,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metric_name=best_metric_name,
                    metric_value=best_metric_value,
                    hyperparameters=hparams,
                    extra=extra,
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
            threshold=threshold,
            collect_pr_data=True,
            idx_to_class=cvdms_metadata.idx_to_class,
        )

        test_summary = _build_result_summary(
            result=test_result,
            cvdms_metadata=cvdms_metadata,
            include_precision_recall=True,
            include_threshold_sweep=log_threshold_sweep,
        )

        if writer is not None:
            _log_test_to_tensorboard(
                writer=writer,
                test_result=test_result,
                cvdms_metadata=cvdms_metadata,
                log_test_figures=log_test_figures,
                log_threshold_sweep=log_threshold_sweep,
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
                            f"test_hamming_acc={test_result.hamming_accuracy:.4f}",
                            f"test_subset_acc={_fmt_optional(test_result.subset_accuracy)}",
                            f"test_precision_micro={test_result.precision_micro:.4f}",
                            f"test_recall_micro={test_result.recall_micro:.4f}",
                            f"test_f1_micro={test_result.f1_micro:.4f}",
                            f"test_f1_macro={test_result.f1_macro:.4f}",
                            f"test_mAP={_fmt_optional(test_result.macro_average_precision)}",
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
        "train_hamming_accuracy": [],
        "train_hamming_loss": [],
        "train_subset_accuracy": [],
        "train_precision_micro": [],
        "train_recall_micro": [],
        "train_f1_micro": [],
        "train_precision_macro": [],
        "train_recall_macro": [],
        "train_f1_macro": [],
        "train_precision_weighted": [],
        "train_recall_weighted": [],
        "train_f1_weighted": [],
        "val_loss": [],
        "val_hamming_accuracy": [],
        "val_hamming_loss": [],
        "val_subset_accuracy": [],
        "val_precision_micro": [],
        "val_recall_micro": [],
        "val_f1_micro": [],
        "val_precision_macro": [],
        "val_recall_macro": [],
        "val_f1_macro": [],
        "val_precision_weighted": [],
        "val_recall_weighted": [],
        "val_f1_weighted": [],
        "val_macro_average_precision": [],
        "val_micro_average_precision": [],
    }

def _append_history(
    *,
    history: dict[str, list[float]],
    split: str,
    result: EpochResult,
) -> None:
    history[f"{split}_loss"].append(result.loss)
    history[f"{split}_hamming_accuracy"].append(result.hamming_accuracy)
    history[f"{split}_hamming_loss"].append(result.hamming_loss)

    if result.subset_accuracy is not None:
        history[f"{split}_subset_accuracy"].append(result.subset_accuracy)

    history[f"{split}_precision_micro"].append(result.precision_micro)
    history[f"{split}_recall_micro"].append(result.recall_micro)
    history[f"{split}_f1_micro"].append(result.f1_micro)
    history[f"{split}_precision_macro"].append(result.precision_macro)
    history[f"{split}_recall_macro"].append(result.recall_macro)
    history[f"{split}_f1_macro"].append(result.f1_macro)
    history[f"{split}_precision_weighted"].append(result.precision_weighted)
    history[f"{split}_recall_weighted"].append(result.recall_weighted)
    history[f"{split}_f1_weighted"].append(result.f1_weighted)

    if result.macro_average_precision is not None:
        history[f"{split}_macro_average_precision"].append(result.macro_average_precision)

    if result.micro_average_precision is not None:
        history[f"{split}_micro_average_precision"].append(result.micro_average_precision)

def _build_result_summary(
    *,
    result: EpochResult,
    cvdms_metadata: CvdmsDatasetMetadata,
    include_precision_recall: bool,
    include_threshold_sweep: bool,
) -> dict[str, Any]:
    summary = {
        "loss": result.loss,
        "threshold": result.threshold,
        "hamming_accuracy": result.hamming_accuracy,
        "hamming_loss": result.hamming_loss,
        "subset_accuracy": result.subset_accuracy,
        "precision_micro": result.precision_micro,
        "recall_micro": result.recall_micro,
        "f1_micro": result.f1_micro,
        "precision_macro": result.precision_macro,
        "recall_macro": result.recall_macro,
        "f1_macro": result.f1_macro,
        "precision_weighted": result.precision_weighted,
        "recall_weighted": result.recall_weighted,
        "f1_weighted": result.f1_weighted,
        "macro_average_precision": result.macro_average_precision,
        "micro_average_precision": result.micro_average_precision,
        "total_examples": result.total_examples,
        "exact_match_count": result.exact_match_count,
        "elapsed_seconds": result.elapsed_seconds,
        "confusion_counts": confusion_counts_to_nested_list(result.confusion_counts),
        "per_class_metrics": per_class_metrics_with_names(
            result.confusion_counts,
            idx_to_class=cvdms_metadata.idx_to_class,
        ),
    }

    if include_precision_recall and result.probabilities is not None and result.targets is not None:
        summary["precision_recall"] = precision_recall_summary(
            probabilities=result.probabilities,
            targets=result.targets,
            idx_to_class=cvdms_metadata.idx_to_class,
        ).to_dict()

    if include_threshold_sweep and result.probabilities is not None and result.targets is not None:
        summary["threshold_sweep"] = threshold_sweep_summary(
            probabilities=result.probabilities,
            targets=result.targets,
        )

    return summary

def _log_epoch_to_tensorboard(
    *,
    writer: SummaryWriter | None,
    epoch: int,
    train_result: EpochResult,
    val_result: EpochResult,
    optimizer: Optimizer,
    cvdms_metadata: CvdmsDatasetMetadata,
    log_val_precision_recall_curve: bool,
    log_threshold_sweep: bool,
) -> None:
    if writer is None:
        return

    _log_scalar_metrics(writer=writer, split="train", result=train_result, step=epoch)
    _log_scalar_metrics(writer=writer, split="val", result=val_result, step=epoch)

    lr = _get_current_learning_rate(optimizer)
    if lr is not None:
        writer.add_scalar("learning_rate", lr, epoch)

    if log_val_precision_recall_curve and val_result.probabilities is not None:
        _log_precision_recall_figure(
            writer=writer,
            tag="precision_recall/val",
            result=val_result,
            cvdms_metadata=cvdms_metadata,
            step=epoch,
            title_prefix=f"Validation Multi-Label PR Curves - Epoch {epoch}",
        )

    if log_threshold_sweep and val_result.probabilities is not None and val_result.targets is not None:
        _log_threshold_sweep_to_tensorboard(
            writer=writer,
            split="val",
            result=val_result,
            step=epoch,
        )

def _log_test_to_tensorboard(
    *,
    writer: SummaryWriter,
    test_result: EpochResult,
    cvdms_metadata: CvdmsDatasetMetadata,
    log_test_figures: bool,
    log_threshold_sweep: bool,
) -> None:
    _log_scalar_metrics(writer=writer, split="test", result=test_result, step=0)

    if log_threshold_sweep and test_result.probabilities is not None and test_result.targets is not None:
        _log_threshold_sweep_to_tensorboard(
            writer=writer,
            split="test",
            result=test_result,
            step=0,
        )

    if not log_test_figures:
        return

    if test_result.probabilities is not None:
        _log_precision_recall_figure(
            writer=writer,
            tag="precision_recall/test",
            result=test_result,
            cvdms_metadata=cvdms_metadata,
            step=0,
            title_prefix="Test Multi-Label PR Curves",
        )

def _log_scalar_metrics(
    *,
    writer: SummaryWriter,
    split: str,
    result: EpochResult,
    step: int,
) -> None:
    writer.add_scalar(f"loss/{split}", result.loss, step)
    writer.add_scalar(f"hamming_accuracy/{split}", result.hamming_accuracy, step)
    writer.add_scalar(f"hamming_loss/{split}", result.hamming_loss, step)

    if result.subset_accuracy is not None:
        writer.add_scalar(f"subset_accuracy/{split}", result.subset_accuracy, step)

    writer.add_scalar(f"precision_micro/{split}", result.precision_micro, step)
    writer.add_scalar(f"recall_micro/{split}", result.recall_micro, step)
    writer.add_scalar(f"f1_micro/{split}", result.f1_micro, step)
    writer.add_scalar(f"precision_macro/{split}", result.precision_macro, step)
    writer.add_scalar(f"recall_macro/{split}", result.recall_macro, step)
    writer.add_scalar(f"f1_macro/{split}", result.f1_macro, step)
    writer.add_scalar(f"precision_weighted/{split}", result.precision_weighted, step)
    writer.add_scalar(f"recall_weighted/{split}", result.recall_weighted, step)
    writer.add_scalar(f"f1_weighted/{split}", result.f1_weighted, step)

    if result.macro_average_precision is not None:
        writer.add_scalar(
            f"average_precision_macro/{split}",
            result.macro_average_precision,
            step,
        )

    if result.micro_average_precision is not None:
        writer.add_scalar(
            f"average_precision_micro/{split}",
            result.micro_average_precision,
            step,
        )

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

def _log_threshold_sweep_to_tensorboard(
    *,
    writer: SummaryWriter,
    split: str,
    result: EpochResult,
    step: int,
) -> None:
    if result.probabilities is None or result.targets is None:
        return

    sweep = threshold_sweep_summary(
        probabilities=result.probabilities,
        targets=result.targets,
    )

    for item in sweep:
        threshold = item["threshold"]
        if not isinstance(threshold, float):
            continue

        writer.add_scalar(
            f"threshold_sweep/{split}/f1_macro",
            item["f1_macro"],
            int(round(threshold * 1000)) + step * 10000,
        )
        writer.add_scalar(
            f"threshold_sweep/{split}/f1_micro",
            item["f1_micro"],
            int(round(threshold * 1000)) + step * 10000,
        )

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
            f"train_hamming_acc={train_result.hamming_accuracy:.4f}",
            f"train_f1_macro={train_result.f1_macro:.4f}",
            f"val_loss={val_result.loss:.4f}",
            f"val_hamming_acc={val_result.hamming_accuracy:.4f}",
            f"val_f1_macro={val_result.f1_macro:.4f}",
            f"val_mAP={_fmt_optional(val_result.macro_average_precision)}",
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
    supported: dict[str, float | None] = {
        "val_hamming_accuracy": val_result.hamming_accuracy,
        "val_hamming_loss": val_result.hamming_loss,
        "val_subset_accuracy": val_result.subset_accuracy,
        "val_loss": val_result.loss,
        "val_precision_micro": val_result.precision_micro,
        "val_recall_micro": val_result.recall_micro,
        "val_f1_micro": val_result.f1_micro,
        "val_precision_macro": val_result.precision_macro,
        "val_recall_macro": val_result.recall_macro,
        "val_f1_macro": val_result.f1_macro,
        "val_precision_weighted": val_result.precision_weighted,
        "val_recall_weighted": val_result.recall_weighted,
        "val_f1_weighted": val_result.f1_weighted,
        "val_macro_average_precision": val_result.macro_average_precision,
        "val_micro_average_precision": val_result.micro_average_precision,
        "train_hamming_accuracy": train_result.hamming_accuracy,
        "train_hamming_loss": train_result.hamming_loss,
        "train_subset_accuracy": train_result.subset_accuracy,
        "train_loss": train_result.loss,
        "train_precision_micro": train_result.precision_micro,
        "train_recall_micro": train_result.recall_micro,
        "train_f1_micro": train_result.f1_micro,
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

    value = supported[metric_name]
    if value is None:
        raise ValueError(
            f"Metric {metric_name!r} is None. If this is an AP/mAP metric, "
            "make sure validation PR data is being collected."
        )

    return value

def _best_metric_requires_pr_data(metric_name: str) -> bool:
    return metric_name in {
        "val_macro_average_precision",
        "val_micro_average_precision",
    }

def _resolve_device(device: torch.device | str | None) -> torch.device:
    if device is not None:
        return torch.device(device)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _validate_epochs(epochs: int) -> None:
    if isinstance(epochs, bool) or not isinstance(epochs, int):
        raise TypeError(f"epochs must be an int, got {type(epochs).__name__}")

    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")

def _fmt_optional(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"