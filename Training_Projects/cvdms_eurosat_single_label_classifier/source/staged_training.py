"""
Project-specific staged training loop for the CVDMS EuroSAT single-label classifier.

This module implements a three-phase transfer-learning / fine-tuning workflow:

    Phase 1:
        train classifier head only

    Phase 2:
        unfreeze selected later ResNet layers plus classifier head

    Phase 3:
        unfreeze more ResNet layers plus classifier head

The staged logic is intentionally project-specific because layer names such as
"layer3", "layer4", and "fc" are ResNet-specific.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from cvdms_training_common.checkpoints import (
    BestCheckpointTracker,
    save_checkpoint,
    save_class_map,
    save_cvdms_training_metadata,
    save_json_artifact,
)
from cvdms_training_common.loops.single_label import (
    EpochResult,
    evaluate_classifier,
    train_one_epoch,
)
from cvdms_training_common.metadata import CvdmsDatasetMetadata
from cvdms_training_common.metrics.single_label import (
    confusion_matrix_to_nested_list,
    make_confusion_matrix_figure,
    make_precision_recall_figure,
    per_class_metrics_with_names,
    precision_recall_summary,
)

from early_stopping import (
    EarlyStoppingState,
    epoch_metrics_dict,
    make_early_stopping_state,
)
from model_summary import save_initial_model_summary, save_phase_model_summary
from models import (
    build_optimizer_for_phase,
    configure_trainable_layers,
    count_parameters,
    get_current_learning_rates,
)

def run_staged_training(
    *,
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    loss_fn: nn.Module,
    cvdms_metadata: CvdmsDatasetMetadata,
    model_name: str,
    phases: list[dict[str, Any]],
    output_dir: str | Path,
    tensorboard_dir: str | Path | None,
    batch_size: int,
    channels: int,
    image_size: int,
    metadata_uri: str | None = None,
    device: torch.device | str | None = None,
    best_metric_name: str = "val_f1_macro",
    best_metric_mode: str = "max",
    hyperparameters: dict[str, Any] | None = None,
    extra_checkpoint_metadata: dict[str, Any] | None = None,
    log_train_confusion_matrix: bool = False,
    log_val_confusion_matrix: bool = True,
    log_val_precision_recall_curve: bool = True,
    log_test_figures: bool = True,
    train_sampler: Any | None = None,
    is_main_process: bool = True,
    print_fn: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    """
    Run staged single-label classifier training.

    Args:
        model:
            Project model, usually ResNet18 with a replaced classifier head.
        train_loader, val_loader, test_loader:
            PyTorch DataLoaders.
        loss_fn:
            Usually nn.CrossEntropyLoss().
        cvdms_metadata:
            CVDMS dataset-version metadata.
        model_name:
            Human-readable model name.
        phases:
            Phase dictionaries from config.yaml.
        output_dir:
            Directory where checkpoints, summaries, and evaluation JSON are saved.
        tensorboard_dir:
            TensorBoard log directory. If None, TensorBoard logging is disabled.
        batch_size, channels, image_size:
            Used for torchinfo model summary input shape.
        metadata_uri:
            Optional S3 URI to CVDMS metadata.json.
        device:
            Device override. If None, uses cuda when available, else cpu.
        best_metric_name:
            Metric to select the global best checkpoint.
        best_metric_mode:
            "max" or "min".
        train_sampler:
            Optional DistributedSampler. If provided and it supports set_epoch(),
            set_epoch(global_epoch) is called each epoch.
        is_main_process:
            In distributed training, only the main process should write logs,
            checkpoints, and JSON artifacts.
    """
    _validate_phases(phases)

    resolved_device = _resolve_device(device)
    model.to(resolved_device)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    hparams = dict(hyperparameters or {})
    extra = dict(extra_checkpoint_metadata or {})
    base_model = _unwrap_model(model)

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

        save_initial_model_summary(
            model=base_model,
            output_dir=output_path,
            batch_size=batch_size,
            channels=channels,
            image_size=image_size,
        )

    writer = (
        SummaryWriter(log_dir=str(tensorboard_dir))
        if tensorboard_dir and is_main_process
        else None
    )

    best_tracker = BestCheckpointTracker(
        output_dir=output_path,
        filename="best_checkpoint.pt",
        metric_name=best_metric_name,
        mode=best_metric_mode,
    )

    history = _new_history()
    phase_summaries: list[dict[str, Any]] = []
    global_epoch = 0

    try:
        for phase_index, phase in enumerate(phases, start=1):
            phase_name = _require_nonempty_string(phase.get("name"), "phase.name")
            max_epochs = _require_positive_int(phase.get("max_epochs"), f"{phase_name}.max_epochs")
            trainable_layers = _require_string_list(
                phase.get("trainable_layers"),
                f"{phase_name}.trainable_layers",
            )
            learning_rates = _require_dict(phase.get("learning_rates"), f"{phase_name}.learning_rates")
            head_lr = _require_positive_float(learning_rates.get("head"), f"{phase_name}.learning_rates.head")
            backbone_lr = _optional_positive_float(
                learning_rates.get("backbone"),
                f"{phase_name}.learning_rates.backbone",
            )
            weight_decay = _require_nonnegative_float(
                phase.get("weight_decay", 0.0),
                f"{phase_name}.weight_decay",
            )

            configure_trainable_layers(
                base_model,
                trainable_layers=trainable_layers,
            )

            optimizer = build_optimizer_for_phase(
                base_model,
                trainable_layers=trainable_layers,
                head_lr=head_lr,
                backbone_lr=backbone_lr,
                weight_decay=weight_decay,
            )

            early_stopping = make_early_stopping_state(phase.get("early_stopping"))

            if is_main_process:
                save_phase_model_summary(
                    model=base_model,
                    output_dir=output_path,
                    phase_index=phase_index,
                    phase_name=phase_name,
                    batch_size=batch_size,
                    channels=channels,
                    image_size=image_size,
                    optimizer=optimizer,
                    extra={
                        "max_epochs": max_epochs,
                        "trainable_layers": trainable_layers,
                        "weight_decay": weight_decay,
                        "early_stopping": early_stopping.summary(),
                    },
                )

            _log_phase_start(
                writer=writer,
                phase_index=phase_index,
                phase_name=phase_name,
                global_epoch=global_epoch,
                model=base_model,
                optimizer=optimizer,
                max_epochs=max_epochs,
                trainable_layers=trainable_layers,
            )

            if is_main_process and print_fn is not None:
                print_fn(
                    " | ".join(
                        [
                            f"phase={phase_index}",
                            f"name={phase_name}",
                            f"max_epochs={max_epochs}",
                            f"trainable_layers={trainable_layers}",
                            f"lr={get_current_learning_rates(optimizer)}",
                            f"trainable_params={count_parameters(base_model).trainable_parameters:,}",
                        ]
                    )
                )

            phase_summary = _new_phase_summary(
                phase_index=phase_index,
                phase_name=phase_name,
                max_epochs=max_epochs,
                trainable_layers=trainable_layers,
                optimizer=optimizer,
                model=base_model,
                early_stopping=early_stopping,
            )

            for phase_epoch in range(1, max_epochs + 1):
                global_epoch += 1

                if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                    train_sampler.set_epoch(global_epoch)

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
                    phase_index=phase_index,
                    phase_name=phase_name,
                    global_epoch=global_epoch,
                    phase_epoch=phase_epoch,
                    train_result=train_result,
                    val_result=val_result,
                    optimizer=optimizer,
                    model=base_model,
                )

                _log_epoch_to_tensorboard(
                    writer=writer,
                    global_epoch=global_epoch,
                    phase_index=phase_index,
                    phase_epoch=phase_epoch,
                    phase_name=phase_name,
                    train_result=train_result,
                    val_result=val_result,
                    optimizer=optimizer,
                    model=base_model,
                    cvdms_metadata=cvdms_metadata,
                    log_train_confusion_matrix=log_train_confusion_matrix,
                    log_val_confusion_matrix=log_val_confusion_matrix,
                    log_val_precision_recall_curve=log_val_precision_recall_curve,
                )

                is_new_best = False
                best_metric_value = _select_metric_value(
                    metric_name=best_metric_name,
                    train_result=train_result,
                    val_result=val_result,
                )

                if is_main_process:
                    save_checkpoint(
                        path=output_path / "last_checkpoint.pt",
                        model=model,
                        epoch=global_epoch,
                        cvdms_metadata=cvdms_metadata,
                        model_name=model_name,
                        optimizer=optimizer,
                        scheduler=None,
                        metric_name=best_metric_name,
                        metric_value=best_metric_value,
                        hyperparameters=hparams,
                        extra={
                            **extra,
                            "phase_index": phase_index,
                            "phase_name": phase_name,
                            "phase_epoch": phase_epoch,
                            "global_epoch": global_epoch,
                        },
                    )

                    is_new_best = best_tracker.maybe_save(
                        metric_value=best_metric_value,
                        model=model,
                        epoch=global_epoch,
                        cvdms_metadata=cvdms_metadata,
                        model_name=model_name,
                        optimizer=optimizer,
                        scheduler=None,
                        hyperparameters=hparams,
                        extra={
                            **extra,
                            "phase_index": phase_index,
                            "phase_name": phase_name,
                            "phase_epoch": phase_epoch,
                            "global_epoch": global_epoch,
                        },
                    )

                train_metrics = _result_metrics_dict(split="train", result=train_result)
                val_metrics = _result_metrics_dict(split="val", result=val_result)
                combined_metrics = {**train_metrics, **val_metrics}
                should_stop_phase = early_stopping.update(
                    epoch=phase_epoch,
                    metrics=combined_metrics,
                )

                _update_phase_summary(
                    phase_summary=phase_summary,
                    phase_epoch=phase_epoch,
                    global_epoch=global_epoch,
                    train_result=train_result,
                    val_result=val_result,
                    best_metric_name=best_metric_name,
                    best_metric_value=best_metric_value,
                    is_new_best=is_new_best,
                    early_stopping=early_stopping,
                )

                if is_main_process and print_fn is not None:
                    marker = " *best*" if is_new_best else ""
                    early_stop_marker = " *phase_stop*" if should_stop_phase else ""
                    print_fn(
                        _format_epoch_log(
                            global_epoch=global_epoch,
                            phase_index=phase_index,
                            phase_epoch=phase_epoch,
                            max_phase_epochs=max_epochs,
                            phase_name=phase_name,
                            train_result=train_result,
                            val_result=val_result,
                            optimizer=optimizer,
                        )
                        + marker
                        + early_stop_marker
                    )

                if should_stop_phase:
                    break

            phase_summary["early_stopping"] = early_stopping.summary()
            phase_summary["completed_epochs"] = len(phase_summary["epochs"])
            phase_summaries.append(phase_summary)

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

        final_summary = {
            "history": history,
            "phases": phase_summaries,
            "best_checkpoint": best_tracker.summary(),
            "test_metrics": test_summary,
            "total_epochs": global_epoch,
        }

        if is_main_process:
            save_json_artifact(
                path=output_path / "evaluation_summary.json",
                payload=final_summary,
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
                            f"total_epochs={global_epoch}",
                        ]
                    )
                )

        return final_summary

    finally:
        if writer is not None:
            writer.flush()
            writer.close()

def _new_history() -> dict[str, list[Any]]:
    return {
        "global_epoch": [],
        "phase_index": [],
        "phase_name": [],
        "phase_epoch": [],
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
        "learning_rates": [],
        "trainable_parameters": [],
        "total_parameters": [],
    }

def _append_history(
    *,
    history: dict[str, list[Any]],
    phase_index: int,
    phase_name: str,
    global_epoch: int,
    phase_epoch: int,
    train_result: EpochResult,
    val_result: EpochResult,
    optimizer: Optimizer,
    model: nn.Module,
) -> None:
    counts = count_parameters(model)

    history["global_epoch"].append(global_epoch)
    history["phase_index"].append(phase_index)
    history["phase_name"].append(phase_name)
    history["phase_epoch"].append(phase_epoch)

    _append_result_metrics(history=history, split="train", result=train_result)
    _append_result_metrics(history=history, split="val", result=val_result)

    history["learning_rates"].append(get_current_learning_rates(optimizer))
    history["trainable_parameters"].append(counts.trainable_parameters)
    history["total_parameters"].append(counts.total_parameters)

def _append_result_metrics(
    *,
    history: dict[str, list[Any]],
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

def _new_phase_summary(
    *,
    phase_index: int,
    phase_name: str,
    max_epochs: int,
    trainable_layers: list[str],
    optimizer: Optimizer,
    model: nn.Module,
    early_stopping: EarlyStoppingState,
) -> dict[str, Any]:
    return {
        "phase_index": phase_index,
        "phase_name": phase_name,
        "max_epochs": max_epochs,
        "trainable_layers": list(trainable_layers),
        "initial_learning_rates": get_current_learning_rates(optimizer),
        "initial_parameter_counts": count_parameters(model).to_dict(),
        "early_stopping_initial": early_stopping.summary(),
        "epochs": [],
    }

def _update_phase_summary(
    *,
    phase_summary: dict[str, Any],
    phase_epoch: int,
    global_epoch: int,
    train_result: EpochResult,
    val_result: EpochResult,
    best_metric_name: str,
    best_metric_value: float,
    is_new_best: bool,
    early_stopping: EarlyStoppingState,
) -> None:
    phase_summary["epochs"].append(
        {
            "phase_epoch": phase_epoch,
            "global_epoch": global_epoch,
            "train": train_result.to_summary(),
            "val": val_result.to_summary(),
            "best_metric_name": best_metric_name,
            "best_metric_value": best_metric_value,
            "is_new_best": is_new_best,
            "early_stopping": early_stopping.summary(),
        }
    )

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

def _log_phase_start(
    *,
    writer: SummaryWriter | None,
    phase_index: int,
    phase_name: str,
    global_epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    max_epochs: int,
    trainable_layers: list[str],
) -> None:
    if writer is None:
        return

    counts = count_parameters(model)

    writer.add_scalar("phase/current_index", phase_index, global_epoch)
    writer.add_scalar("phase/max_epochs", max_epochs, global_epoch)
    writer.add_scalar("parameters/trainable", counts.trainable_parameters, global_epoch)
    writer.add_scalar("parameters/frozen", counts.frozen_parameters, global_epoch)

    for group_name, lr in get_current_learning_rates(optimizer).items():
        writer.add_scalar(f"learning_rate/{group_name}", lr, global_epoch)

    writer.add_text(
        "phase/current",
        f"Phase {phase_index}: {phase_name}\n\nTrainable layers: {trainable_layers}",
        global_epoch,
    )

def _log_epoch_to_tensorboard(
    *,
    writer: SummaryWriter | None,
    global_epoch: int,
    phase_index: int,
    phase_epoch: int,
    phase_name: str,
    train_result: EpochResult,
    val_result: EpochResult,
    optimizer: Optimizer,
    model: nn.Module,
    cvdms_metadata: CvdmsDatasetMetadata,
    log_train_confusion_matrix: bool,
    log_val_confusion_matrix: bool,
    log_val_precision_recall_curve: bool,
) -> None:
    if writer is None:
        return

    _log_scalar_metrics(writer=writer, split="train", result=train_result, step=global_epoch)
    _log_scalar_metrics(writer=writer, split="val", result=val_result, step=global_epoch)

    writer.add_scalar("phase/current_index", phase_index, global_epoch)
    writer.add_scalar("phase/epoch_within_phase", phase_epoch, global_epoch)

    counts = count_parameters(model)
    writer.add_scalar("parameters/trainable", counts.trainable_parameters, global_epoch)
    writer.add_scalar("parameters/frozen", counts.frozen_parameters, global_epoch)

    for group_name, lr in get_current_learning_rates(optimizer).items():
        writer.add_scalar(f"learning_rate/{group_name}", lr, global_epoch)

    if log_train_confusion_matrix:
        _log_confusion_matrix_figure(
            writer=writer,
            tag="confusion_matrix/train",
            confusion_matrix=train_result.confusion_matrix,
            cvdms_metadata=cvdms_metadata,
            step=global_epoch,
            title_prefix=f"Train Confusion Matrix - {phase_name} - Epoch {global_epoch}",
        )

    if log_val_confusion_matrix:
        _log_confusion_matrix_figure(
            writer=writer,
            tag="confusion_matrix/val",
            confusion_matrix=val_result.confusion_matrix,
            cvdms_metadata=cvdms_metadata,
            step=global_epoch,
            title_prefix=f"Validation Confusion Matrix - {phase_name} - Epoch {global_epoch}",
        )

    if log_val_precision_recall_curve and val_result.probabilities is not None:
        _log_precision_recall_figure(
            writer=writer,
            tag="precision_recall/val",
            result=val_result,
            cvdms_metadata=cvdms_metadata,
            step=global_epoch,
            title_prefix=f"Validation One-vs-Rest PR Curves - {phase_name} - Epoch {global_epoch}",
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
    *,
    global_epoch: int,
    phase_index: int,
    phase_epoch: int,
    max_phase_epochs: int,
    phase_name: str,
    train_result: EpochResult,
    val_result: EpochResult,
    optimizer: Optimizer,
) -> str:
    return " | ".join(
        [
            f"global_epoch={global_epoch}",
            f"phase={phase_index}:{phase_name}",
            f"phase_epoch={phase_epoch}/{max_phase_epochs}",
            f"train_loss={train_result.loss:.4f}",
            f"train_acc={train_result.accuracy:.4f}",
            f"train_f1={train_result.f1_macro:.4f}",
            f"val_loss={val_result.loss:.4f}",
            f"val_acc={val_result.accuracy:.4f}",
            f"val_f1={val_result.f1_macro:.4f}",
            f"lr={get_current_learning_rates(optimizer)}",
            f"time={train_result.elapsed_seconds + val_result.elapsed_seconds:.1f}s",
        ]
    )

def _result_metrics_dict(
    *,
    split: str,
    result: EpochResult,
) -> dict[str, float]:
    return epoch_metrics_dict(
        split=split,
        loss=result.loss,
        accuracy=result.accuracy,
        precision_macro=result.precision_macro,
        recall_macro=result.recall_macro,
        f1_macro=result.f1_macro,
        precision_weighted=result.precision_weighted,
        recall_weighted=result.recall_weighted,
        f1_weighted=result.f1_weighted,
    )

def _select_metric_value(
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
            "Unsupported metric name. Expected one of "
            f"{sorted(supported)}, got {metric_name!r}"
        )

    return float(supported[metric_name])

def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model

def _resolve_device(device: torch.device | str | None) -> torch.device:
    if device is not None:
        return torch.device(device)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _validate_phases(phases: list[dict[str, Any]]) -> None:
    if not isinstance(phases, list):
        raise TypeError(f"phases must be a list, got {type(phases).__name__}")

    if not phases:
        raise ValueError("phases cannot be empty")

def _require_nonempty_string(value: Any, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} cannot be null")

    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} cannot be empty")

    return text

def _require_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int, got {value!r}")

    if value < 1:
        raise ValueError(f"{field_name} must be >= 1, got {value}")

    return value

def _require_positive_float(value: Any, field_name: str) -> float:
    if value is None:
        raise ValueError(f"{field_name} cannot be null")

    number = float(value)
    if number <= 0:
        raise ValueError(f"{field_name} must be > 0, got {number}")

    return number

def _optional_positive_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None

    number = float(value)
    if number <= 0:
        raise ValueError(f"{field_name} must be > 0 when provided, got {number}")

    return number

def _require_nonnegative_float(value: Any, field_name: str) -> float:
    number = float(value)
    if number < 0:
        raise ValueError(f"{field_name} must be >= 0, got {number}")

    return number

def _require_dict(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a dict, got {type(value).__name__}")

    return value

def _require_string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list[str], got {type(value).__name__}")

    out: list[str] = []

    for idx, item in enumerate(value):
        text = str(item).strip()
        if not text:
            raise ValueError(f"{field_name}[{idx}] cannot be empty")
        out.append(text)

    if not out:
        raise ValueError(f"{field_name} cannot be empty")

    return out