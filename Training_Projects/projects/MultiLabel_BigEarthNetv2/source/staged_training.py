"""
Project-specific staged training loop for the CVDMS BigEarthNet v2 multi-label classifier.

This module implements a staged ResNet transfer-learning workflow for Project 2.
The project is multi-label, not single-label multiclass:

    model output: raw logits shaped [batch_size, num_classes]
    targets:      multi-hot float vectors shaped [batch_size, num_classes]
    loss:         BCEWithLogitsLoss-style loss over raw logits
    metrics:      sigmoid(logits), then optional thresholding

Important metric note
---------------------
In this project, mAP means macro-averaged Average Precision. Concretely,
`macro_average_precision` is computed by calculating AP separately for each
class and then averaging those per-class AP values across supported classes.
User-facing docs and README text should call this value "mAP" or
"macro average precision (mAP)". `micro_average_precision` should be called
"micro-AP" because it flattens all class decisions before computing AP.

The code below keeps the original staged-training behavior, but adds two major
Project 2 fixes:

1. The primary test result is now computed from `best_checkpoint.pt`, not from
   the final model state. The final model is still evaluated and saved for
   comparison.
2. Richer multi-label diagnostics are saved to disk: per-class CSV/JSON tables,
   PR curves, per-class binary confusion grids, probability heatmaps, and
   thresholded association/error heatmaps.
"""

from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
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
from cvdms_training_common.loops.multi_label import (
    EpochResult,
    evaluate_classifier,
    train_one_epoch,
)
from cvdms_training_common.metadata import CvdmsDatasetMetadata
from cvdms_training_common.metrics.multi_label import (
    conditional_prediction_probability_matrix,
    confusion_counts_to_nested_list,
    false_association_probability_matrix,
    make_binary_confusion_grid_figure,
    make_conditional_prediction_probability_heatmap_figure,
    make_false_association_probability_heatmap_figure,
    make_missed_vs_extra_heatmap_figure,
    make_matrix_heatmap_figure,
    make_per_class_metric_bar_figure,
    make_per_class_precision_recall_figures,
    make_precision_recall_figure,
    make_precision_recall_small_multiples_figure,
    make_thresholded_cooccurrence_heatmap_figure,
    matrix_to_nested_list,
    new_confusion_counts,
    missed_vs_extra_label_matrix,
    per_class_metrics_table,
    per_class_metrics_with_names,
    precision_recall_summary,
    summary_metrics_from_confusion_counts,
    save_figures,
    threshold_sweep_summary,
    update_confusion_counts_from_predictions,
    thresholded_true_predicted_cooccurrence_matrix,
    write_json,
    write_metric_rows_csv,
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
    threshold: float = 0.5,
    best_metric_name: str = "val_f1_macro",
    best_metric_mode: str = "max",
    hyperparameters: dict[str, Any] | None = None,
    extra_checkpoint_metadata: dict[str, Any] | None = None,
    log_val_precision_recall_curve: bool = True,
    log_test_figures: bool = True,
    log_threshold_sweep: bool = True,
    threshold_sweep_values: list[float] | None = None,
    save_best_validation_diagnostics: bool = True,
    save_test_diagnostics: bool = True,
    evaluate_per_class_thresholds: bool = True,
    train_sampler: Any | None = None,
    is_main_process: bool = True,
    print_fn: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    """
    Run staged multi-label classifier training.

    The training loop is intentionally project-specific because the phase logic
    refers to ResNet layer names such as `layer3`, `layer4`, and `fc`.

    Best-checkpoint policy:
        During training, the global best checkpoint is selected by
        `best_metric_name` and `best_metric_mode`. After all phases complete,
        this function evaluates the final model state and then reloads
        `best_checkpoint.pt` for the primary test evaluation. This avoids the
        misleading situation where the printed test result comes from a later,
        worse-overfit final epoch instead of the validation-selected checkpoint.

    Metric interpretation:
        Thresholded metrics such as hamming accuracy, subset accuracy,
        precision, recall, and F1 use `threshold` to convert sigmoid
        probabilities into binary predictions. AP/mAP metrics are threshold-free
        ranking diagnostics. In this project, mAP is `macro_average_precision`:
        the mean of per-class AP values.
    """
    _validate_phases(phases)
    _validate_threshold(threshold, "threshold")

    threshold_values = _normalize_threshold_sweep_values(threshold_sweep_values)
    val_needs_pr_data = (
        log_val_precision_recall_curve
        or save_best_validation_diagnostics
        or evaluate_per_class_thresholds
        or (log_threshold_sweep and bool(threshold_values))
        or _best_metric_requires_pr_data(best_metric_name)
    )

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
                extra={**extra, "metric_notes": _metric_notes()},
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

        save_json_artifact(
            path=output_path / "metric_notes.json",
            payload=_metric_notes(),
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

            configure_trainable_layers(base_model, trainable_layers=trainable_layers)
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
                        "threshold": threshold,
                        "early_stopping": early_stopping.summary(),
                        "metric_notes": _metric_notes(),
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
                            f"threshold={threshold}",
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
                threshold=threshold,
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
                    log_val_precision_recall_curve=log_val_precision_recall_curve,
                    log_threshold_sweep=log_threshold_sweep,
                    threshold_sweep_values=threshold_values,
                )

                best_metric_value = _select_metric_value(
                    metric_name=best_metric_name,
                    train_result=train_result,
                    val_result=val_result,
                )
                is_new_best = False

                if is_main_process:
                    checkpoint_extra = {
                        **extra,
                        "phase_index": phase_index,
                        "phase_name": phase_name,
                        "phase_epoch": phase_epoch,
                        "global_epoch": global_epoch,
                        "threshold": threshold,
                        "metric_notes": _metric_notes(),
                    }

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
                        extra=checkpoint_extra,
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
                        extra=checkpoint_extra,
                    )

                    if is_new_best and save_best_validation_diagnostics:
                        _save_result_artifacts(
                            result=val_result,
                            cvdms_metadata=cvdms_metadata,
                            output_dir=output_path / "diagnostics" / "best_validation",
                            split_name="best_validation",
                            threshold_sweep_values=threshold_values if log_threshold_sweep else [],
                            include_figures=log_val_precision_recall_curve,
                            title_prefix=(
                                f"Best Validation Diagnostics - {phase_name} - "
                                f"Epoch {global_epoch}"
                            ),
                        )
                        save_json_artifact(
                            path=output_path / "best_validation_summary.json",
                            payload=_build_result_summary(
                                result=val_result,
                                cvdms_metadata=cvdms_metadata,
                                include_precision_recall=True,
                                threshold_sweep_values=threshold_values if log_threshold_sweep else [],
                            ),
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

        final_test_result = evaluate_classifier(
            model=model,
            dataloader=test_loader,
            loss_fn=loss_fn,
            device=resolved_device,
            num_classes=cvdms_metadata.num_classes,
            threshold=threshold,
            collect_pr_data=True,
            idx_to_class=cvdms_metadata.idx_to_class,
        )

        final_test_summary = _build_result_summary(
            result=final_test_result,
            cvdms_metadata=cvdms_metadata,
            include_precision_recall=True,
            threshold_sweep_values=threshold_values if log_threshold_sweep else [],
        )

        best_checkpoint_path = output_path / "best_checkpoint.pt"
        best_checkpoint_load = _load_checkpoint_model_state(
            checkpoint_path=best_checkpoint_path,
            model=model,
            device=resolved_device,
        )

        # Re-evaluate the validation set with the actual best checkpoint. This is
        # used only to derive Project-2-specific per-class thresholds. The test
        # set is never used to choose thresholds.
        best_val_result = evaluate_classifier(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=resolved_device,
            num_classes=cvdms_metadata.num_classes,
            threshold=threshold,
            collect_pr_data=True,
            idx_to_class=cvdms_metadata.idx_to_class,
        )

        best_val_summary = _build_result_summary(
            result=best_val_result,
            cvdms_metadata=cvdms_metadata,
            include_precision_recall=True,
            threshold_sweep_values=threshold_values if log_threshold_sweep else [],
        )

        best_test_result = evaluate_classifier(
            model=model,
            dataloader=test_loader,
            loss_fn=loss_fn,
            device=resolved_device,
            num_classes=cvdms_metadata.num_classes,
            threshold=threshold,
            collect_pr_data=True,
            idx_to_class=cvdms_metadata.idx_to_class,
        )

        best_test_summary = _build_result_summary(
            result=best_test_result,
            cvdms_metadata=cvdms_metadata,
            include_precision_recall=True,
            threshold_sweep_values=threshold_values if log_threshold_sweep else [],
        )

        per_class_threshold_strategy: dict[str, Any] | None = None
        best_val_per_class_threshold_summary: dict[str, Any] | None = None
        best_test_per_class_threshold_summary: dict[str, Any] | None = None

        if evaluate_per_class_thresholds:
            per_class_threshold_strategy = _derive_per_class_threshold_strategy(
                validation_result=best_val_result,
                cvdms_metadata=cvdms_metadata,
                fallback_threshold=threshold,
            )
            best_val_per_class_threshold_summary = _build_per_class_threshold_summary(
                result=best_val_result,
                cvdms_metadata=cvdms_metadata,
                threshold_strategy=per_class_threshold_strategy,
            )
            best_test_per_class_threshold_summary = _build_per_class_threshold_summary(
                result=best_test_result,
                cvdms_metadata=cvdms_metadata,
                threshold_strategy=per_class_threshold_strategy,
            )

        diagnostic_artifacts: dict[str, Any] = {}
        if is_main_process and save_test_diagnostics:
            diagnostic_artifacts["test_final_model_global_threshold"] = _save_result_artifacts(
                result=final_test_result,
                cvdms_metadata=cvdms_metadata,
                output_dir=output_path / "diagnostics" / "test_final_model" / _global_threshold_dir_name(threshold),
                split_name="test_final_model_global_threshold",
                threshold_sweep_values=threshold_values if log_threshold_sweep else [],
                include_figures=log_test_figures,
                title_prefix=f"Test Diagnostics - Final Model State - Global Threshold {threshold:.2f}",
            )
            diagnostic_artifacts["test_best_checkpoint_global_threshold"] = _save_result_artifacts(
                result=best_test_result,
                cvdms_metadata=cvdms_metadata,
                output_dir=output_path / "diagnostics" / "test_best_checkpoint" / _global_threshold_dir_name(threshold),
                split_name="test_best_checkpoint_global_threshold",
                threshold_sweep_values=threshold_values if log_threshold_sweep else [],
                include_figures=log_test_figures,
                title_prefix=f"Test Diagnostics - Best Validation Checkpoint - Global Threshold {threshold:.2f}",
            )

            if per_class_threshold_strategy is not None:
                diagnostic_artifacts["threshold_strategy"] = _save_threshold_strategy_artifacts(
                    threshold_strategy=per_class_threshold_strategy,
                    output_dir=output_path / "diagnostics" / "threshold_strategy",
                )
                diagnostic_artifacts["best_validation_per_class_thresholds"] = _save_per_class_threshold_artifacts(
                    result=best_val_result,
                    cvdms_metadata=cvdms_metadata,
                    threshold_strategy=per_class_threshold_strategy,
                    output_dir=output_path / "diagnostics" / "best_validation" / "per_class_thresholds_from_val",
                    split_name="best_validation_per_class_thresholds_from_val",
                    include_figures=log_test_figures,
                    title_prefix="Best Validation Diagnostics - Per-Class Thresholds Derived from Validation",
                )
                diagnostic_artifacts["test_best_checkpoint_per_class_thresholds"] = _save_per_class_threshold_artifacts(
                    result=best_test_result,
                    cvdms_metadata=cvdms_metadata,
                    threshold_strategy=per_class_threshold_strategy,
                    output_dir=output_path / "diagnostics" / "test_best_checkpoint" / "per_class_thresholds_from_val",
                    split_name="test_best_checkpoint_per_class_thresholds_from_val",
                    include_figures=log_test_figures,
                    title_prefix="Test Diagnostics - Best Checkpoint - Per-Class Thresholds from Validation",
                )

        if writer is not None:
            _log_test_to_tensorboard(
                writer=writer,
                final_test_result=final_test_result,
                best_test_result=best_test_result,
                cvdms_metadata=cvdms_metadata,
                log_test_figures=log_test_figures,
                log_threshold_sweep=log_threshold_sweep,
                threshold_sweep_values=threshold_values,
                step=global_epoch,
            )
            if per_class_threshold_strategy is not None and log_test_figures:
                _log_per_class_threshold_figures_to_tensorboard(
                    writer=writer,
                    tag_prefix="diagnostics/test_best_checkpoint_per_class_thresholds_from_val",
                    result=best_test_result,
                    cvdms_metadata=cvdms_metadata,
                    threshold_strategy=per_class_threshold_strategy,
                    step=global_epoch,
                    title_prefix="Test Best Checkpoint - Per-Class Thresholds from Validation",
                )

        final_summary = {
            "history": history,
            "phases": phase_summaries,
            "best_checkpoint": best_tracker.summary(),
            "best_checkpoint_load": best_checkpoint_load,
            "threshold_strategies": {
                "global": {
                    "type": "global",
                    "threshold": threshold,
                },
                "per_class_validation_f1": per_class_threshold_strategy,
            },
            "validation_metrics_best_checkpoint_global_threshold": best_val_summary,
            "validation_metrics_best_checkpoint_per_class_thresholds": best_val_per_class_threshold_summary,
            "test_metrics_primary": "test_metrics_best_checkpoint_global_threshold",
            "test_metrics_best_checkpoint_global_threshold": best_test_summary,
            "test_metrics_best_checkpoint_per_class_thresholds": best_test_per_class_threshold_summary,
            "test_metrics_final_model_global_threshold": final_test_summary,
            # Backward-compatible aliases for older scripts/notebooks.
            "test_metrics_best_checkpoint": best_test_summary,
            "test_metrics_final_model": final_test_summary,
            "test_metrics": best_test_summary,
            "diagnostic_artifacts": diagnostic_artifacts,
            "metric_notes": _metric_notes(),
            "total_epochs": global_epoch,
            "threshold": threshold,
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
                            "test_best_checkpoint",
                            f"test_loss={best_test_result.loss:.4f}",
                            f"test_hamming_acc={best_test_result.hamming_accuracy:.4f}",
                            f"test_subset_acc={_format_optional_metric(best_test_result.subset_accuracy)}",
                            f"test_f1_micro={best_test_result.f1_micro:.4f}",
                            f"test_f1_macro={best_test_result.f1_macro:.4f}",
                            f"test_mAP={_format_optional_metric(best_test_result.macro_average_precision)}",
                            f"test_micro_AP={_format_optional_metric(best_test_result.micro_average_precision)}",
                            f"total_epochs={global_epoch}",
                        ]
                    )
                )
                if best_test_per_class_threshold_summary is not None:
                    print_fn(
                        " | ".join(
                            [
                                "test_best_checkpoint_per_class_thresholds",
                                f"test_loss={best_test_per_class_threshold_summary['loss']:.4f}",
                                f"test_hamming_acc={best_test_per_class_threshold_summary['hamming_accuracy']:.4f}",
                                f"test_subset_acc={_format_optional_metric(best_test_per_class_threshold_summary['subset_accuracy'])}",
                                f"test_f1_micro={best_test_per_class_threshold_summary['f1_micro']:.4f}",
                                f"test_f1_macro={best_test_per_class_threshold_summary['f1_macro']:.4f}",
                                f"test_mAP={_format_optional_metric(best_test_per_class_threshold_summary['macro_average_precision'])}",
                                "threshold_source=best_validation",
                            ]
                        )
                    )
                print_fn(
                    " | ".join(
                        [
                            "test_final_model_reference",
                            f"test_loss={final_test_result.loss:.4f}",
                            f"test_f1_macro={final_test_result.f1_macro:.4f}",
                            f"test_mAP={_format_optional_metric(final_test_result.macro_average_precision)}",
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
        "train_macro_average_precision": [],
        "train_map": [],
        "train_micro_average_precision": [],
        "train_micro_ap": [],
        "val_loss": [],
        "val_accuracy": [],
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
        "val_map": [],
        "val_micro_average_precision": [],
        "val_micro_ap": [],
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
    history[f"{split}_hamming_accuracy"].append(result.hamming_accuracy)
    history[f"{split}_hamming_loss"].append(result.hamming_loss)
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
    history[f"{split}_macro_average_precision"].append(result.macro_average_precision)
    history[f"{split}_map"].append(result.macro_average_precision)
    history[f"{split}_micro_average_precision"].append(result.micro_average_precision)
    history[f"{split}_micro_ap"].append(result.micro_average_precision)


def _new_phase_summary(
    *,
    phase_index: int,
    phase_name: str,
    max_epochs: int,
    trainable_layers: list[str],
    optimizer: Optimizer,
    model: nn.Module,
    early_stopping: EarlyStoppingState,
    threshold: float,
) -> dict[str, Any]:
    return {
        "phase_index": phase_index,
        "phase_name": phase_name,
        "max_epochs": max_epochs,
        "trainable_layers": list(trainable_layers),
        "threshold": threshold,
        "initial_learning_rates": get_current_learning_rates(optimizer),
        "initial_parameter_counts": count_parameters(model).to_dict(),
        "early_stopping_initial": early_stopping.summary(),
        "metric_notes": _metric_notes(),
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
    threshold_sweep_values: list[float],
) -> dict[str, Any]:
    pr_summary = None
    if include_precision_recall and result.probabilities is not None and result.targets is not None:
        pr_summary = precision_recall_summary(
            probabilities=result.probabilities,
            targets=result.targets,
            idx_to_class=cvdms_metadata.idx_to_class,
        )

    per_class_rows = per_class_metrics_table(
        result.confusion_counts,
        idx_to_class=cvdms_metadata.idx_to_class,
        pr_summary=pr_summary,
    )

    summary = {
        "loss": result.loss,
        "threshold": result.threshold,
        "accuracy": result.accuracy,
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
        "map": result.macro_average_precision,
        "micro_average_precision": result.micro_average_precision,
        "micro_ap": result.micro_average_precision,
        "total_examples": result.total_examples,
        "exact_match_count": result.exact_match_count,
        "elapsed_seconds": result.elapsed_seconds,
        "confusion_counts": confusion_counts_to_nested_list(result.confusion_counts),
        "per_class_metrics": per_class_metrics_with_names(
            result.confusion_counts,
            idx_to_class=cvdms_metadata.idx_to_class,
        ),
        "per_class_metrics_table": per_class_rows,
        "metric_notes": _metric_notes(),
    }

    if pr_summary is not None:
        summary["precision_recall"] = pr_summary.to_dict()

    if threshold_sweep_values and result.probabilities is not None and result.targets is not None:
        summary["threshold_sweep"] = threshold_sweep_summary(
            probabilities=result.probabilities,
            targets=result.targets,
            thresholds=threshold_sweep_values,
        )

    if result.probabilities is not None and result.targets is not None:
        summary["diagnostic_matrices"] = _build_diagnostic_matrices_summary(
            result=result,
            threshold=result.threshold,
        )

    return summary


def _build_diagnostic_matrices_summary(
    *,
    result: EpochResult,
    threshold: float,
) -> dict[str, Any]:
    if result.probabilities is None or result.targets is None:
        return {}

    conditional_matrix, conditional_support = conditional_prediction_probability_matrix(
        probabilities=result.probabilities,
        targets=result.targets,
    )
    false_matrix, false_support = false_association_probability_matrix(
        probabilities=result.probabilities,
        targets=result.targets,
    )
    cooccurrence_counts = thresholded_true_predicted_cooccurrence_matrix(
        probabilities=result.probabilities,
        targets=result.targets,
        threshold=threshold,
        normalize_rows=False,
    )
    cooccurrence_rates = thresholded_true_predicted_cooccurrence_matrix(
        probabilities=result.probabilities,
        targets=result.targets,
        threshold=threshold,
        normalize_rows=True,
    )
    missed_extra_counts = missed_vs_extra_label_matrix(
        probabilities=result.probabilities,
        targets=result.targets,
        threshold=threshold,
        normalize_rows=False,
    )
    missed_extra_rates = missed_vs_extra_label_matrix(
        probabilities=result.probabilities,
        targets=result.targets,
        threshold=threshold,
        normalize_rows=True,
    )

    return {
        "conditional_prediction_probability": {
            "interpretation": (
                "Row i selects samples where true class i is present; cell (i,j) "
                "is the average predicted probability for class j. This is an "
                "association diagnostic, not a confusion matrix."
            ),
            "matrix": matrix_to_nested_list(conditional_matrix),
            "row_support": conditional_support.astype(int).tolist(),
        },
        "false_association_probability": {
            "interpretation": (
                "Off-diagonal cell (i,j) averages p(class j) over samples where "
                "true class i is present and true class j is absent. The diagonal "
                "is mean p(class i) where true class i is present."
            ),
            "matrix": matrix_to_nested_list(false_matrix),
            "cell_support": false_support.astype(int).tolist(),
        },
        "thresholded_true_predicted_cooccurrence_counts": {
            "threshold": threshold,
            "matrix": matrix_to_nested_list(cooccurrence_counts),
        },
        "thresholded_true_predicted_cooccurrence_rates": {
            "threshold": threshold,
            "matrix": matrix_to_nested_list(cooccurrence_rates),
        },
        "missed_vs_extra_label_counts": {
            "threshold": threshold,
            "matrix": matrix_to_nested_list(missed_extra_counts),
        },
        "missed_vs_extra_label_rates": {
            "threshold": threshold,
            "matrix": matrix_to_nested_list(missed_extra_rates),
        },
    }



def _derive_per_class_threshold_strategy(
    *,
    validation_result: EpochResult,
    cvdms_metadata: CvdmsDatasetMetadata,
    fallback_threshold: float,
) -> dict[str, Any]:
    """
    Derive one decision threshold per class from the best checkpoint's
    validation predictions.

    This is intentionally Project-2-specific. It tests whether the model's
    weaker global-threshold metrics are partly caused by uneven per-class
    calibration. The validation set chooses thresholds; the test set only uses
    the frozen thresholds.
    """
    if validation_result.probabilities is None or validation_result.targets is None:
        raise ValueError("validation_result must contain probabilities and targets")

    pr_summary = precision_recall_summary(
        probabilities=validation_result.probabilities,
        targets=validation_result.targets,
        idx_to_class=cvdms_metadata.idx_to_class,
    )

    thresholds: list[float] = []
    thresholds_by_class: dict[str, float] = {}
    rows: list[dict[str, Any]] = []

    for class_idx in range(cvdms_metadata.num_classes):
        class_name = cvdms_metadata.idx_to_class[class_idx]
        class_summary = pr_summary.per_class[class_name]
        threshold_value = class_summary.best_threshold
        source = "validation_best_f1"

        if threshold_value is None:
            threshold_value = fallback_threshold
            source = "fallback_global_threshold"

        threshold_float = float(threshold_value)
        thresholds.append(threshold_float)
        thresholds_by_class[class_name] = threshold_float
        rows.append(
            {
                "class_idx": class_idx,
                "class_name": class_name,
                "threshold": threshold_float,
                "source": source,
                "validation_best_f1": class_summary.best_f1,
                "validation_best_precision": class_summary.best_precision,
                "validation_best_recall": class_summary.best_recall,
                "validation_average_precision": class_summary.average_precision,
                "validation_positive_count": class_summary.positive_count,
            }
        )

    return {
        "type": "per_class_best_f1_from_validation",
        "source_split": "validation",
        "source_model": "best_checkpoint",
        "fallback_threshold": fallback_threshold,
        "metric_optimized_per_class": "f1",
        "important_note": (
            "Thresholds are chosen from validation predictions only, then frozen "
            "before test evaluation. Test predictions are not used to choose thresholds."
        ),
        "thresholds": thresholds,
        "thresholds_by_class": thresholds_by_class,
        "rows": rows,
    }


def _threshold_vector_from_strategy(
    *,
    threshold_strategy: dict[str, Any],
    num_classes: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    raw = threshold_strategy.get("thresholds")
    if not isinstance(raw, list) or len(raw) != num_classes:
        raise ValueError(
            "threshold_strategy['thresholds'] must be a list with one threshold per class"
        )

    values = [float(value) for value in raw]
    for idx, value in enumerate(values):
        _validate_threshold(value, f"threshold_strategy.thresholds[{idx}]")
    return torch.tensor(values, dtype=dtype)


def _predictions_from_threshold_strategy(
    *,
    probabilities: torch.Tensor,
    threshold_strategy: dict[str, Any],
) -> torch.Tensor:
    if probabilities.ndim != 2:
        raise ValueError(
            "probabilities must have shape [num_examples, num_classes], "
            f"got {tuple(probabilities.shape)}"
        )

    probs_cpu = probabilities.detach().cpu()
    thresholds = _threshold_vector_from_strategy(
        threshold_strategy=threshold_strategy,
        num_classes=int(probs_cpu.shape[1]),
        dtype=probs_cpu.dtype,
    )
    return probs_cpu >= thresholds.unsqueeze(0)


def _confusion_counts_from_threshold_strategy(
    *,
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    threshold_strategy: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, int]:
    predictions = _predictions_from_threshold_strategy(
        probabilities=probabilities,
        threshold_strategy=threshold_strategy,
    )
    targets_cpu = targets.detach().cpu().to(dtype=torch.float32)
    counts = new_confusion_counts(num_classes=int(probabilities.shape[1]))
    update_confusion_counts_from_predictions(
        counts,
        predictions=predictions,
        targets=targets_cpu,
    )
    exact_match_count = int((predictions == targets_cpu.to(dtype=torch.bool)).all(dim=1).sum().item())
    return counts, predictions, exact_match_count


def _build_per_class_threshold_summary(
    *,
    result: EpochResult,
    cvdms_metadata: CvdmsDatasetMetadata,
    threshold_strategy: dict[str, Any],
) -> dict[str, Any]:
    if result.probabilities is None or result.targets is None:
        raise ValueError("result must contain probabilities and targets")

    counts, predictions, exact_match_count = _confusion_counts_from_threshold_strategy(
        probabilities=result.probabilities,
        targets=result.targets,
        threshold_strategy=threshold_strategy,
    )
    summary_metrics = summary_metrics_from_confusion_counts(
        counts,
        threshold=0.0,
        total_examples=int(result.targets.shape[0]),
        exact_match_count=exact_match_count,
    ).to_dict()
    summary_metrics["threshold"] = None
    summary_metrics["threshold_strategy"] = threshold_strategy

    pr_summary = precision_recall_summary(
        probabilities=result.probabilities,
        targets=result.targets,
        idx_to_class=cvdms_metadata.idx_to_class,
    )
    per_class_rows = per_class_metrics_table(
        counts,
        idx_to_class=cvdms_metadata.idx_to_class,
        pr_summary=pr_summary,
    )
    _attach_applied_thresholds_to_rows(
        rows=per_class_rows,
        threshold_strategy=threshold_strategy,
    )

    return {
        "loss": result.loss,
        **summary_metrics,
        "accuracy": summary_metrics["hamming_accuracy"],
        "macro_average_precision": pr_summary.macro_average_precision,
        "map": pr_summary.macro_average_precision,
        "micro_average_precision": pr_summary.micro_average_precision,
        "micro_ap": pr_summary.micro_average_precision,
        "total_examples": int(result.targets.shape[0]),
        "exact_match_count": exact_match_count,
        "elapsed_seconds": result.elapsed_seconds,
        "confusion_counts": confusion_counts_to_nested_list(counts),
        "per_class_metrics": per_class_metrics_with_names(
            counts,
            idx_to_class=cvdms_metadata.idx_to_class,
        ),
        "per_class_metrics_table": per_class_rows,
        "precision_recall": pr_summary.to_dict(),
        "diagnostic_matrices": _build_per_class_threshold_matrices_summary(
            probabilities=result.probabilities,
            targets=result.targets,
            threshold_strategy=threshold_strategy,
        ),
        "metric_notes": _metric_notes(),
    }


def _attach_applied_thresholds_to_rows(
    *,
    rows: list[dict[str, Any]],
    threshold_strategy: dict[str, Any],
) -> None:
    thresholds_by_class = threshold_strategy.get("thresholds_by_class", {})
    if not isinstance(thresholds_by_class, dict):
        return

    for row in rows:
        class_name = str(row.get("class_name"))
        row["applied_threshold"] = thresholds_by_class.get(class_name)
        row["threshold_strategy"] = threshold_strategy.get("type")


def _build_per_class_threshold_matrices_summary(
    *,
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    threshold_strategy: dict[str, Any],
) -> dict[str, Any]:
    conditional_matrix, conditional_support = conditional_prediction_probability_matrix(
        probabilities=probabilities,
        targets=targets,
    )
    false_matrix, false_support = false_association_probability_matrix(
        probabilities=probabilities,
        targets=targets,
    )
    cooccurrence_counts = _thresholded_true_predicted_cooccurrence_matrix_from_predictions(
        probabilities=probabilities,
        targets=targets,
        threshold_strategy=threshold_strategy,
        normalize_rows=False,
    )
    cooccurrence_rates = _thresholded_true_predicted_cooccurrence_matrix_from_predictions(
        probabilities=probabilities,
        targets=targets,
        threshold_strategy=threshold_strategy,
        normalize_rows=True,
    )
    missed_extra_counts = _missed_vs_extra_label_matrix_from_predictions(
        probabilities=probabilities,
        targets=targets,
        threshold_strategy=threshold_strategy,
        normalize_rows=False,
    )
    missed_extra_rates = _missed_vs_extra_label_matrix_from_predictions(
        probabilities=probabilities,
        targets=targets,
        threshold_strategy=threshold_strategy,
        normalize_rows=True,
    )

    return {
        "threshold_strategy": threshold_strategy,
        "conditional_prediction_probability": {
            "interpretation": (
                "Row i selects samples where true class i is present; cell (i,j) "
                "is the average predicted probability for class j. This is an "
                "association diagnostic, not a confusion matrix."
            ),
            "matrix": matrix_to_nested_list(conditional_matrix),
            "row_support": conditional_support.astype(int).tolist(),
        },
        "false_association_probability": {
            "interpretation": (
                "Off-diagonal cell (i,j) averages p(class j) over samples where "
                "true class i is present and true class j is absent. The diagonal "
                "is mean p(class i) where true class i is present."
            ),
            "matrix": matrix_to_nested_list(false_matrix),
            "cell_support": false_support.astype(int).tolist(),
        },
        "thresholded_true_predicted_cooccurrence_counts": {
            "threshold_strategy": threshold_strategy.get("type"),
            "matrix": matrix_to_nested_list(cooccurrence_counts),
        },
        "thresholded_true_predicted_cooccurrence_rates": {
            "threshold_strategy": threshold_strategy.get("type"),
            "matrix": matrix_to_nested_list(cooccurrence_rates),
        },
        "missed_vs_extra_label_counts": {
            "threshold_strategy": threshold_strategy.get("type"),
            "matrix": matrix_to_nested_list(missed_extra_counts),
        },
        "missed_vs_extra_label_rates": {
            "threshold_strategy": threshold_strategy.get("type"),
            "matrix": matrix_to_nested_list(missed_extra_rates),
        },
    }


def _thresholded_true_predicted_cooccurrence_matrix_from_predictions(
    *,
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    threshold_strategy: dict[str, Any],
    normalize_rows: bool,
) -> np.ndarray:
    predictions = _predictions_from_threshold_strategy(
        probabilities=probabilities,
        threshold_strategy=threshold_strategy,
    ).numpy().astype(np.float64)
    targets_np = targets.detach().cpu().numpy().astype(np.float64)
    matrix = targets_np.T @ predictions

    if not normalize_rows:
        return matrix

    support = targets_np.sum(axis=0)
    return np.divide(
        matrix,
        support[:, None],
        out=np.zeros_like(matrix, dtype=np.float64),
        where=support[:, None] > 0,
    )


def _missed_vs_extra_label_matrix_from_predictions(
    *,
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    threshold_strategy: dict[str, Any],
    normalize_rows: bool,
) -> np.ndarray:
    predictions = _predictions_from_threshold_strategy(
        probabilities=probabilities,
        threshold_strategy=threshold_strategy,
    ).numpy().astype(bool)
    targets_bool = targets.detach().cpu().numpy().astype(bool)
    missed = targets_bool & ~predictions
    extra = ~targets_bool & predictions
    matrix = missed.astype(np.float64).T @ extra.astype(np.float64)

    if not normalize_rows:
        return matrix

    missed_support = missed.sum(axis=0).astype(np.float64)
    return np.divide(
        matrix,
        missed_support[:, None],
        out=np.zeros_like(matrix, dtype=np.float64),
        where=missed_support[:, None] > 0,
    )


def _save_threshold_strategy_artifacts(
    *,
    threshold_strategy: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    strategy_json = output_path / "per_class_threshold_strategy.json"
    strategy_csv = output_path / "per_class_thresholds.csv"
    write_json(threshold_strategy, strategy_json)
    write_metric_rows_csv(threshold_strategy.get("rows", []), strategy_csv)
    return {
        "strategy_json": str(strategy_json),
        "thresholds_csv": str(strategy_csv),
    }


def _save_per_class_threshold_artifacts(
    *,
    result: EpochResult,
    cvdms_metadata: CvdmsDatasetMetadata,
    threshold_strategy: dict[str, Any],
    output_dir: str | Path,
    split_name: str,
    include_figures: bool,
    title_prefix: str,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary = _build_per_class_threshold_summary(
        result=result,
        cvdms_metadata=cvdms_metadata,
        threshold_strategy=threshold_strategy,
    )
    per_class_rows = list(summary["per_class_metrics_table"])

    manifest: dict[str, Any] = {
        "split_name": split_name,
        "output_dir": str(output_path),
        "threshold_strategy": threshold_strategy,
        "metric_notes": _metric_notes(),
        "files": {},
    }

    per_class_csv = output_path / "per_class_metrics.csv"
    per_class_json = output_path / "per_class_metrics.json"
    summary_json = output_path / "summary.json"
    matrices_json = output_path / "diagnostic_matrices.json"
    thresholds_json = output_path / "threshold_strategy.json"
    thresholds_csv = output_path / "per_class_thresholds.csv"

    write_metric_rows_csv(per_class_rows, per_class_csv)
    write_json(per_class_rows, per_class_json)
    write_json(summary, summary_json)
    write_json(summary["diagnostic_matrices"], matrices_json)
    write_json(threshold_strategy, thresholds_json)
    write_metric_rows_csv(threshold_strategy.get("rows", []), thresholds_csv)

    manifest["files"].update(
        {
            "per_class_metrics_csv": str(per_class_csv),
            "per_class_metrics_json": str(per_class_json),
            "summary_json": str(summary_json),
            "diagnostic_matrices_json": str(matrices_json),
            "threshold_strategy_json": str(thresholds_json),
            "per_class_thresholds_csv": str(thresholds_csv),
        }
    )

    if include_figures:
        manifest["figures"] = _save_per_class_threshold_figures(
            result=result,
            cvdms_metadata=cvdms_metadata,
            threshold_strategy=threshold_strategy,
            per_class_rows=per_class_rows,
            output_dir=output_path / "figures",
            title_prefix=title_prefix,
        )

    write_json(manifest, output_path / "artifact_manifest.json")
    return manifest


def _save_per_class_threshold_figures(
    *,
    result: EpochResult,
    cvdms_metadata: CvdmsDatasetMetadata,
    threshold_strategy: dict[str, Any],
    per_class_rows: list[dict[str, Any]],
    output_dir: str | Path,
    title_prefix: str,
    threshold_sweep_values: list[float] | None = None,
) -> dict[str, Any]:
    if result.probabilities is None or result.targets is None:
        return {}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    counts, _, _ = _confusion_counts_from_threshold_strategy(
        probabilities=result.probabilities,
        targets=result.targets,
        threshold_strategy=threshold_strategy,
    )
    cooccurrence_rates = _thresholded_true_predicted_cooccurrence_matrix_from_predictions(
        probabilities=result.probabilities,
        targets=result.targets,
        threshold_strategy=threshold_strategy,
        normalize_rows=True,
    )
    missed_extra_rates = _missed_vs_extra_label_matrix_from_predictions(
        probabilities=result.probabilities,
        targets=result.targets,
        threshold_strategy=threshold_strategy,
        normalize_rows=True,
    )

    figures = {
        "binary_confusion_grid": make_binary_confusion_grid_figure(
            confusion_counts=counts,
            idx_to_class=cvdms_metadata.idx_to_class,
            title_prefix=f"{title_prefix} - Per-Class Binary Confusion Matrices",
        ),
        "per_class_f1_bar": make_per_class_metric_bar_figure(
            rows=per_class_rows,
            metric_key="f1",
            title=f"{title_prefix} - Per-Class F1",
            ylabel="F1 using validation-derived per-class thresholds",
        ),
        "per_class_precision_bar": make_per_class_metric_bar_figure(
            rows=per_class_rows,
            metric_key="precision",
            title=f"{title_prefix} - Per-Class Precision",
            ylabel="Precision using validation-derived per-class thresholds",
        ),
        "per_class_recall_bar": make_per_class_metric_bar_figure(
            rows=per_class_rows,
            metric_key="recall",
            title=f"{title_prefix} - Per-Class Recall",
            ylabel="Recall using validation-derived per-class thresholds",
        ),
        "thresholded_cooccurrence_heatmap": make_matrix_heatmap_figure(
            matrix=cooccurrence_rates,
            idx_to_class=cvdms_metadata.idx_to_class,
            title=f"{title_prefix} - True Label vs Predicted Label Co-occurrence",
            xlabel="Predicted-positive class",
            ylabel="True-present class",
            colorbar_label="Row-normalized rate",
            value_format=".2f",
        ),
        "missed_vs_extra_heatmap": make_matrix_heatmap_figure(
            matrix=missed_extra_rates,
            idx_to_class=cvdms_metadata.idx_to_class,
            title=f"{title_prefix} - Missed Label vs Extra Label Errors",
            xlabel="Extra predicted class",
            ylabel="Missed true class",
            colorbar_label="Row-normalized rate",
            value_format=".2f",
        ),
        "conditional_prediction_probability_heatmap": make_conditional_prediction_probability_heatmap_figure(
            probabilities=result.probabilities,
            targets=result.targets,
            idx_to_class=cvdms_metadata.idx_to_class,
            title=f"{title_prefix} - Conditional Prediction Probability Heatmap",
        ),
        "false_association_probability_heatmap": make_false_association_probability_heatmap_figure(
            probabilities=result.probabilities,
            targets=result.targets,
            idx_to_class=cvdms_metadata.idx_to_class,
            title=f"{title_prefix} - False-Association Probability Heatmap",
        ),
        "precision_recall_combined": make_precision_recall_figure(
            probabilities=result.probabilities,
            targets=result.targets,
            idx_to_class=cvdms_metadata.idx_to_class,
            title_prefix=f"{title_prefix} - Combined Precision-Recall Curves",
            annotate_best_f1=False,
        ),
        "precision_recall_small_multiples": make_precision_recall_small_multiples_figure(
            probabilities=result.probabilities,
            targets=result.targets,
            idx_to_class=cvdms_metadata.idx_to_class,
            title_prefix=f"{title_prefix} - PR Curves by Class",
            annotate_best_f1=False,
        ),
    }

    if any(row.get("average_precision") is not None for row in per_class_rows):
        figures["per_class_ap_bar"] = make_per_class_metric_bar_figure(
            rows=per_class_rows,
            metric_key="average_precision",
            title=f"{title_prefix} - Per-Class Average Precision",
            ylabel="AP",
        )

    figure_files = save_figures(figures, output_path, close=True)
    per_class_pr_files = save_figures(
        make_per_class_precision_recall_figures(
            probabilities=result.probabilities,
            targets=result.targets,
            idx_to_class=cvdms_metadata.idx_to_class,
            title_prefix=f"{title_prefix} - Precision-Recall",
            annotate_best_f1=False,
        ),
        output_path / "precision_recall_by_class",
        close=True,
    )
    return {
        "summary_figures": figure_files,
        "per_class_precision_recall_figures": per_class_pr_files,
    }


def _save_result_artifacts(
    *,
    result: EpochResult,
    cvdms_metadata: CvdmsDatasetMetadata,
    output_dir: str | Path,
    split_name: str,
    threshold_sweep_values: list[float],
    include_figures: bool,
    title_prefix: str,
) -> dict[str, Any]:
    """
    Save rich diagnostics for one result object and return an artifact manifest.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pr_summary = None
    if result.probabilities is not None and result.targets is not None:
        pr_summary = precision_recall_summary(
            probabilities=result.probabilities,
            targets=result.targets,
            idx_to_class=cvdms_metadata.idx_to_class,
        )

    per_class_rows = per_class_metrics_table(
        result.confusion_counts,
        idx_to_class=cvdms_metadata.idx_to_class,
        pr_summary=pr_summary,
    )

    manifest: dict[str, Any] = {
        "split_name": split_name,
        "output_dir": str(output_path),
        "metric_notes": _metric_notes(),
        "files": {},
    }

    per_class_csv = output_path / "per_class_metrics.csv"
    per_class_json = output_path / "per_class_metrics.json"
    write_metric_rows_csv(per_class_rows, per_class_csv)
    write_json(per_class_rows, per_class_json)
    manifest["files"]["per_class_metrics_csv"] = str(per_class_csv)
    manifest["files"]["per_class_metrics_json"] = str(per_class_json)

    summary_json = output_path / "summary.json"
    write_json(
        _build_result_summary(
            result=result,
            cvdms_metadata=cvdms_metadata,
            include_precision_recall=True,
            threshold_sweep_values=threshold_sweep_values,
        ),
        summary_json,
    )
    manifest["files"]["summary_json"] = str(summary_json)

    if threshold_sweep_values and result.probabilities is not None and result.targets is not None:
        sweep_rows = threshold_sweep_summary(
            probabilities=result.probabilities,
            targets=result.targets,
            thresholds=threshold_sweep_values,
        )
        sweep_csv = output_path / "threshold_sweep.csv"
        sweep_json = output_path / "threshold_sweep.json"
        write_metric_rows_csv(sweep_rows, sweep_csv)
        write_json(sweep_rows, sweep_json)
        manifest["files"]["threshold_sweep_csv"] = str(sweep_csv)
        manifest["files"]["threshold_sweep_json"] = str(sweep_json)

    if result.probabilities is not None and result.targets is not None:
        matrices_json = output_path / "diagnostic_matrices.json"
        write_json(
            _build_diagnostic_matrices_summary(result=result, threshold=result.threshold),
            matrices_json,
        )
        manifest["files"]["diagnostic_matrices_json"] = str(matrices_json)

    if include_figures:
        figure_manifest = _save_result_figures(
            result=result,
            cvdms_metadata=cvdms_metadata,
            per_class_rows=per_class_rows,
            output_dir=output_path / "figures",
            title_prefix=title_prefix,
            threshold_sweep_values=threshold_sweep_values,
        )
        manifest["figures"] = figure_manifest

    write_json(manifest, output_path / "artifact_manifest.json")
    return manifest


def _save_result_figures(
    *,
    result: EpochResult,
    cvdms_metadata: CvdmsDatasetMetadata,
    per_class_rows: list[dict[str, Any]],
    output_dir: str | Path,
    title_prefix: str,
    threshold_sweep_values: list[float] | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    figures = {
        "binary_confusion_grid": make_binary_confusion_grid_figure(
            confusion_counts=result.confusion_counts,
            idx_to_class=cvdms_metadata.idx_to_class,
            title_prefix=f"{title_prefix} - Per-Class Binary Confusion Matrices",
        ),
        "per_class_f1_bar": make_per_class_metric_bar_figure(
            rows=per_class_rows,
            metric_key="f1",
            title=f"{title_prefix} - Per-Class F1",
            ylabel="F1 at configured threshold",
        ),
        "per_class_precision_bar": make_per_class_metric_bar_figure(
            rows=per_class_rows,
            metric_key="precision",
            title=f"{title_prefix} - Per-Class Precision",
            ylabel="Precision at configured threshold",
        ),
        "per_class_recall_bar": make_per_class_metric_bar_figure(
            rows=per_class_rows,
            metric_key="recall",
            title=f"{title_prefix} - Per-Class Recall",
            ylabel="Recall at configured threshold",
        ),
    }

    if any(row.get("average_precision") is not None for row in per_class_rows):
        figures["per_class_ap_bar"] = make_per_class_metric_bar_figure(
            rows=per_class_rows,
            metric_key="average_precision",
            title=f"{title_prefix} - Per-Class Average Precision",
            ylabel="AP",
        )

    if result.probabilities is not None and result.targets is not None:
        figures.update(
            {
                "precision_recall_combined": make_precision_recall_figure(
                    probabilities=result.probabilities,
                    targets=result.targets,
                    idx_to_class=cvdms_metadata.idx_to_class,
                    title_prefix=f"{title_prefix} - Combined Precision-Recall Curves",
                    annotate_best_f1=False,
                ),
                "precision_recall_small_multiples": make_precision_recall_small_multiples_figure(
                    probabilities=result.probabilities,
                    targets=result.targets,
                    idx_to_class=cvdms_metadata.idx_to_class,
                    title_prefix=f"{title_prefix} - PR Curves by Class",
                    annotate_best_f1=False,
                ),
                "conditional_prediction_probability_heatmap": make_conditional_prediction_probability_heatmap_figure(
                    probabilities=result.probabilities,
                    targets=result.targets,
                    idx_to_class=cvdms_metadata.idx_to_class,
                    title=f"{title_prefix} - Conditional Prediction Probability Heatmap",
                ),
                "false_association_probability_heatmap": make_false_association_probability_heatmap_figure(
                    probabilities=result.probabilities,
                    targets=result.targets,
                    idx_to_class=cvdms_metadata.idx_to_class,
                    title=f"{title_prefix} - False-Association Probability Heatmap",
                ),
                "thresholded_cooccurrence_heatmap": make_thresholded_cooccurrence_heatmap_figure(
                    probabilities=result.probabilities,
                    targets=result.targets,
                    idx_to_class=cvdms_metadata.idx_to_class,
                    threshold=result.threshold,
                    normalize_rows=True,
                    title=f"{title_prefix} - True Label vs Predicted Label Co-occurrence",
                ),
                "missed_vs_extra_heatmap": make_missed_vs_extra_heatmap_figure(
                    probabilities=result.probabilities,
                    targets=result.targets,
                    idx_to_class=cvdms_metadata.idx_to_class,
                    threshold=result.threshold,
                    normalize_rows=True,
                    title=f"{title_prefix} - Missed Label vs Extra Label Errors",
                ),
            }
        )

    if threshold_sweep_values and result.probabilities is not None and result.targets is not None:
        sweep_rows = threshold_sweep_summary(
            probabilities=result.probabilities,
            targets=result.targets,
            thresholds=threshold_sweep_values,
        )
        figures.update(
            _make_threshold_sweep_figures(
                rows=sweep_rows,
                title_prefix=f"{title_prefix} - Threshold Sweep",
            )
        )

    figure_files = save_figures(figures, output_path, close=True)
    per_class_pr_files: dict[str, str] = {}

    if result.probabilities is not None and result.targets is not None:
        per_class_figures = make_per_class_precision_recall_figures(
            probabilities=result.probabilities,
            targets=result.targets,
            idx_to_class=cvdms_metadata.idx_to_class,
            title_prefix=f"{title_prefix} - Precision-Recall",
            annotate_best_f1=False,
        )
        per_class_pr_files = save_figures(
            per_class_figures,
            output_path / "precision_recall_by_class",
            close=True,
        )

    return {
        "summary_figures": figure_files,
        "per_class_precision_recall_figures": per_class_pr_files,
    }


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
    log_val_precision_recall_curve: bool,
    log_threshold_sweep: bool,
    threshold_sweep_values: list[float],
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

    if log_val_precision_recall_curve and val_result.probabilities is not None:
        _log_precision_recall_figure(
            writer=writer,
            tag="precision_recall/val_combined",
            result=val_result,
            cvdms_metadata=cvdms_metadata,
            step=global_epoch,
            title_prefix=f"Validation Multi-Label PR Curves - {phase_name} - Epoch {global_epoch}",
        )

    if log_threshold_sweep and threshold_sweep_values:
        _log_threshold_sweep_to_tensorboard(
            writer=writer,
            tag_prefix="threshold_sweep/val",
            result=val_result,
            thresholds=threshold_sweep_values,
            step=global_epoch,
        )


def _log_test_to_tensorboard(
    *,
    writer: SummaryWriter,
    final_test_result: EpochResult,
    best_test_result: EpochResult,
    cvdms_metadata: CvdmsDatasetMetadata,
    log_test_figures: bool,
    log_threshold_sweep: bool,
    threshold_sweep_values: list[float],
    step: int,
) -> None:
    _log_scalar_metrics(writer=writer, split="test_best_checkpoint", result=best_test_result, step=step)
    _log_scalar_metrics(writer=writer, split="test_final_model", result=final_test_result, step=step)

    if log_threshold_sweep and threshold_sweep_values:
        _log_threshold_sweep_to_tensorboard(
            writer=writer,
            tag_prefix="threshold_sweep/test_best_checkpoint",
            result=best_test_result,
            thresholds=threshold_sweep_values,
            step=step,
        )
        _log_threshold_sweep_to_tensorboard(
            writer=writer,
            tag_prefix="threshold_sweep/test_final_model",
            result=final_test_result,
            thresholds=threshold_sweep_values,
            step=step,
        )

    if not log_test_figures:
        return

    _log_result_figures_to_tensorboard(
        writer=writer,
        tag_prefix="diagnostics/test_best_checkpoint",
        result=best_test_result,
        cvdms_metadata=cvdms_metadata,
        step=step,
        title_prefix="Test Best Checkpoint",
    )
    _log_result_figures_to_tensorboard(
        writer=writer,
        tag_prefix="diagnostics/test_final_model",
        result=final_test_result,
        cvdms_metadata=cvdms_metadata,
        step=step,
        title_prefix="Test Final Model",
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
        writer.add_scalar(f"average_precision_macro/{split}", result.macro_average_precision, step)
        writer.add_scalar(f"mAP/{split}", result.macro_average_precision, step)

    if result.micro_average_precision is not None:
        writer.add_scalar(f"average_precision_micro/{split}", result.micro_average_precision, step)
        writer.add_scalar(f"micro_AP/{split}", result.micro_average_precision, step)


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


def _log_result_figures_to_tensorboard(
    *,
    writer: SummaryWriter,
    tag_prefix: str,
    result: EpochResult,
    cvdms_metadata: CvdmsDatasetMetadata,
    step: int,
    title_prefix: str,
) -> None:
    if result.probabilities is None or result.targets is None:
        return

    figures = {
        "precision_recall_combined": make_precision_recall_figure(
            probabilities=result.probabilities,
            targets=result.targets,
            idx_to_class=cvdms_metadata.idx_to_class,
            title_prefix=f"{title_prefix} - Combined PR Curves",
            annotate_best_f1=False,
        ),
        "precision_recall_small_multiples": make_precision_recall_small_multiples_figure(
            probabilities=result.probabilities,
            targets=result.targets,
            idx_to_class=cvdms_metadata.idx_to_class,
            title_prefix=f"{title_prefix} - PR Curves by Class",
            annotate_best_f1=False,
        ),
        "binary_confusion_grid": make_binary_confusion_grid_figure(
            confusion_counts=result.confusion_counts,
            idx_to_class=cvdms_metadata.idx_to_class,
            title_prefix=f"{title_prefix} - Per-Class Binary Confusion Matrices",
        ),
        "conditional_prediction_probability_heatmap": make_conditional_prediction_probability_heatmap_figure(
            probabilities=result.probabilities,
            targets=result.targets,
            idx_to_class=cvdms_metadata.idx_to_class,
            title=f"{title_prefix} - Conditional Prediction Probability Heatmap",
        ),
        "false_association_probability_heatmap": make_false_association_probability_heatmap_figure(
            probabilities=result.probabilities,
            targets=result.targets,
            idx_to_class=cvdms_metadata.idx_to_class,
            title=f"{title_prefix} - False-Association Probability Heatmap",
        ),
        "thresholded_cooccurrence_heatmap": make_thresholded_cooccurrence_heatmap_figure(
            probabilities=result.probabilities,
            targets=result.targets,
            idx_to_class=cvdms_metadata.idx_to_class,
            threshold=result.threshold,
            normalize_rows=True,
            title=f"{title_prefix} - True Label vs Predicted Label Co-occurrence",
        ),
        "missed_vs_extra_heatmap": make_missed_vs_extra_heatmap_figure(
            probabilities=result.probabilities,
            targets=result.targets,
            idx_to_class=cvdms_metadata.idx_to_class,
            threshold=result.threshold,
            normalize_rows=True,
            title=f"{title_prefix} - Missed Label vs Extra Label Errors",
        ),
    }

    for name, fig in figures.items():
        try:
            writer.add_figure(f"{tag_prefix}/{name}", fig, step)
        finally:
            plt.close(fig)



def _log_per_class_threshold_figures_to_tensorboard(
    *,
    writer: SummaryWriter,
    tag_prefix: str,
    result: EpochResult,
    cvdms_metadata: CvdmsDatasetMetadata,
    threshold_strategy: dict[str, Any],
    step: int,
    title_prefix: str,
) -> None:
    if result.probabilities is None or result.targets is None:
        return

    counts, _, _ = _confusion_counts_from_threshold_strategy(
        probabilities=result.probabilities,
        targets=result.targets,
        threshold_strategy=threshold_strategy,
    )
    summary = _build_per_class_threshold_summary(
        result=result,
        cvdms_metadata=cvdms_metadata,
        threshold_strategy=threshold_strategy,
    )
    per_class_rows = summary["per_class_metrics_table"]
    cooccurrence_rates = _thresholded_true_predicted_cooccurrence_matrix_from_predictions(
        probabilities=result.probabilities,
        targets=result.targets,
        threshold_strategy=threshold_strategy,
        normalize_rows=True,
    )
    missed_extra_rates = _missed_vs_extra_label_matrix_from_predictions(
        probabilities=result.probabilities,
        targets=result.targets,
        threshold_strategy=threshold_strategy,
        normalize_rows=True,
    )

    figures = {
        "binary_confusion_grid": make_binary_confusion_grid_figure(
            confusion_counts=counts,
            idx_to_class=cvdms_metadata.idx_to_class,
            title_prefix=f"{title_prefix} - Per-Class Binary Confusion Matrices",
        ),
        "per_class_f1_bar": make_per_class_metric_bar_figure(
            rows=per_class_rows,
            metric_key="f1",
            title=f"{title_prefix} - Per-Class F1",
            ylabel="F1 using validation-derived per-class thresholds",
        ),
        "per_class_precision_bar": make_per_class_metric_bar_figure(
            rows=per_class_rows,
            metric_key="precision",
            title=f"{title_prefix} - Per-Class Precision",
            ylabel="Precision using validation-derived per-class thresholds",
        ),
        "per_class_recall_bar": make_per_class_metric_bar_figure(
            rows=per_class_rows,
            metric_key="recall",
            title=f"{title_prefix} - Per-Class Recall",
            ylabel="Recall using validation-derived per-class thresholds",
        ),
        "thresholded_cooccurrence_heatmap": make_matrix_heatmap_figure(
            matrix=cooccurrence_rates,
            idx_to_class=cvdms_metadata.idx_to_class,
            title=f"{title_prefix} - True Label vs Predicted Label Co-occurrence",
            xlabel="Predicted-positive class",
            ylabel="True-present class",
            colorbar_label="Row-normalized rate",
            value_format=".2f",
        ),
        "missed_vs_extra_heatmap": make_matrix_heatmap_figure(
            matrix=missed_extra_rates,
            idx_to_class=cvdms_metadata.idx_to_class,
            title=f"{title_prefix} - Missed Label vs Extra Label Errors",
            xlabel="Extra predicted class",
            ylabel="Missed true class",
            colorbar_label="Row-normalized rate",
            value_format=".2f",
        ),
    }

    for name, fig in figures.items():
        try:
            writer.add_figure(f"{tag_prefix}/{name}", fig, step)
        finally:
            plt.close(fig)


def _make_threshold_sweep_figures(
    *,
    rows: list[dict[str, Any]],
    title_prefix: str,
) -> dict[str, Any]:
    if not rows:
        return {}

    thresholds = [float(row["threshold"]) for row in rows]

    def _plot_lines(name: str, title: str, metric_keys: list[str], ylabel: str = "Metric value"):
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        for key in metric_keys:
            values = [np.nan if row.get(key) is None else float(row[key]) for row in rows]
            ax.plot(thresholds, values, marker="o", linewidth=1.8, label=key)
        ax.set_title(f"{title_prefix} - {title}")
        ax.set_xlabel("Global threshold")
        ax.set_ylabel(ylabel)
        ax.set_xlim(min(thresholds), max(thresholds))
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        return name, fig

    figures = dict(
        [
            _plot_lines(
                "macro_precision_recall_f1",
                "Macro Precision / Recall / F1 vs Threshold",
                ["precision_macro", "recall_macro", "f1_macro"],
            ),
            _plot_lines(
                "micro_precision_recall_f1",
                "Micro Precision / Recall / F1 vs Threshold",
                ["precision_micro", "recall_micro", "f1_micro"],
            ),
            _plot_lines(
                "accuracy",
                "Accuracy Metrics vs Threshold",
                ["hamming_accuracy", "subset_accuracy"],
            ),
            _plot_lines(
                "weighted_precision_recall_f1",
                "Weighted Precision / Recall / F1 vs Threshold",
                ["precision_weighted", "recall_weighted", "f1_weighted"],
            ),
        ]
    )
    return figures


def _log_threshold_sweep_to_tensorboard(
    *,
    writer: SummaryWriter,
    tag_prefix: str,
    result: EpochResult,
    thresholds: list[float],
    step: int,
) -> None:
    """
    Log threshold sweep curves to TensorBoard.

    The old implementation wrote one scalar tag per metric per threshold, which
    created many one-point TensorBoard plots. This Project 2 version logs compact
    figures instead: threshold on the x-axis, metric value on the y-axis.
    """
    if result.probabilities is None or result.targets is None:
        return

    rows = threshold_sweep_summary(
        probabilities=result.probabilities,
        targets=result.targets,
        thresholds=thresholds,
    )
    figures = _make_threshold_sweep_figures(rows=rows, title_prefix=tag_prefix.replace("/", " - "))

    for name, fig in figures.items():
        try:
            writer.add_figure(f"{tag_prefix}/{name}", fig, step)
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
            f"train_hamm={train_result.hamming_accuracy:.4f}",
            f"train_f1_macro={train_result.f1_macro:.4f}",
            f"val_loss={val_result.loss:.4f}",
            f"val_hamm={val_result.hamming_accuracy:.4f}",
            f"val_subset={_format_optional_metric(val_result.subset_accuracy)}",
            f"val_f1_macro={val_result.f1_macro:.4f}",
            f"val_mAP={_format_optional_metric(val_result.macro_average_precision)}",
            f"val_micro_AP={_format_optional_metric(val_result.micro_average_precision)}",
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
        hamming_accuracy=result.hamming_accuracy,
        hamming_loss=result.hamming_loss,
        subset_accuracy=result.subset_accuracy if result.subset_accuracy is not None else 0.0,
        precision_micro=result.precision_micro,
        recall_micro=result.recall_micro,
        f1_micro=result.f1_micro,
        precision_macro=result.precision_macro,
        recall_macro=result.recall_macro,
        f1_macro=result.f1_macro,
        precision_weighted=result.precision_weighted,
        recall_weighted=result.recall_weighted,
        f1_weighted=result.f1_weighted,
        macro_average_precision=result.macro_average_precision,
        micro_average_precision=result.micro_average_precision,
    )


def _select_metric_value(
    *,
    metric_name: str,
    train_result: EpochResult,
    val_result: EpochResult,
) -> float:
    supported = {
        "val_accuracy": val_result.accuracy,
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
        "val_map": val_result.macro_average_precision,
        "val_mAP": val_result.macro_average_precision,
        "val_micro_average_precision": val_result.micro_average_precision,
        "val_micro_ap": val_result.micro_average_precision,
        "val_micro_AP": val_result.micro_average_precision,
        "train_accuracy": train_result.accuracy,
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
        "train_macro_average_precision": train_result.macro_average_precision,
        "train_map": train_result.macro_average_precision,
        "train_mAP": train_result.macro_average_precision,
        "train_micro_average_precision": train_result.micro_average_precision,
        "train_micro_ap": train_result.micro_average_precision,
        "train_micro_AP": train_result.micro_average_precision,
    }

    if metric_name not in supported:
        raise ValueError(
            "Unsupported metric name. Expected one of "
            f"{sorted(supported)}, got {metric_name!r}"
        )

    value = supported[metric_name]
    if value is None:
        raise ValueError(
            f"Metric {metric_name!r} is None. If using AP/mAP as the best metric, "
            "make sure probability/PR data is collected for that split."
        )

    return float(value)


def _best_metric_requires_pr_data(metric_name: str) -> bool:
    return metric_name in {
        "val_macro_average_precision",
        "val_map",
        "val_mAP",
        "val_micro_average_precision",
        "val_micro_ap",
        "val_micro_AP",
        "train_macro_average_precision",
        "train_map",
        "train_mAP",
        "train_micro_average_precision",
        "train_micro_ap",
        "train_micro_AP",
    }


def _load_checkpoint_model_state(
    *,
    checkpoint_path: Path,
    model: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = _extract_model_state_dict(checkpoint)
    adapted_state_dict = _adapt_state_dict_keys_for_model(state_dict, model)
    missing, unexpected = model.load_state_dict(adapted_state_dict, strict=False)

    if missing or unexpected:
        raise RuntimeError(
            "Loaded best checkpoint with incompatible keys. "
            f"missing={missing}, unexpected={unexpected}"
        )

    return {
        "path": str(checkpoint_path),
        "checkpoint_keys": sorted(checkpoint.keys()) if isinstance(checkpoint, dict) else [],
        "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "metric_name": checkpoint.get("metric_name") if isinstance(checkpoint, dict) else None,
        "metric_value": checkpoint.get("metric_value") if isinstance(checkpoint, dict) else None,
    }


def _extract_model_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in (
            "model_state_dict",
            "model_state",
            "state_dict",
            "model",
            "net",
        ):
            value = checkpoint.get(key)
            if _looks_like_state_dict(value):
                return value
        if _looks_like_state_dict(checkpoint):
            return checkpoint

    raise ValueError(
        "Could not find a model state_dict in checkpoint. Expected one of "
        "model_state_dict, model_state, state_dict, model, or net."
    )


def _looks_like_state_dict(value: Any) -> bool:
    return isinstance(value, dict) and bool(value) and all(
        isinstance(key, str) and torch.is_tensor(tensor)
        for key, tensor in value.items()
    )


def _adapt_state_dict_keys_for_model(
    state_dict: dict[str, torch.Tensor],
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    model_keys = list(model.state_dict().keys())
    state_keys = list(state_dict.keys())
    if not model_keys or not state_keys:
        return state_dict

    model_uses_module = model_keys[0].startswith("module.")
    state_uses_module = state_keys[0].startswith("module.")

    if state_uses_module and not model_uses_module:
        return {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

    if model_uses_module and not state_uses_module:
        return {
            f"module.{key}": value
            for key, value in state_dict.items()
        }

    return state_dict


def _metric_notes() -> dict[str, str]:
    return {
        "multi_label_setup": (
            "Each image can have zero, one, or many positive classes. The model "
            "outputs raw logits and metrics apply sigmoid(logits) to obtain probabilities."
        ),
        "threshold": (
            "Thresholded metrics convert probabilities to binary predictions using "
            "prediction = probability >= threshold. The default threshold is usually 0.5."
        ),
        "hamming_accuracy": (
            "Fraction of individual class decisions that are correct across all images "
            "and classes. It can look high on imbalanced datasets because most labels are absent."
        ),
        "subset_accuracy": (
            "Also called exact-match accuracy. A sample is correct only if the entire "
            "predicted label set exactly matches the true label set. This is strict for multi-label tasks."
        ),
        "micro_f1": (
            "Aggregates TP, FP, and FN across all classes first, then computes F1. "
            "Common classes tend to dominate this score."
        ),
        "macro_f1": (
            "Computes F1 separately per class, then averages classes equally. This is "
            "useful for detecting whether rare classes are being ignored."
        ),
        "weighted_f1": (
            "Computes F1 separately per class, then averages using true class support. "
            "It sits between macro and micro behavior."
        ),
        "average_precision": (
            "Per-class threshold-free precision-recall ranking summary. It measures "
            "whether positive examples for a class are ranked above negative examples."
        ),
        "map": (
            "mAP is macro-averaged AP: compute AP separately for each class, then "
            "average those AP values across supported classes. In code this is macro_average_precision."
        ),
        "micro_ap": (
            "Micro-AP flattens all image/class decisions into one binary ranking problem "
            "before computing AP. Common classes and common decisions influence it more."
        ),
        "conditional_prediction_probability_heatmap": (
            "Row i selects samples where true class i is present. Cell (i,j) is the "
            "average predicted probability for class j. High off-diagonal values may be legitimate co-occurrence."
        ),
        "false_association_probability_heatmap": (
            "Off-diagonal cell (i,j) averages predicted probability for class j when "
            "true class i is present but true class j is absent. This better highlights false associations."
        ),
    }


def _global_threshold_dir_name(threshold: float) -> str:
    """
    Return a stable, filesystem-friendly folder name for a global threshold.

    Examples:
        0.5  -> "global_threshold_0_50"
        0.75 -> "global_threshold_0_75"

    Keeping the threshold in the folder name makes it clear which diagnostics
    were computed with the default/global scalar threshold versus the separate
    validation-derived per-class threshold strategy.
    """
    _validate_threshold(threshold, "threshold")
    return f"global_threshold_{float(threshold):.2f}".replace(".", "_")

def _format_optional_metric(value: float | None) -> str:
    if value is None:
        return "None"
    return f"{value:.4f}"


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


def _validate_threshold(value: Any, field_name: str) -> None:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a float in [0, 1], got {value!r}")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1], got {number}")


def _normalize_threshold_sweep_values(values: list[float] | None) -> list[float]:
    if values is None:
        return []
    if not isinstance(values, list):
        raise TypeError(f"threshold_sweep_values must be a list, got {type(values).__name__}")

    thresholds: list[float] = []
    for idx, value in enumerate(values):
        _validate_threshold(value, f"threshold_sweep_values[{idx}]")
        thresholds.append(float(value))
    return sorted(set(thresholds))


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