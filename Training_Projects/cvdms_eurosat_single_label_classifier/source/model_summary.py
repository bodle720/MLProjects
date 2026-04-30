"""
Model summary helpers for the CVDMS EuroSAT single-label classifier.

These utilities save architecture and parameter summaries to text files so each
training phase is easy to inspect later. This is especially useful for staged
fine-tuning, where the set of trainable layers changes by phase.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch.nn as nn
from torchinfo import summary

from models import count_parameters, get_current_learning_rates

def save_model_summary(
    *,
    model: nn.Module,
    output_dir: str | Path,
    filename: str,
    batch_size: int,
    channels: int,
    image_size: int,
    phase_name: str | None = None,
    phase_index: int | None = None,
    optimizer=None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """
    Save a readable model summary text file.

    Args:
        model:
            PyTorch model to summarize.
        output_dir:
            Directory where the summary file should be written.
        filename:
            Output text filename.
        batch_size:
            Batch size used for torchinfo input shape.
        channels:
            Number of image channels, usually 3 for RGB or 1 for grayscale.
        image_size:
            Image height/width used by the training transform.
        phase_name:
            Optional staged-training phase name.
        phase_index:
            Optional 1-based phase index.
        optimizer:
            Optional optimizer. If provided, current parameter-group learning
            rates are included.
        extra:
            Optional additional metadata to write into the summary.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_path = output_path / filename
    parameter_counts = count_parameters(model)

    lines: list[str] = []

    lines.append("CVDMS Model Summary")
    lines.append("=" * 80)
    lines.append("")

    if phase_name is not None:
        lines.append(f"Phase name: {phase_name}")

    if phase_index is not None:
        lines.append(f"Phase index: {phase_index}")

    lines.append(f"Input size: ({batch_size}, {channels}, {image_size}, {image_size})")
    lines.append("")

    lines.append("Parameter counts")
    lines.append("-" * 80)
    lines.append(f"Total parameters:     {parameter_counts.total_parameters:,}")
    lines.append(f"Trainable parameters: {parameter_counts.trainable_parameters:,}")
    lines.append(f"Frozen parameters:    {parameter_counts.frozen_parameters:,}")
    lines.append("")

    if optimizer is not None:
        lines.append("Learning rates")
        lines.append("-" * 80)

        for group_name, lr in get_current_learning_rates(optimizer).items():
            lines.append(f"{group_name}: {lr}")

        lines.append("")

    if extra:
        lines.append("Extra metadata")
        lines.append("-" * 80)

        for key, value in sorted(extra.items()):
            lines.append(f"{key}: {value}")

        lines.append("")

    lines.append("Architecture")
    lines.append("-" * 80)
    lines.append(str(model))
    lines.append("")

    lines.append("torchinfo summary")
    lines.append("-" * 80)
    lines.append(
        _build_torchinfo_summary(
            model=model,
            batch_size=batch_size,
            channels=channels,
            image_size=image_size,
        )
    )
    lines.append("")

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path

def save_initial_model_summary(
    *,
    model: nn.Module,
    output_dir: str | Path,
    batch_size: int,
    channels: int,
    image_size: int,
) -> Path:
    """
    Save a model summary before staged training begins.
    """
    return save_model_summary(
        model=model,
        output_dir=output_dir,
        filename="model_summary_initial.txt",
        batch_size=batch_size,
        channels=channels,
        image_size=image_size,
        phase_name="initial",
        phase_index=0,
    )

def save_phase_model_summary(
    *,
    model: nn.Module,
    output_dir: str | Path,
    phase_index: int,
    phase_name: str,
    batch_size: int,
    channels: int,
    image_size: int,
    optimizer=None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """
    Save a model summary after phase-specific trainability and optimizer setup.
    """
    safe_phase_name = _safe_filename_part(phase_name)

    return save_model_summary(
        model=model,
        output_dir=output_dir,
        filename=f"model_summary_phase_{phase_index}_{safe_phase_name}.txt",
        batch_size=batch_size,
        channels=channels,
        image_size=image_size,
        phase_name=phase_name,
        phase_index=phase_index,
        optimizer=optimizer,
        extra=extra,
    )

def _build_torchinfo_summary(
    *,
    model: nn.Module,
    batch_size: int,
    channels: int,
    image_size: int,
) -> str:
    try:
        model_summary = summary(
            model,
            input_size=(batch_size, channels, image_size, image_size),
            col_names=("input_size", "output_size", "num_params", "trainable"),
            verbose=0,
        )
        return str(model_summary)
    except Exception as exc:
        return (
            "torchinfo summary failed.\n"
            f"Error type: {type(exc).__name__}\n"
            f"Error message: {exc}\n"
        )

def _safe_filename_part(value: str) -> str:
    text = value.strip().lower()
    safe_chars: list[str] = []

    for char in text:
        if char.isalnum():
            safe_chars.append(char)
        elif char in {"-", "_"}:
            safe_chars.append(char)
        elif char.isspace():
            safe_chars.append("_")
        else:
            safe_chars.append("_")

    safe = "".join(safe_chars).strip("_")
    return safe or "phase"