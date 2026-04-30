"""
Checkpoint helpers for CVDMS training projects.

This module provides small utilities for saving model checkpoints together with
training metadata and CVDMS dataset provenance.

The goal is to make every saved model traceable back to:
    - dataset_id
    - dataset version
    - metadata.json URI
    - label_type
    - effective_classes
    - class_to_idx / idx_to_class
    - training hyperparameters
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:  # pragma: no cover - older PyTorch fallback
    LRScheduler = Any  # type: ignore

from cvdms_training_common.metadata import CvdmsDatasetMetadata

@dataclass(frozen=True)
class CheckpointMetadata:
    """
    Serializable metadata saved alongside model weights.

    This is intentionally generic. Project-specific training scripts can pass
    model_name, hyperparameters, notes, and arbitrary extra metadata.
    """

    dataset: dict[str, Any]
    model_name: str
    epoch: int
    metric_name: str | None = None
    metric_value: float | None = None
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

def build_checkpoint_payload(
    *,
    model: nn.Module,
    epoch: int,
    cvdms_metadata: CvdmsDatasetMetadata,
    model_name: str,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    metric_name: str | None = None,
    metric_value: float | None = None,
    hyperparameters: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a checkpoint payload suitable for torch.save().

    Args:
        model:
            PyTorch model.
        epoch:
            Epoch number associated with this checkpoint.
        cvdms_metadata:
            Version-scoped CVDMS dataset metadata.
        model_name:
            Human-readable model identifier, e.g. "resnet18".
        optimizer:
            Optional optimizer. If provided, its state is saved.
        scheduler:
            Optional learning-rate scheduler. If provided, its state is saved.
        metric_name:
            Optional metric associated with the checkpoint, e.g. "val_accuracy".
        metric_value:
            Optional metric value.
        hyperparameters:
            Optional project/training hyperparameters.
        extra:
            Optional extra JSON-serializable metadata.

    Returns:
        Dictionary ready for torch.save().
    """
    if epoch < 0:
        raise ValueError(f"epoch must be >= 0, got {epoch}")

    metadata = CheckpointMetadata(
        dataset=cvdms_metadata.as_training_provenance(),
        model_name=_require_nonempty_string(model_name, "model_name"),
        epoch=epoch,
        metric_name=metric_name,
        metric_value=metric_value,
        hyperparameters=dict(hyperparameters or {}),
        extra=dict(extra or {}),
    )

    payload: dict[str, Any] = {
        "format": "cvdms_training_checkpoint",
        "format_version": 1,
        "epoch": epoch,
        "model_name": metadata.model_name,
        "model_state_dict": model.state_dict(),
        "metadata": asdict(metadata),
    }

    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()

    return payload

def save_checkpoint(
    *,
    path: str | Path,
    model: nn.Module,
    epoch: int,
    cvdms_metadata: CvdmsDatasetMetadata,
    model_name: str,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    metric_name: str | None = None,
    metric_value: float | None = None,
    hyperparameters: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """
    Save a CVDMS-aware PyTorch checkpoint.

    Returns:
        Path to the saved checkpoint.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = build_checkpoint_payload(
        model=model,
        epoch=epoch,
        cvdms_metadata=cvdms_metadata,
        model_name=model_name,
        optimizer=optimizer,
        scheduler=scheduler,
        metric_name=metric_name,
        metric_value=metric_value,
        hyperparameters=hyperparameters,
        extra=extra,
    )

    torch.save(payload, output_path)
    return output_path

def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device | None = "cpu",
) -> dict[str, Any]:
    """
    Load a checkpoint payload.

    This function returns the raw checkpoint dictionary so project-specific code
    can decide how to rebuild the model architecture before loading weights.
    """
    checkpoint_path = Path(path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location=map_location)

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected checkpoint payload to be a dict, got {type(payload).__name__}"
        )

    return payload

def load_model_state_dict(
    *,
    model: nn.Module,
    checkpoint_path: str | Path,
    map_location: str | torch.device | None = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """
    Load model weights from a checkpoint into an already-constructed model.

    The caller is responsible for constructing the correct model architecture.
    """
    payload = load_checkpoint(checkpoint_path, map_location=map_location)

    state_dict = payload.get("model_state_dict")
    if state_dict is None:
        raise ValueError(f"Checkpoint is missing 'model_state_dict': {checkpoint_path}")

    model.load_state_dict(state_dict, strict=strict)
    return payload

def save_json_artifact(
    *,
    path: str | Path,
    payload: dict[str, Any],
) -> Path:
    """
    Save a JSON artifact with deterministic formatting.

    Useful for:
        - evaluation_summary.json
        - cvdms_training_metadata.json
        - class_map.json
        - hyperparameters.json
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return output_path

def save_cvdms_training_metadata(
    *,
    path: str | Path,
    cvdms_metadata: CvdmsDatasetMetadata,
    metadata_uri: str,
    model_name: str | None = None,
    hyperparameters: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """
    Save a compact training provenance JSON artifact.

    This should be saved alongside model outputs so a future reader can identify
    exactly which CVDMS dataset version trained the model.
    """
    payload: dict[str, Any] = {
        "metadata_uri": metadata_uri,
        "dataset": cvdms_metadata.as_training_provenance(),
        "hyperparameters": dict(hyperparameters or {}),
        "extra": dict(extra or {}),
    }

    if model_name is not None:
        payload["model_name"] = _require_nonempty_string(model_name, "model_name")

    return save_json_artifact(path=path, payload=payload)

def save_class_map(
    *,
    path: str | Path,
    cvdms_metadata: CvdmsDatasetMetadata,
) -> Path:
    """
    Save class mapping information as a standalone JSON artifact.
    """
    payload = {
        "dataset_id": cvdms_metadata.dataset_id,
        "version": cvdms_metadata.version,
        "label_type": cvdms_metadata.label_type,
        "effective_classes": list(cvdms_metadata.effective_classes),
        "class_to_idx": dict(cvdms_metadata.class_to_idx),
        "idx_to_class": {
            str(idx): class_name
            for idx, class_name in cvdms_metadata.idx_to_class.items()
        },
        "num_classes": cvdms_metadata.num_classes,
    }

    return save_json_artifact(path=path, payload=payload)

def is_better_metric(
    *,
    current: float,
    best: float | None,
    mode: str,
) -> bool:
    """
    Decide whether a metric improved.

    Args:
        current:
            Current metric value.
        best:
            Previous best value. If None, current is always better.
        mode:
            "max" for metrics where larger is better, e.g. accuracy.
            "min" for metrics where smaller is better, e.g. loss.
    """
    if mode not in {"max", "min"}:
        raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")

    if best is None:
        return True

    if mode == "max":
        return current > best

    return current < best

@dataclass
class BestCheckpointTracker:
    """
    Helper for saving best checkpoints during training.

    Example:
        tracker = BestCheckpointTracker(
            output_dir="outputs/model",
            filename="best_checkpoint.pt",
            metric_name="val_accuracy",
            mode="max",
        )

        tracker.maybe_save(
            metric_value=val_acc,
            model=model,
            epoch=epoch,
            cvdms_metadata=metadata,
            model_name="resnet18",
            optimizer=optimizer,
            hyperparameters=hparams,
        )
    """

    output_dir: str | Path
    filename: str
    metric_name: str
    mode: str

    best_value: float | None = None
    best_epoch: int | None = None
    best_path: Path | None = None

    def maybe_save(
        self,
        *,
        metric_value: float,
        model: nn.Module,
        epoch: int,
        cvdms_metadata: CvdmsDatasetMetadata,
        model_name: str,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        hyperparameters: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> bool:
        """
        Save checkpoint if metric_value improves over the current best.

        Returns:
            True if a new best checkpoint was saved, otherwise False.
        """
        if not is_better_metric(
            current=metric_value,
            best=self.best_value,
            mode=self.mode,
        ):
            return False

        self.best_value = metric_value
        self.best_epoch = epoch

        output_path = Path(self.output_dir) / self.filename
        self.best_path = save_checkpoint(
            path=output_path,
            model=model,
            epoch=epoch,
            cvdms_metadata=cvdms_metadata,
            model_name=model_name,
            optimizer=optimizer,
            scheduler=scheduler,
            metric_name=self.metric_name,
            metric_value=metric_value,
            hyperparameters=hyperparameters,
            extra=extra,
        )

        return True

    def summary(self) -> dict[str, Any]:
        """
        Return JSON-serializable best-checkpoint summary.
        """
        return {
            "metric_name": self.metric_name,
            "mode": self.mode,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "best_path": str(self.best_path) if self.best_path else None,
        }

def _require_nonempty_string(value: Any, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} cannot be null")

    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} cannot be empty")

    return text

def _json_safe(value: Any) -> Any:
    """
    Convert common non-JSON-native values into JSON-safe equivalents.

    This is intentionally conservative. It handles the common values expected in
    training summaries without trying to serialize arbitrary Python objects.
    """
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)