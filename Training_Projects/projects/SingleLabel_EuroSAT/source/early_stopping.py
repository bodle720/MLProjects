"""
Early stopping utilities for staged transfer learning.

Early stopping is applied within each training phase. A phase can end early if
the monitored validation metric stops improving, but it never runs longer than
the phase's configured max_epochs.

This project uses early stopping to decide whether to shorten a phase, not to
decide the entire training schedule.
"""

from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class EarlyStoppingConfig:
    enabled: bool
    monitor: str
    mode: str
    patience: int
    min_delta: float

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EarlyStoppingConfig":
        enabled = bool(payload.get("enabled", True))
        monitor = str(payload.get("monitor", "val_f1_macro")).strip()
        mode = str(payload.get("mode", "max")).strip()
        patience = int(payload.get("patience", 3))
        min_delta = float(payload.get("min_delta", 0.0))

        if not monitor:
            raise ValueError("early_stopping.monitor cannot be empty")

        if mode not in {"max", "min"}:
            raise ValueError(f"early_stopping.mode must be 'max' or 'min', got {mode!r}")

        if patience < 1:
            raise ValueError(f"early_stopping.patience must be >= 1, got {patience}")

        if min_delta < 0:
            raise ValueError(f"early_stopping.min_delta must be >= 0, got {min_delta}")

        return cls(
            enabled=enabled,
            monitor=monitor,
            mode=mode,
            patience=patience,
            min_delta=min_delta,
        )

@dataclass
class EarlyStoppingState:
    config: EarlyStoppingConfig
    best_value: float | None = None
    best_epoch: int | None = None
    epochs_without_improvement: int = 0
    stopped: bool = False
    stop_epoch: int | None = None
    stop_reason: str | None = None

    def update(self, *, epoch: int, metrics: dict[str, float]) -> bool:
        """
        Update early-stopping state.

        Args:
            epoch:
                Global or phase-local epoch number used for reporting.
            metrics:
                Dictionary containing the monitored metric.

        Returns:
            True if training should stop for this phase, otherwise False.
        """
        if not self.config.enabled:
            return False

        if self.config.monitor not in metrics:
            raise ValueError(
                f"Monitored metric {self.config.monitor!r} not found. "
                f"Available metrics: {sorted(metrics)}"
            )

        current_value = float(metrics[self.config.monitor])

        if self._is_improvement(current_value):
            self.best_value = current_value
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            return False

        self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.config.patience:
            self.stopped = True
            self.stop_epoch = epoch
            self.stop_reason = (
                f"No improvement in {self.config.monitor!r} for "
                f"{self.config.patience} consecutive epoch(s)"
            )
            return True

        return False

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "monitor": self.config.monitor,
            "mode": self.config.mode,
            "patience": self.config.patience,
            "min_delta": self.config.min_delta,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "epochs_without_improvement": self.epochs_without_improvement,
            "stopped": self.stopped,
            "stop_epoch": self.stop_epoch,
            "stop_reason": self.stop_reason,
        }

    def _is_improvement(self, current_value: float) -> bool:
        if self.best_value is None:
            return True

        if self.config.mode == "max":
            return current_value > self.best_value + self.config.min_delta

        return current_value < self.best_value - self.config.min_delta

def make_early_stopping_state(payload: dict[str, Any] | None) -> EarlyStoppingState:
    """
    Build an EarlyStoppingState from a config dictionary.

    If payload is None, early stopping is disabled.
    """
    if payload is None:
        config = EarlyStoppingConfig(
            enabled=False,
            monitor="val_f1_macro",
            mode="max",
            patience=1,
            min_delta=0.0,
        )
        return EarlyStoppingState(config=config)

    return EarlyStoppingState(config=EarlyStoppingConfig.from_dict(payload))

def metric_value_from_results(
    *,
    monitor: str,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
) -> float:
    """
    Resolve a monitored metric from train/val metric dictionaries.

    Examples:
        monitor='val_f1_macro'
        monitor='val_accuracy'
        monitor='train_loss'
    """
    if monitor.startswith("train_"):
        source = train_metrics
    elif monitor.startswith("val_"):
        source = val_metrics
    else:
        raise ValueError(
            f"monitor must start with 'train_' or 'val_', got {monitor!r}"
        )

    if monitor not in source:
        raise ValueError(
            f"Metric {monitor!r} not found. Available metrics: {sorted(source)}"
        )

    return float(source[monitor])

def epoch_metrics_dict(
    *,
    split: str,
    loss: float,
    accuracy: float,
    precision_macro: float,
    recall_macro: float,
    f1_macro: float,
    precision_weighted: float,
    recall_weighted: float,
    f1_weighted: float,
) -> dict[str, float]:
    """
    Build a metric dictionary with split-prefixed keys.

    Example keys:
        val_loss
        val_accuracy
        val_f1_macro
    """
    return {
        f"{split}_loss": float(loss),
        f"{split}_accuracy": float(accuracy),
        f"{split}_precision_macro": float(precision_macro),
        f"{split}_recall_macro": float(recall_macro),
        f"{split}_f1_macro": float(f1_macro),
        f"{split}_precision_weighted": float(precision_weighted),
        f"{split}_recall_weighted": float(recall_weighted),
        f"{split}_f1_weighted": float(f1_weighted),
    }