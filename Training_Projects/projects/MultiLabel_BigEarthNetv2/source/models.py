"""
Project-specific model utilities for the CVDMS BigEarthNet v2 multi-label classifier.

This project uses a staged transfer-learning approach:

Phase 1:
    Train only the classifier head.

Phase 2:
    Unfreeze later ResNet feature layers plus classifier head.

Phase 3:
    Unfreeze more feature layers plus classifier head.

These utilities are intentionally project-specific because layer names such as
"layer3", "layer4", and "fc" are ResNet-specific.

For multi-label classification, the model outputs raw logits of shape
[batch_size, num_classes]. Do not add sigmoid here. BCEWithLogitsLoss expects
raw logits during training, and sigmoid should only be applied for metrics or
inference.
"""

from dataclasses import dataclass
from typing import Iterable

import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torchvision import models

_ALLOWED_TRAINABLE_LAYERS = {"conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"}

@dataclass(frozen=True)
class ParameterCounts:
    total_parameters: int
    trainable_parameters: int
    frozen_parameters: int

    def to_dict(self) -> dict[str, int]:
        return {
            "total_parameters": self.total_parameters,
            "trainable_parameters": self.trainable_parameters,
            "frozen_parameters": self.frozen_parameters,
        }

def build_model(
    *,
    model_name: str,
    num_classes: int,
    pretrained: bool,
) -> nn.Module:
    """
    Build a project-supported multi-label classifier model.
    """
    if model_name == "resnet18":
        return build_resnet18_classifier(
            num_classes=num_classes,
            pretrained=pretrained,
        )

    raise ValueError(f"Unsupported model_name={model_name!r}")

def build_resnet18_classifier(
    *,
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    """
    Build a ResNet18 multi-label classifier with a new task-specific final layer.

    The final layer returns raw logits, not sigmoid probabilities.

    The model is returned with all layers frozen by default except the newly
    created classifier head. The staged trainer will later call
    configure_trainable_layers(...) at the beginning of each phase.
    """
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}")

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    freeze_all_parameters(model)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model

def freeze_all_parameters(model: nn.Module) -> None:
    """
    Freeze every parameter in the model.
    """
    for parameter in model.parameters():
        parameter.requires_grad = False

def configure_trainable_layers(
    model: nn.Module,
    *,
    trainable_layers: Iterable[str],
) -> None:
    """
    Freeze all model parameters, then unfreeze selected top-level ResNet modules.

    Supported layer names:
        conv1, bn1, layer1, layer2, layer3, layer4, fc

    Example:
        configure_trainable_layers(model, trainable_layers=["layer4", "fc"])
    """
    requested_layers = list(trainable_layers)
    _validate_trainable_layers(requested_layers)

    freeze_all_parameters(model)

    for layer_name in requested_layers:
        module = getattr(model, layer_name, None)
        if module is None:
            raise ValueError(f"Model does not have layer {layer_name!r}")

        for parameter in module.parameters():
            parameter.requires_grad = True

def build_optimizer_for_phase(
    model: nn.Module,
    *,
    trainable_layers: Iterable[str],
    head_lr: float,
    backbone_lr: float | None,
    weight_decay: float,
) -> Optimizer:
    """
    Build an AdamW optimizer for the current staged-training phase.

    The classifier head uses head_lr. Any other trainable layers use backbone_lr.

    For a head-only phase:
        trainable_layers = ["fc"]
        backbone_lr = None

    For a fine-tuning phase:
        trainable_layers = ["layer4", "fc"]
        head_lr = 3e-4
        backbone_lr = 5e-5
    """
    requested_layers = list(trainable_layers)
    _validate_trainable_layers(requested_layers)

    if head_lr <= 0:
        raise ValueError(f"head_lr must be > 0, got {head_lr}")

    if backbone_lr is not None and backbone_lr <= 0:
        raise ValueError(f"backbone_lr must be > 0 when provided, got {backbone_lr}")

    if weight_decay < 0:
        raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")

    head_parameters = list(_trainable_parameters_for_layer(model, "fc"))
    backbone_parameters: list[nn.Parameter] = []

    for layer_name in requested_layers:
        if layer_name == "fc":
            continue

        backbone_parameters.extend(_trainable_parameters_for_layer(model, layer_name))

    parameter_groups: list[dict] = []

    if head_parameters:
        parameter_groups.append(
            {
                "params": head_parameters,
                "lr": head_lr,
                "name": "head",
            }
        )

    if backbone_parameters:
        if backbone_lr is None:
            raise ValueError(
                "backbone_lr is required when trainable_layers includes non-fc layers"
            )

        parameter_groups.append(
            {
                "params": backbone_parameters,
                "lr": backbone_lr,
                "name": "backbone",
            }
        )

    if not parameter_groups:
        raise ValueError("No trainable parameters found for this phase")

    return AdamW(
        parameter_groups,
        weight_decay=weight_decay,
    )

def count_parameters(model: nn.Module) -> ParameterCounts:
    """
    Count total, trainable, and frozen parameters.
    """
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return ParameterCounts(
        total_parameters=total,
        trainable_parameters=trainable,
        frozen_parameters=total - trainable,
    )

def get_current_learning_rates(optimizer: Optimizer) -> dict[str, float]:
    """
    Return current learning rates by optimizer parameter group name.

    If a group has no explicit name, it is assigned group_<index>.
    """
    learning_rates: dict[str, float] = {}

    for idx, group in enumerate(optimizer.param_groups):
        group_name = str(group.get("name", f"group_{idx}"))
        learning_rates[group_name] = float(group["lr"])

    return learning_rates

def _trainable_parameters_for_layer(
    model: nn.Module,
    layer_name: str,
) -> list[nn.Parameter]:
    module = getattr(model, layer_name, None)
    if module is None:
        raise ValueError(f"Model does not have layer {layer_name!r}")

    return [
        parameter
        for parameter in module.parameters()
        if parameter.requires_grad
    ]

def _validate_trainable_layers(trainable_layers: list[str]) -> None:
    if not trainable_layers:
        raise ValueError("trainable_layers cannot be empty")

    unknown = sorted(set(trainable_layers) - _ALLOWED_TRAINABLE_LAYERS)
    if unknown:
        raise ValueError(
            f"Unsupported trainable layer(s): {unknown}. "
            f"Allowed values: {sorted(_ALLOWED_TRAINABLE_LAYERS)}"
        )

    if "fc" not in trainable_layers:
        raise ValueError(
            "trainable_layers should include 'fc' so the classifier head remains trainable"
        )