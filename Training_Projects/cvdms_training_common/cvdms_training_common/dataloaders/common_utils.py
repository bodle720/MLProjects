from typing import Any
from torch.utils.data import DataLoader

def build_dataloader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    sampler,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int | None,
    drop_last: bool,
) -> DataLoader:
    """
    Build a DataLoader while avoiding invalid argument combinations.

    PyTorch only allows prefetch_factor and persistent_workers when
    num_workers > 0.
    """
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
    }

    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers

        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(**kwargs)

def validate_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")

    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")

def validate_nonnegative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")

    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")