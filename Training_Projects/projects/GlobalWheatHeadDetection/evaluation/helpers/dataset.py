from pathlib import Path
from typing import Any

import yaml

from evaluation.helpers.paths import resolve_project_path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_dataset_yaml(data_yaml: str | Path) -> dict[str, Any]:
    data_yaml_path = resolve_project_path(data_yaml)

    with data_yaml_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)

    payload["_data_yaml_path"] = str(data_yaml_path)
    payload["_dataset_root"] = str(resolve_dataset_root(data_yaml_path, payload))
    return payload


def resolve_dataset_root(data_yaml_path: Path, dataset_payload: dict[str, Any]) -> Path:
    root_value = dataset_payload.get("path")

    if root_value is None:
        return data_yaml_path.parent

    root_path = Path(root_value)

    if root_path.is_absolute():
        return root_path

    return data_yaml_path.parent / root_path


def get_class_names(dataset_payload: dict[str, Any]) -> dict[int, str]:
    names = dataset_payload.get("names", {})

    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}

    if isinstance(names, dict):
        return {int(idx): str(name) for idx, name in names.items()}

    return {}


def get_split_image_paths(dataset_payload: dict[str, Any], split: str) -> list[Path]:
    if split not in dataset_payload:
        raise KeyError(f"Split '{split}' not found in dataset YAML.")

    dataset_root = Path(dataset_payload["_dataset_root"])
    split_value = dataset_payload[split]
    split_paths = split_value if isinstance(split_value, list) else [split_value]

    image_paths: list[Path] = []

    for raw_path in split_paths:
        split_path = Path(raw_path)

        if not split_path.is_absolute():
            split_path = dataset_root / split_path

        image_paths.extend(resolve_images_from_path(split_path))

    return sorted(set(image_paths))


def resolve_images_from_path(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() == ".txt":
        return read_image_list_file(path)

    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [path]

    if path.is_dir():
        return [
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        ]

    raise FileNotFoundError(f"Could not resolve YOLO split path: {path}")


def read_image_list_file(path: Path) -> list[Path]:
    image_paths: list[Path] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            raw = line.strip()

            if not raw:
                continue

            candidate = Path(raw)

            if not candidate.is_absolute():
                candidate = path.parent / candidate

            image_paths.append(candidate)

    return image_paths