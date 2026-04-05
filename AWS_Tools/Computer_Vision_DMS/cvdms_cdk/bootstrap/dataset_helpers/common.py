from __future__ import annotations

import csv
import json
import mimetypes
import random
import urllib.request
import zipfile
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA = "cvdms.manifest.v1"

TASK_CHOICES = (
    "single-label",
    "multi-label",
    "object-detection",
    "semantic-segmentation",
    "instance-segmentation",
)


@dataclass(frozen=True)
class BootstrapConfig:
    dataset: str
    task: str
    bucket: str
    s3_prefix: str
    aws_region: str | None
    max_items: int | None
    sample_seed: int
    output_dir: Path
    work_dir: Path
    keep_work_dir: bool = False
    overwrite_existing: bool = False


@dataclass
class BootstrapFailure:
    dataset_item_id: str
    reason: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class BootstrapResult:
    manifest_rows: list[dict[str, Any]]
    failures: list[BootstrapFailure] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


class DatasetBootstrapper(ABC):
    dataset_name: str
    supported_tasks: set[str]

    def validate_task(self, task: str) -> None:
        if task not in self.supported_tasks:
            supported = ", ".join(sorted(self.supported_tasks))
            raise ValueError(
                f"Task '{task}' is not valid for dataset '{self.dataset_name}'. "
                f"Supported tasks: {supported}"
            )

    @abstractmethod
    def bootstrap(self, config: BootstrapConfig, s3_client: Any) -> BootstrapResult:
        raise NotImplementedError


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def task_slug(task: str) -> str:
    return task.replace("-", "_")


def build_run_dir_name(dataset: str, task: str, dt: datetime) -> str:
    stamp = dt.strftime("%Y%m%d_%H%M")
    return f"{dataset}_{task_slug(task)}_{stamp}"


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def write_csv_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    preferred_order = [
        "schema",
        "label_type",
        "source_ref",
        "labels",
        "mask_ref",
        "color_map",
        "worker_response_ref",
    ]

    discovered_keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                discovered_keys.append(key)

    ordered_keys = [k for k in preferred_order if k in seen]
    ordered_keys.extend(k for k in discovered_keys if k not in ordered_keys)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_cell(row.get(k)) for k in ordered_keys})


def write_failures_json(path: Path, failures: list[BootstrapFailure]) -> None:
    payload = [asdict(item) for item in failures]
    write_json(path, payload)


def s3_key_join(*parts: str) -> str:
    cleaned = [part.strip("/") for part in parts if part and part.strip("/")]
    return "/".join(cleaned)


def s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def upload_file_to_s3(
    s3_client: Any,
    local_path: Path,
    bucket: str,
    key: str,
) -> str:
    extra_args: dict[str, Any] = {}
    guessed_type, _ = mimetypes.guess_type(str(local_path))
    if guessed_type:
        extra_args["ContentType"] = guessed_type

    if extra_args:
        s3_client.upload_file(str(local_path), bucket, key, ExtraArgs=extra_args)
    else:
        s3_client.upload_file(str(local_path), bucket, key)

    return s3_uri(bucket, key)


def upload_bytes_to_s3(
    s3_client: Any,
    payload: bytes,
    bucket: str,
    key: str,
    content_type: str | None = None,
) -> str:
    extra_args: dict[str, Any] = {}
    if content_type:
        extra_args["ContentType"] = content_type
    s3_client.put_object(Bucket=bucket, Key=key, Body=payload, **extra_args)
    return s3_uri(bucket, key)


def deterministic_sample[T](items: list[T], max_items: int | None, seed: int) -> list[T]:
    if max_items is None or len(items) <= max_items:
        return items
    rng = random.Random(seed)
    sampled = rng.sample(items, max_items)
    return list(sampled)


def download_http_file(url: str, destination: Path) -> Path:
    ensure_dir(destination.parent)
    if destination.exists():
        return destination

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "cvdms-dataset-bootstrap/1.0",
        },
    )

    with urllib.request.urlopen(request) as response, destination.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)

    return destination


def extract_zip(zip_path: Path, destination_dir: Path) -> Path:
    ensure_dir(destination_dir)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(destination_dir)
    return destination_dir


def make_single_label_row(source_ref: str, label: str) -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA,
        "label_type": "single-label",
        "source_ref": source_ref,
        "labels": [label],
    }


def make_multi_label_row(source_ref: str, labels: list[str]) -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA,
        "label_type": "multi-label",
        "source_ref": source_ref,
        "labels": labels,
    }


def make_object_detection_row(
    source_ref: str,
    annotations: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA,
        "label_type": "object-detection",
        "source_ref": source_ref,
        "labels": annotations,
    }


def make_semantic_segmentation_row(
    source_ref: str,
    mask_ref: str,
    color_map: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA,
        "label_type": "semantic-segmentation",
        "source_ref": source_ref,
        "mask_ref": mask_ref,
        "color_map": color_map,
    }


def make_instance_segmentation_row(
    source_ref: str,
    worker_response_ref: str,
) -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA,
        "label_type": "instance-segmentation",
        "source_ref": source_ref,
        "worker_response_ref": worker_response_ref,
    }