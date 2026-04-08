import os
import socket
import time
import urllib.error
import urllib.request
import csv
import json
import math
import mimetypes
import random
import zipfile
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

TASK_CHOICES = (
    "single-label",
    "multi-label",
    "object-detection",
    "semantic-segmentation",
    "instance-segmentation",
)

_RESERVED_CLASS_NAMES_LC = {"bg", "background"}
_BACKGROUND_NAMES_LC = {"bg", "background"}

@dataclass(frozen=True)
class BootstrapConfig:
    dataset: str
    task: str
    bucket: str
    s3_prefix: str
    aws_region: str | None
    reuse_from_run_dir: Path | None
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
    stamp = dt.strftime("%Y%m%d_%H%M%S")
    return f"{dataset}_{task_slug(task)}_{stamp}"

def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

def write_jsonl_manifest(path: Path, task: str, rows: list[dict[str, Any]]) -> None:
    """
    Writes the minimal GT-shaped JSONL that the upload validator accepts.
    No extra Ground Truth fluff is emitted.
    """
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            payload = _manifest_row_to_minimal_gt_json(task, row)
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")

def write_csv_manifest(path: Path, task: str, rows: list[dict[str, Any]]) -> None:
    """
    Writes the exact task-specific CSV format that the upload validator accepts.
    """
    with path.open("w", encoding="utf-8", newline="") as handle:
        if task == "single-label":
            fieldnames = ["source-ref", "class-name"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "source-ref": _require_nonempty_string(row.get("source_ref"), "source_ref"),
                        "class-name": _normalize_class_name(row.get("label"), field_name="label"),
                    }
                )
            return

        if task == "multi-label":
            fieldnames = ["source-ref", "labels"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                labels = _normalize_label_list(row.get("labels"), allow_background=False, field_name="labels")
                writer.writerow(
                    {
                        "source-ref": _require_nonempty_string(row.get("source_ref"), "source_ref"),
                        "labels": ",".join(labels),
                    }
                )
            return

        if task == "object-detection":
            fieldnames = ["source-ref", "class-name", "top", "left", "height", "width"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                source_ref = _require_nonempty_string(row.get("source_ref"), "source_ref")
                annotations = _normalize_object_detection_annotations(row.get("annotations"))
                for ann in annotations:
                    writer.writerow(
                        {
                            "source-ref": source_ref,
                            "class-name": ann["class_name"],
                            "top": _number_to_csv_cell(ann["top"]),
                            "left": _number_to_csv_cell(ann["left"]),
                            "height": _number_to_csv_cell(ann["height"]),
                            "width": _number_to_csv_cell(ann["width"]),
                        }
                    )
            return

        if task == "semantic-segmentation":
            fieldnames = ["source-ref", "semantic-segmentation-ref", "color_map"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                color_map = _normalize_semantic_color_map(row.get("color_map"))
                ordered_color_map = _ordered_semantic_color_map(color_map)
                writer.writerow(
                    {
                        "source-ref": _require_nonempty_string(row.get("source_ref"), "source_ref"),
                        "semantic-segmentation-ref": _require_nonempty_string(row.get("mask_ref"), "mask_ref"),
                        "color_map": json.dumps(
                            ordered_color_map,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    }
                )
            return

        if task == "instance-segmentation":
            fieldnames = ["source-ref", "worker-response-ref"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "source-ref": _require_nonempty_string(row.get("source_ref"), "source_ref"),
                        "worker-response-ref": _require_nonempty_string(
                            row.get("worker_response_ref"),
                            "worker_response_ref",
                        ),
                    }
                )
            return

        raise ValueError(f"Unsupported task for CSV manifest writing: {task}")

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

def deterministic_sample(items: list, max_items: int | None, seed: int) -> list:
    if max_items is None or len(items) <= max_items:
        return items

    rng = random.Random(seed)
    chosen_indices = sorted(rng.sample(range(len(items)), max_items))
    return [items[i] for i in chosen_indices]

def download_http_file(
    url: str,
    destination: Path,
    *,
    chunk_size: int = 1024 * 1024,
    request_timeout_sec: int = 60,
    max_attempts: int = 5,
    progress_every_sec: float = 5.0,
) -> Path:
    """
    Download a remote file robustly.

    Behavior:
    - downloads to <destination>.part first
    - prints periodic progress
    - retries on timeout/stall/network interruption
    - attempts same-run resume via HTTP Range if possible
    - validates final size against Content-Length when available
    - only returns the final destination path after atomic rename

    Important:
    - stale .part files from prior runs are deleted at the start, so each run is isolated
    - this does NOT support cross-run resume by design
    """
    ensure_dir(destination.parent)

    temp_path = destination.with_name(destination.name + ".part")

    if temp_path.exists():
        print(f"[download] removing stale partial file: {temp_path}")
        temp_path.unlink()

    remote_size = _get_remote_content_length(url, timeout=request_timeout_sec)

    if destination.exists():
        local_size = destination.stat().st_size
        if remote_size is not None and local_size == remote_size:
            print(
                f"[download] reusing existing complete file: {destination} "
                f"({format_bytes(local_size)})"
            )
            return destination

        print(
            f"[download] existing destination does not look complete; removing and redownloading: "
            f"{destination}"
        )
        destination.unlink()

    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            start_offset = temp_path.stat().st_size if temp_path.exists() else 0
            use_range = start_offset > 0

            print(
                f"[download] attempt {attempt}/{max_attempts} -> {destination.name}"
                + (
                    f" (resuming at {format_bytes(start_offset)})"
                    if use_range
                    else ""
                )
            )

            request = _build_download_request(url, start_offset if use_range else None)

            with urllib.request.urlopen(request, timeout=request_timeout_sec) as response:
                status = getattr(response, "status", response.getcode())

                if use_range and status != 206:
                    print("[download] server did not honor Range request; restarting this attempt from zero")
                    if temp_path.exists():
                        temp_path.unlink()
                    start_offset = 0
                else:
                    _stream_response_to_temp(
                        response=response,
                        temp_path=temp_path,
                        mode="ab" if start_offset > 0 else "wb",
                        chunk_size=chunk_size,
                        progress_every_sec=progress_every_sec,
                        remote_size=remote_size,
                        start_offset=start_offset,
                    )
                    os.replace(temp_path, destination)
                    print(f"[download] completed: {destination} ({format_bytes(destination.stat().st_size)})")
                    return destination

            # Only reached when we tried Range resume and server ignored it.
            request = _build_download_request(url, None)

            with urllib.request.urlopen(request, timeout=request_timeout_sec) as response:
                _stream_response_to_temp(
                    response=response,
                    temp_path=temp_path,
                    mode="wb",
                    chunk_size=chunk_size,
                    progress_every_sec=progress_every_sec,
                    remote_size=remote_size,
                    start_offset=0,
                )

            os.replace(temp_path, destination)
            print(f"[download] completed: {destination} ({format_bytes(destination.stat().st_size)})")
            return destination

        except (urllib.error.URLError, socket.timeout, TimeoutError, OSError, IOError) as exc:
            last_error = exc
            print(f"[download] attempt {attempt}/{max_attempts} failed: {exc}")

            if attempt == max_attempts:
                break

            sleep_sec = min(2 ** attempt, 15)
            print(f"[download] waiting {sleep_sec}s before retry...")
            time.sleep(sleep_sec)

    if temp_path.exists():
        temp_path.unlink()

    raise RuntimeError(f"Failed downloading {url} after {max_attempts} attempts: {last_error}")

def _build_download_request(url: str, start_offset: int | None) -> urllib.request.Request:
    headers = {
        "User-Agent": "cvdms-dataset-bootstrap/1.0",
    }
    if start_offset is not None and start_offset > 0:
        headers["Range"] = f"bytes={start_offset}-"
    return urllib.request.Request(url, headers=headers)

def _stream_response_to_temp(
    *,
    response: Any,
    temp_path: Path,
    mode: str,
    chunk_size: int,
    progress_every_sec: float,
    remote_size: int | None,
    start_offset: int,
) -> None:
    total_size = remote_size
    if total_size is None:
        content_length = response.headers.get("Content-Length")
        if content_length:
            try:
                content_length_int = int(content_length)
                total_size = content_length_int + start_offset if start_offset > 0 else content_length_int
            except ValueError:
                total_size = None

    last_print_time = time.time()

    with temp_path.open(mode) as out:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break

            out.write(chunk)

            now = time.time()
            if now - last_print_time >= progress_every_sec:
                current_size = temp_path.stat().st_size
                _print_download_progress(current_size, total_size)
                last_print_time = now

    current_size = temp_path.stat().st_size
    _print_download_progress(current_size, total_size)

    if total_size is not None and current_size != total_size:
        raise IOError(
            f"download incomplete: expected {format_bytes(total_size)}, "
            f"got {format_bytes(current_size)}"
        )

def _get_remote_content_length(url: str, *, timeout: int) -> int | None:
    """
    Best-effort HEAD probe for Content-Length.
    Returns None if unavailable.
    """
    headers = {
        "User-Agent": "cvdms-dataset-bootstrap/1.0",
    }

    try:
        request = urllib.request.Request(url, headers=headers, method="HEAD")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_length = response.headers.get("Content-Length")
            if not content_length:
                return None
            return int(content_length)
    except Exception:
        return None

def _print_download_progress(current_size: int, total_size: int | None) -> None:
    if total_size and total_size > 0:
        pct = (current_size / total_size) * 100.0
        print(
            f"[download] {format_bytes(current_size)} / {format_bytes(total_size)} "
            f"({pct:.1f}%)"
        )
    else:
        print(f"[download] {format_bytes(current_size)} downloaded")

def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    unit_idx = 0

    while value >= 1024.0 and unit_idx < len(units) - 1:
        value /= 1024.0
        unit_idx += 1

    return f"{value:.1f} {units[unit_idx]}"

def extract_zip(zip_path: Path, destination_dir: Path) -> Path:
    ensure_dir(destination_dir)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(destination_dir)
    return destination_dir

# -------------------------------------------------------------------
# Internal manifest-row builders
# These create a small normalized internal representation.
# The writers then render that representation into accepted CSV/JSONL.
# -------------------------------------------------------------------
def make_single_label_row(source_ref: str, label: str) -> dict[str, Any]:
    return {
        "source_ref": _require_nonempty_string(source_ref, "source_ref"),
        "label": _normalize_class_name(label, field_name="label"),
    }

def make_multi_label_row(source_ref: str, labels: list[str]) -> dict[str, Any]:
    return {
        "source_ref": _require_nonempty_string(source_ref, "source_ref"),
        "labels": _normalize_label_list(labels, allow_background=False, field_name="labels"),
    }

def make_object_detection_row(
    source_ref: str,
    annotations: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "source_ref": _require_nonempty_string(source_ref, "source_ref"),
        "annotations": _normalize_object_detection_annotations(annotations),
    }

def make_semantic_segmentation_row(
    source_ref: str,
    mask_ref: str,
    color_map: dict[str, list[str]],
) -> dict[str, Any]:
    return {
        "source_ref": _require_nonempty_string(source_ref, "source_ref"),
        "mask_ref": _require_nonempty_string(mask_ref, "mask_ref"),
        "color_map": _normalize_semantic_color_map(color_map),
    }

def make_instance_segmentation_row(
    source_ref: str,
    worker_response_ref: str,
) -> dict[str, Any]:
    return {
        "source_ref": _require_nonempty_string(source_ref, "source_ref"),
        "worker_response_ref": _require_nonempty_string(
            worker_response_ref,
            "worker_response_ref",
        ),
    }

# -------------------------------------------------------------------
# Rendering helpers
# -------------------------------------------------------------------
def _manifest_row_to_minimal_gt_json(task: str, row: dict[str, Any]) -> dict[str, Any]:
    if task == "single-label":
        return {
            "source-ref": _require_nonempty_string(row.get("source_ref"), "source_ref"),
            "single-label-metadata": {
                "class-name": _normalize_class_name(row.get("label"), field_name="label"),
            },
        }

    if task == "multi-label":
        labels = _normalize_label_list(row.get("labels"), allow_background=False, field_name="labels")
        class_map = {str(idx): label for idx, label in enumerate(labels)}
        return {
            "source-ref": _require_nonempty_string(row.get("source_ref"), "source_ref"),
            "multi-label-metadata": {
                "class-map": class_map,
            },
        }

    if task == "object-detection":
        source_ref = _require_nonempty_string(row.get("source_ref"), "source_ref")
        annotations = _normalize_object_detection_annotations(row.get("annotations"))

        class_names = sorted({ann["class_name"] for ann in annotations})
        class_name_to_id = {name: idx for idx, name in enumerate(class_names)}
        class_map = {str(idx): name for idx, name in enumerate(class_names)}

        gt_annotations = []
        for ann in annotations:
            gt_annotations.append(
                {
                    "class_id": class_name_to_id[ann["class_name"]],
                    "top": ann["top"],
                    "left": ann["left"],
                    "height": ann["height"],
                    "width": ann["width"],
                }
            )

        return {
            "source-ref": source_ref,
            "object-detection": {
                "annotations": gt_annotations,
            },
            "object-detection-metadata": {
                "class-map": class_map,
            },
        }

    if task == "semantic-segmentation":
        color_map = _normalize_semantic_color_map(row.get("color_map"))
        ordered = _ordered_semantic_color_map(color_map)

        internal_color_map: dict[str, dict[str, str]] = {}
        idx = 0
        for class_name, hex_list in ordered.items():
            internal_color_map[str(idx)] = {
                "class-name": class_name,
                "hex-color": hex_list[0],
            }
            idx += 1

        return {
            "source-ref": _require_nonempty_string(row.get("source_ref"), "source_ref"),
            "semantic-segmentation-ref": _require_nonempty_string(row.get("mask_ref"), "mask_ref"),
            "semantic-segmentation-ref-metadata": {
                "internal-color-map": internal_color_map,
            },
        }

    if task == "instance-segmentation":
        return {
            "source-ref": _require_nonempty_string(row.get("source_ref"), "source_ref"),
            "instance-segmentation-metadata": {
                "worker-response-ref": _require_nonempty_string(
                    row.get("worker_response_ref"),
                    "worker_response_ref",
                ),
            },
        }

    raise ValueError(f"Unsupported task for JSONL manifest writing: {task}")

# -------------------------------------------------------------------
# Normalization / validation helpers
# -------------------------------------------------------------------
def _require_nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")
    out = value.strip()
    if out == "":
        raise ValueError(f"{field_name} cannot be empty")
    return out

def _normalize_class_name(value: Any, *, field_name: str, allow_background: bool = False) -> str:
    out = _require_nonempty_string(value, field_name).strip().lower()
    if out == "":
        raise ValueError(f"{field_name} cannot be empty after stripping")
    if not allow_background and out in _RESERVED_CLASS_NAMES_LC:
        raise ValueError(f"{field_name} uses reserved class name: {out}")
    return out

def _normalize_label_list(values: Any, *, allow_background: bool, field_name: str) -> list[str]:
    if not isinstance(values, list):
        raise TypeError(f"{field_name} must be a list[str], got {type(values).__name__}")

    normalized = []
    seen = set()
    for idx, value in enumerate(values):
        item = _normalize_class_name(
            value,
            field_name=f"{field_name}[{idx}]",
            allow_background=allow_background,
        )
        if item not in seen:
            seen.add(item)
            normalized.append(item)

    if not normalized:
        raise ValueError(f"{field_name} cannot be empty")

    return sorted(normalized)

def _normalize_numeric(value: Any, *, field_name: str) -> int | float:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be numeric, got bool")

    if isinstance(value, (int, float)):
        num = float(value)
    elif isinstance(value, str):
        raw = value.strip()
        if raw == "":
            raise ValueError(f"{field_name} cannot be an empty string")
        num = float(raw)
    else:
        raise TypeError(f"{field_name} must be numeric or numeric string, got {type(value).__name__}")

    if not math.isfinite(num):
        raise ValueError(f"{field_name} must be finite")

    if num.is_integer():
        return int(num)
    return num

def _normalize_object_detection_annotations(annotations: Any) -> list[dict[str, Any]]:
    if not isinstance(annotations, list):
        raise TypeError(f"annotations must be a list, got {type(annotations).__name__}")
    if not annotations:
        raise ValueError("annotations cannot be empty")

    normalized = []
    for idx, ann in enumerate(annotations):
        if not isinstance(ann, dict):
            raise TypeError(f"annotations[{idx}] must be a dict, got {type(ann).__name__}")

        class_name = _normalize_class_name(
            ann.get("class_name"),
            field_name=f"annotations[{idx}].class_name",
            allow_background=False,
        )
        top = _normalize_numeric(ann.get("top"), field_name=f"annotations[{idx}].top")
        left = _normalize_numeric(ann.get("left"), field_name=f"annotations[{idx}].left")
        height = _normalize_numeric(ann.get("height"), field_name=f"annotations[{idx}].height")
        width = _normalize_numeric(ann.get("width"), field_name=f"annotations[{idx}].width")

        if top < 0:
            raise ValueError(f"annotations[{idx}].top must be non-negative")
        if left < 0:
            raise ValueError(f"annotations[{idx}].left must be non-negative")
        if height <= 0:
            raise ValueError(f"annotations[{idx}].height must be > 0")
        if width <= 0:
            raise ValueError(f"annotations[{idx}].width must be > 0")

        normalized.append(
            {
                "class_name": class_name,
                "top": top,
                "left": left,
                "height": height,
                "width": width,
            }
        )

    return normalized

def _normalize_semantic_color_map(color_map: Any) -> dict[str, list[str]]:
    if not isinstance(color_map, dict):
        raise TypeError(f"color_map must be a dict[str, list[str]], got {type(color_map).__name__}")
    if not color_map:
        raise ValueError("color_map cannot be empty")

    normalized: dict[str, list[str]] = {}
    saw_background = False

    for raw_class_name, raw_colors in color_map.items():
        class_name = _normalize_class_name(
            raw_class_name,
            field_name="color_map class name",
            allow_background=True,
        )

        if class_name in _BACKGROUND_NAMES_LC:
            class_name = "background"
            if saw_background:
                raise ValueError("color_map must contain exactly one background class, not multiple")
            saw_background = True

        if class_name in normalized:
            raise ValueError(f"Duplicate class name after normalization in color_map: {class_name}")

        if not isinstance(raw_colors, list):
            raise TypeError(f"color_map['{raw_class_name}'] must be a list[str]")

        if len(raw_colors) != 1:
            raise ValueError(
                f"color_map['{raw_class_name}'] must contain exactly one hex color for semantic segmentation"
            )

        raw_hex = raw_colors[0]
        hex_color = _normalize_hex_color(raw_hex, field_name=f"color_map['{raw_class_name}'][0]")

        normalized[class_name] = [hex_color]

    if not saw_background:
        raise ValueError("color_map must contain exactly one 'bg' or 'background' class")

    non_background = [k for k in normalized.keys() if k != "background"]
    if not non_background:
        raise ValueError("color_map must contain at least one non-background class")

    return normalized

def _normalize_hex_color(value: Any, *, field_name: str) -> str:
    s = _require_nonempty_string(value, field_name)
    if len(s) != 7 or not s.startswith("#"):
        raise ValueError(f"{field_name} must be exactly '#RRGGBB'")
    hex_part = s[1:]
    if any(ch not in "0123456789abcdefABCDEF" for ch in hex_part):
        raise ValueError(f"{field_name} contains invalid hex digits: {s}")
    return "#" + hex_part.lower()

def _ordered_semantic_color_map(color_map: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Stable output order:
    - background first
    - then all other classes alphabetically
    """
    out: dict[str, list[str]] = {}
    if "background" in color_map:
        out["background"] = color_map["background"]

    for class_name in sorted(k for k in color_map.keys() if k != "background"):
        out[class_name] = color_map[class_name]

    return out

def _number_to_csv_cell(value: int | float | str) -> str:
    try:
        d = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError(f"bad numeric: {value}")

    if not d.is_finite():
        raise ValueError("non-finite number")

    d = d.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    if d == 0:
        d = Decimal("0.0000")

    s = format(d, "f")
    if "." not in s:
        s += ".0000"
    else:
        whole, frac = s.split(".", 1)
        s = whole + "." + (frac + "0000")[:4]

    return s

def normalize_class_name_for_bootstrap(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"class name must be a string, got {type(value).__name__}")
    out = value.strip().lower()
    if out == "":
        raise ValueError("class name cannot be empty after stripping")
    return out