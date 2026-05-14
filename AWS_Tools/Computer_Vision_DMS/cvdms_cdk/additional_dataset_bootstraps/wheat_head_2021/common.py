import csv
import hashlib
import json
import math
import mimetypes
import os
import random
import shutil
import socket
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BootstrapConfig:
    bucket: str
    s3_prefix: str
    aws_region: str | None
    reuse_from_run_dir: Path | None
    max_items: int | None
    sample_seed: int
    split: str
    output_dir: Path
    work_dir: Path


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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_run_dir_name(split: str, dt: datetime) -> str:
    stamp = dt.strftime("%Y%m%d_%H%M%S")
    return f"wheat_head_2021_object_detection_{stamp}_{split}"


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_failures_json(path: Path, failures: list[BootstrapFailure]) -> None:
    write_json(path, [asdict(item) for item in failures])


def s3_key_join(*parts: str) -> str:
    cleaned = [part.strip("/") for part in parts if part and part.strip("/")]
    return "/".join(cleaned)


def s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def upload_file_to_s3(
    *,
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


def deterministic_sample(items: list[Any], max_items: int | None, seed: int) -> list[Any]:
    if max_items is None or len(items) <= max_items:
        return items

    rng = random.Random(seed)
    chosen_indices = sorted(rng.sample(range(len(items)), max_items))
    return [items[i] for i in chosen_indices]


def canonicalize_class_name(value: Any, *, field_name: str = "class_name") -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")

    cleaned = value.strip().lower().replace("-", "_").replace(" ", "_")
    cleaned = "_".join(part for part in cleaned.split("_") if part)

    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")

    return cleaned


def make_object_detection_row(
    *,
    source_ref: str,
    annotations: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "source_ref": require_nonempty_string(source_ref, "source_ref"),
        "annotations": normalize_object_detection_annotations(annotations),
    }


def write_object_detection_jsonl_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            payload = object_detection_row_to_minimal_gt_json(row)
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def write_object_detection_csv_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["source-ref", "class-name", "top", "left", "height", "width"]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            source_ref = require_nonempty_string(row.get("source_ref"), "source_ref")
            annotations = normalize_object_detection_annotations(row.get("annotations"))

            for ann in annotations:
                writer.writerow(
                    {
                        "source-ref": source_ref,
                        "class-name": canonicalize_class_name(ann["class_name"]),
                        "top": number_to_csv_cell(ann["top"]),
                        "left": number_to_csv_cell(ann["left"]),
                        "height": number_to_csv_cell(ann["height"]),
                        "width": number_to_csv_cell(ann["width"]),
                    }
                )


def object_detection_row_to_minimal_gt_json(row: dict[str, Any]) -> dict[str, Any]:
    source_ref = require_nonempty_string(row.get("source_ref"), "source_ref")
    annotations = normalize_object_detection_annotations(row.get("annotations"))

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


def require_nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")

    out = value.strip()
    if out == "":
        raise ValueError(f"{field_name} cannot be empty")

    return out


def normalize_object_detection_annotations(annotations: Any) -> list[dict[str, Any]]:
    if not isinstance(annotations, list):
        raise TypeError(f"annotations must be a list, got {type(annotations).__name__}")

    if not annotations:
        raise ValueError("annotations cannot be empty")

    normalized = []
    for idx, ann in enumerate(annotations):
        if not isinstance(ann, dict):
            raise TypeError(f"annotations[{idx}] must be a dict, got {type(ann).__name__}")

        class_name = canonicalize_class_name(
            ann.get("class_name"),
            field_name=f"annotations[{idx}].class_name",
        )
        top = normalize_numeric(ann.get("top"), field_name=f"annotations[{idx}].top")
        left = normalize_numeric(ann.get("left"), field_name=f"annotations[{idx}].left")
        height = normalize_numeric(ann.get("height"), field_name=f"annotations[{idx}].height")
        width = normalize_numeric(ann.get("width"), field_name=f"annotations[{idx}].width")

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


def normalize_numeric(value: Any, *, field_name: str) -> int | float:
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


def number_to_csv_cell(value: int | float | str) -> str:
    try:
        d = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"bad numeric: {value}") from exc

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


def download_http_file(
    url: str,
    destination: Path,
    *,
    expected_md5: str | None = None,
    chunk_size: int = 1024 * 1024,
    request_timeout_sec: int = 60,
    max_attempts: int = 5,
    progress_every_sec: float = 5.0,
) -> Path:
    ensure_dir(destination.parent)

    temp_path = destination.with_name(destination.name + ".part")

    if temp_path.exists():
        print(f"[download] removing stale partial file: {temp_path}")
        temp_path.unlink()

    remote_size = get_remote_content_length(url, timeout=request_timeout_sec)

    if destination.exists():
        local_size = destination.stat().st_size
        size_ok = remote_size is None or local_size == remote_size
        md5_ok = expected_md5 is None or file_md5(destination) == expected_md5

        if size_ok and md5_ok:
            print(
                f"[download] reusing existing complete file: {destination} "
                f"({format_bytes(local_size)})"
            )
            return destination

        print(f"[download] existing destination failed validation; removing: {destination}")
        destination.unlink()

    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            start_offset = temp_path.stat().st_size if temp_path.exists() else 0
            use_range = start_offset > 0

            print(
                f"[download] attempt {attempt}/{max_attempts} -> {destination.name}"
                + (f" (resuming at {format_bytes(start_offset)})" if use_range else "")
            )

            request = build_download_request(url, start_offset if use_range else None)

            with urllib.request.urlopen(request, timeout=request_timeout_sec) as response:
                status = getattr(response, "status", response.getcode())

                if use_range and status != 206:
                    print("[download] server did not honor Range request; restarting from zero")
                    if temp_path.exists():
                        temp_path.unlink()
                    start_offset = 0
                    request = build_download_request(url, None)

                    with urllib.request.urlopen(request, timeout=request_timeout_sec) as retry_response:
                        stream_response_to_temp(
                            response=retry_response,
                            temp_path=temp_path,
                            mode="wb",
                            chunk_size=chunk_size,
                            progress_every_sec=progress_every_sec,
                            remote_size=remote_size,
                            start_offset=0,
                        )
                else:
                    stream_response_to_temp(
                        response=response,
                        temp_path=temp_path,
                        mode="ab" if start_offset > 0 else "wb",
                        chunk_size=chunk_size,
                        progress_every_sec=progress_every_sec,
                        remote_size=remote_size,
                        start_offset=start_offset,
                    )

            os.replace(temp_path, destination)

            if expected_md5 is not None:
                actual_md5 = file_md5(destination)
                if actual_md5 != expected_md5:
                    destination.unlink(missing_ok=True)
                    raise IOError(
                        f"downloaded file md5 mismatch: expected={expected_md5}, actual={actual_md5}"
                    )

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


def build_download_request(url: str, start_offset: int | None) -> urllib.request.Request:
    headers = {
        "User-Agent": "cvdms-wheat-head-2021-bootstrap/1.0",
    }
    if start_offset is not None and start_offset > 0:
        headers["Range"] = f"bytes={start_offset}-"
    return urllib.request.Request(url, headers=headers)


def stream_response_to_temp(
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
                print_download_progress(current_size, total_size)
                last_print_time = now

    current_size = temp_path.stat().st_size
    print_download_progress(current_size, total_size)

    if total_size is not None and current_size != total_size:
        raise IOError(
            f"download incomplete: expected {format_bytes(total_size)}, "
            f"got {format_bytes(current_size)}"
        )


def get_remote_content_length(url: str, *, timeout: int) -> int | None:
    headers = {
        "User-Agent": "cvdms-wheat-head-2021-bootstrap/1.0",
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


def print_download_progress(current_size: int, total_size: int | None) -> None:
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


def file_md5(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def extract_zip(zip_path: Path, destination_dir: Path) -> Path:
    ensure_dir(destination_dir)
    dest_root = destination_dir.resolve()

    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            target = (dest_root / member.filename).resolve()

            try:
                target.relative_to(dest_root)
            except ValueError as exc:
                raise ValueError(f"Unsafe zip member path: {member.filename}") from exc

            if member.is_dir():
                ensure_dir(target)
                continue

            ensure_dir(target.parent)
            with archive.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)

    return destination_dir