import csv
import json
import os

import math
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple, List, DefaultDict

_V1_SCHEMA = "cvdms.manifest.v1"
_ALLOWED_EXTS = {".jsonl", ".ndjson", ".manifest", ".csv"}
_ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
_RESERVED_CLASS_NAMES_LC = {"bg", "background"}

def validate_manifest(manifest_path: str, label_type: str) -> Dict[str, Any]:
    """
    Validates and filters manifest in O(1) memory and writes a normalized v1 JSONL.

    Inputs allowed:
      - Ground Truth style JSONL/NDJSON/.manifest with EXACT keys as specified
      - CSV (converted to GT-style JSONL first, then normalized)

    Output:
      - Always writes a v1 JSONL manifest (minimal keys) to *.formatted.jsonl
      - Applies skip rules only for label types 1-3 (single/multi/object-detection)

    Returns:
      {
        "success": bool,
        "error": str,
        "local_path": str,
        "skipped_count": int,
        "kept_count": int,
        "total_nonempty": int,
      }
    """
    in_extension = os.path.splitext(manifest_path)[1].lower()
    if in_extension not in _ALLOWED_EXTS:
        return {
            "success": False,
            "error": f"Invalid extension '{in_extension}'. Allowed: {sorted(_ALLOWED_EXTS)}",
            "local_path": "",
        }

    # If CSV, convert to a GT-shaped JSONL first (exact GT keys), then normalize to v1 below.
    if in_extension == ".csv":
        try:
            jsonl_manifest_path = _convert_csv_to_jsonl_input(manifest_path, label_type=label_type)
        except Exception as e:
            return {"success": False, "error": f"CSV conversion failed: {type(e).__name__}: {e}", "local_path": ""}
    else:
        jsonl_manifest_path = manifest_path

    try:
        p = Path(jsonl_manifest_path)
        if not p.exists() or not p.is_file():
            return {"success": False, "error": f"File not found: {jsonl_manifest_path}", "local_path": ""}

        formatted_manifest_path = _make_formatted_manifest_path(p)
        tmp_out = Path(str(formatted_manifest_path) + ".tmp")
        formatted_manifest_path.parent.mkdir(parents=True, exist_ok=True)

        # Stream read + stream write (O(1) memory)
        skipped_count = 0
        kept_count = 0
        total_nonempty = 0
        with p.open("r", encoding="utf-8-sig") as fin, tmp_out.open("w", encoding="utf-8") as fout:
            for lineno, raw in enumerate(fin, start=1):
                line = raw.strip()
                if not line:
                    continue  # ignore blank lines

                total_nonempty += 1

                obj = _parse_json_object_line(line=line)
                if obj is None:
                    _safe_unlink(tmp_out)
                    return {
                        "success": False,
                        "error": f"Line {lineno}: not valid JSON (or not a single JSON object).",
                        "local_path": "",
                    }

                ok, err = _validate_source_ref(obj=obj, lineno=lineno)
                if not ok:
                    _safe_unlink(tmp_out)
                    return {"success": False, "error": err, "local_path": ""}

                skip, v1_obj, err = _gt_row_to_v1(obj=obj, label_type=label_type, lineno=lineno)
                if err:
                    _safe_unlink(tmp_out)
                    return {"success": False, "error": err, "local_path": ""}

                if skip:
                    skipped_count += 1
                    continue

                fout.write(json.dumps(v1_obj, separators=(",", ":"), ensure_ascii=False) + "\n")
                kept_count += 1

        if total_nonempty == 0:
            _safe_unlink(tmp_out)
            return {"success": False, "error": "Manifest has no JSON lines (empty or only blank lines).", "local_path": ""}

        # Publish atomically
        os.replace(tmp_out, formatted_manifest_path)

        return {
            "success": True,
            "error": "",
            "local_path": str(formatted_manifest_path),
            "skipped_count": skipped_count,
            "kept_count": kept_count,
            "total_nonempty": total_nonempty,
        }

    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {type(e).__name__}: {e}", "local_path": ""}

# --------------------------------
# CSV -> GT-shaped JSONL
# --------------------------------
def _convert_csv_to_jsonl_input(csv_path: str, *, label_type: str) -> str:
    """
    Converts CSV into GT-shaped JSONL using EXACT keys we expect later:
      - source-ref
      - single-label-metadata.class-name
      - multi-label + multi-label-metadata.class-map
      - object-detection + object-detection-metadata.class-map
      - semantic-segmentation-ref + semantic-segmentation-ref-metadata.internal-color-map
      - instance-segmentation-metadata.worker-response-ref

    This is strict. If required columns are missing or malformed, it raises.

    Output path: alongside the CSV, with a unique .converted.jsonl suffix.
    """
    p = Path(csv_path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    out_path = _unique_sibling_path(p, suffix=".converted.jsonl")
    tmp_path = Path(str(out_path) + ".tmp")

    # CSV is read fully streaming, but some label types require grouping:
    # - object-detection groups rows by source-ref to build annotations arrays
    # For object-detection we’ll buffer per-image in memory (still usually small).
    if label_type == "object-detection":
        _csv_to_gt_object_detection(p, tmp_path)
    elif label_type == "single-label":
        _csv_to_gt_single_label(p, tmp_path)
    elif label_type == "multi-label":
        _csv_to_gt_multi_label(p, tmp_path)
    elif label_type == "semantic-segmentation":
        _csv_to_gt_semantic_seg(p, tmp_path)
    elif label_type == "instance-segmentation":
        _csv_to_gt_instance_seg(p, tmp_path)
    else:
        raise ValueError(f"Unsupported label_type '{label_type}'")

    os.replace(tmp_path, out_path)
    return str(out_path)

def _csv_to_gt_single_label(csv_file: Path, out_tmp: Path) -> None:
    # Required columns (strict):
    #   source-ref, class-name
    with csv_file.open("r", newline="", encoding="utf-8-sig") as fin, out_tmp.open("w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        _require_columns(reader, {"source-ref", "class-name"})

        for i, row in enumerate(reader, start=2):
            src = (row.get("source-ref") or "").strip()
            cn = (row.get("class-name") or "").strip()

            obj = {
                "source-ref": src,
                "single-label": 0 if cn else -1,  # value ignored later; keep GT-like
                "single-label-metadata": {"class-name": cn},
            }
            fout.write(json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n")

def _csv_to_gt_multi_label(csv_file: Path, out_tmp: Path) -> None:
    # Required columns:
    #   source-ref, labels
    # where labels is a comma-separated list of class strings, e.g. "person,inanimate"
    with csv_file.open("r", newline="", encoding="utf-8-sig") as fin, out_tmp.open("w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        _require_columns(reader, {"source-ref", "labels"})

        for i, row in enumerate(reader, start=2):
            src = (row.get("source-ref") or "").strip()
            labels_raw = (row.get("labels") or "").strip()

            if not labels_raw:
                # emit explicit empty (normalize stage will skip)
                obj = {
                    "source-ref": src,
                    "multi-label": [],
                    "multi-label-metadata": {"class-map": {}},
                }
                fout.write(json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n")
                continue

            labels = [x.strip() for x in labels_raw.split(",") if x.strip()]
            # stable ids: sort labels for determinism
            labels = sorted(set(labels))
            class_map = {str(idx): name for idx, name in enumerate(labels)}
            multi_label_ids = [idx for idx in range(len(labels))]

            obj = {
                "source-ref": src,
                "multi-label": multi_label_ids,
                "multi-label-metadata": {"class-map": class_map},
            }
            fout.write(json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n")

def _csv_to_gt_object_detection(csv_file: Path, out_tmp: Path) -> None:
    # One row per box.
    # Required columns:
    #   source-ref, class-name, top, left, height, width
    groups: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)

    with csv_file.open("r", newline="", encoding="utf-8-sig") as fin:
        reader = csv.DictReader(fin)
        _require_columns(reader, {"source-ref", "class-name", "top", "left", "height", "width"})

        for i, row in enumerate(reader, start=2):
            src = (row.get("source-ref") or "").strip()
            cn = (row.get("class-name") or "").strip()
            if not cn:
                raise ValueError(f"CSV line {i}: object-detection class-name cannot be empty")

            ann = {
                "class_name": cn,
                "top": _parse_number_field(row.get("top"), i, "top"),
                "left": _parse_number_field(row.get("left"), i, "left"),
                "height": _parse_number_field(row.get("height"), i, "height"),
                "width": _parse_number_field(row.get("width"), i, "width"),
            }
            groups[src].append(ann)

    with out_tmp.open("w", encoding="utf-8") as fout:
        for src, anns in groups.items():
            # stable class_id assignment per image
            class_names = sorted({a["class_name"] for a in anns})
            class_id_by_name = {name: idx for idx, name in enumerate(class_names)}
            class_map = {str(idx): name for name, idx in class_id_by_name.items()}

            gt_anns = []
            for a in anns:
                gt_anns.append({
                    "class_id": class_id_by_name[a["class_name"]],
                    "top": a["top"],
                    "left": a["left"],
                    "height": a["height"],
                    "width": a["width"],
                })

            obj = {
                "source-ref": src,
                "object-detection": {"annotations": gt_anns},
                "object-detection-metadata": {"class-map": class_map},
            }
            fout.write(json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n")

def _csv_to_gt_semantic_seg(csv_file: Path, out_tmp: Path) -> None:
    # Required columns:
    #   source-ref, semantic-segmentation-ref, color_map
    # where color_map is JSON mapping class_name -> "#RRGGBB" (or list with one element)
    with csv_file.open("r", newline="", encoding="utf-8-sig") as fin, out_tmp.open("w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        _require_columns(reader, {"source-ref", "semantic-segmentation-ref", "color_map"})

        for i, row in enumerate(reader, start=2):
            src = (row.get("source-ref") or "").strip()
            mask = (row.get("semantic-segmentation-ref") or "").strip()
            cm_raw = (row.get("color_map") or "").strip()
            if not cm_raw:
                raise ValueError(f"CSV line {i}: color_map is required for semantic-segmentation")

            try:
                cm = json.loads(cm_raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"CSV line {i}: color_map is not valid JSON: {e}")

            if not isinstance(cm, dict) or not cm:
                raise ValueError(f"CSV line {i}: color_map must be a non-empty JSON object")

            # build internal-color-map GT shape with numeric keys
            icm: Dict[str, Dict[str, Any]] = {}
            k = 0
            for class_name, colors in cm.items():
                if isinstance(colors, str):
                    colors_list = [colors]
                elif isinstance(colors, list):
                    colors_list = colors
                else:
                    raise ValueError(f"CSV line {i}: color_map[{class_name}] must be string or list")

                if len(colors_list) == 0 or len(colors_list) > 1:
                    raise ValueError(f"CSV line {i}: semantic color_map[{class_name}] must have exactly 1 color")

                color = colors_list[0]
                _require_hex_rrggbb(color, f"CSV line {i}: color_map[{class_name}]")

                icm[str(k)] = {"class-name": str(class_name), "hex-color": color}
                k += 1

            obj = {
                "source-ref": src,
                "semantic-segmentation-ref": mask,
                "semantic-segmentation-ref-metadata": {"internal-color-map": icm},
            }
            fout.write(json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n")

def _csv_to_gt_instance_seg(csv_file: Path, out_tmp: Path) -> None:
    # Required columns:
    #   source-ref, worker-response-ref
    with csv_file.open("r", newline="", encoding="utf-8-sig") as fin, out_tmp.open("w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        _require_columns(reader, {"source-ref", "worker-response-ref"})

        for i, row in enumerate(reader, start=2):
            src = (row.get("source-ref") or "").strip()
            wrr = (row.get("worker-response-ref") or "").strip()

            obj = {
                "source-ref": src,
                "instance-segmentation": {},
                "instance-segmentation-metadata": {"worker-response-ref": wrr},
            }
            fout.write(json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n")

# --------------------------------
# GT row -> v1 row
# --------------------------------
def _gt_row_to_v1(*, obj: Dict[str, Any], label_type: str, lineno: int) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Returns: (skip, v1_obj_or_none, error_string_if_any)
    """
    src = obj.get("source-ref")
    v1_base = {"schema": _V1_SCHEMA, "label_type": label_type, "source_ref": src}

    if label_type == "single-label":
        meta = obj.get("single-label-metadata")
        if not isinstance(meta, dict):
            return False, None, f"Line {lineno}: missing or invalid 'single-label-metadata' (must be an object)."
        cn = meta.get("class-name")
        if not isinstance(cn, str):
            return False, None, f"Line {lineno}: 'single-label-metadata.class-name' must be a string (can be empty)."
        cn = cn.strip()
        if cn == "":
            return True, None, ""  # skip empty

        if cn in _RESERVED_CLASS_NAMES_LC:
            return True, None, ""  # skip empty

        v1 = dict(v1_base)
        v1["labels"] = [cn]
        return False, v1, ""

    if label_type == "multi-label":
        ml = obj.get("multi-label")
        if not isinstance(ml, list):
            return False, None, f"Line {lineno}: missing or invalid 'multi-label' (must be a list)."

        meta = obj.get("multi-label-metadata")
        if not isinstance(meta, dict):
            return False, None, f"Line {lineno}: missing or invalid 'multi-label-metadata' (must be an object)."

        class_map = meta.get("class-map")
        if not isinstance(class_map, dict):
            return False, None, f"Line {lineno}: 'multi-label-metadata.class-map' must be an object/dict."

        # Strict consistency checks
        if len(ml) == 0 and len(class_map) == 0:
            return True, None, ""  # skip

        if len(ml) == 0 and len(class_map) != 0:
            return False, None, f"Line {lineno}: inconsistent multi-label: multi-label=[] but class-map is non-empty."

        if len(ml) != 0 and len(class_map) == 0:
            return False, None, f"Line {lineno}: inconsistent multi-label: multi-label is non-empty but class-map is empty."

        # Validate class-map values strings
        for k, v in class_map.items():
            if not isinstance(v, str) or not v.strip():
                return False, None, f"Line {lineno}: 'multi-label-metadata.class-map' values must be strings and non empty."
            if v.strip().lower() in _RESERVED_CLASS_NAMES_LC:
                return False, None, f"Line {lineno}: 'multi-label-metadata.class-map' contains a reserved class name: {v}"

        # Validate every id in ml exists in class_map (stringified)
        labels: List[str] = []
        for idx, class_id in enumerate(ml):
            if not isinstance(class_id, int):
                return False, None, f"Line {lineno}: multi-label[{idx}] must be an integer class id."
            key = str(class_id)
            if key not in class_map:
                return False, None, f"Line {lineno}: class id {class_id} missing from multi-label-metadata.class-map."
            labels.append(class_map[key].strip().lower())

        # Normalize: dedup + sort for determinism
        labels = sorted(set(labels))
        if len(labels) == 0:
            # should not happen given consistency checks, but keep strict
            return True, None, ""

        v1 = dict(v1_base)
        v1["labels"] = labels
        return False, v1, ""

    if label_type == "object-detection":
        od = obj.get("object-detection")
        if not isinstance(od, dict):
            return False, None, f"Line {lineno}: missing or invalid 'object-detection' (must be an object)."
        annotations = od.get("annotations")
        if not isinstance(annotations, list):
            return False, None, f"Line {lineno}: 'object-detection.annotations' must be a list."
        if len(annotations) == 0:
            return True, None, ""  # skip empty

        meta = obj.get("object-detection-metadata")
        if not isinstance(meta, dict):
            return False, None, f"Line {lineno}: missing or invalid 'object-detection-metadata' (must be an object)."
        class_map = meta.get("class-map")
        if not isinstance(class_map, dict):
            return False, None, f"Line {lineno}: 'object-detection-metadata.class-map' must be an object/dict."
        for _, v in class_map.items():
            if not isinstance(v, str) or v.strip() == "":
                return False, None, f"Line {lineno}: object-detection class-map values must be non-empty strings."

        boxes = []
        for i, ann in enumerate(annotations):
            if not isinstance(ann, dict):
                return False, None, f"Line {lineno}: annotation[{i}] must be an object."

            for field in ("top", "left", "height", "width"):
                if field not in ann:
                    return False, None, f"Line {lineno}: annotation[{i}] missing required field '{field}'."
                if not isinstance(ann[field], (int, float)):
                    return False, None, f"Line {lineno}: annotation[{i}].{field} must be a number."

                if field in ["top", "left"] and ann[field] < 0:
                    return False, None, f"Line {lineno}: annotation[{i}].{field} must be non-negative."

                if field in ["height", "width"] and ann[field] <= 0:
                    return False, None, f"Line {lineno}: annotation[{i}].{field} must be a positive number."


            if not _is_finite_number(ann[field]):
                return False, None, f"Line {lineno}: annotation[{i}].{field} must be a finite number (not NaN/Infinity)."

            if "class_id" not in ann:
                return False, None, f"Line {lineno}: annotation[{i}] missing required field 'class_id'."
            if not isinstance(ann["class_id"], int):
                return False, None, f"Line {lineno}: annotation[{i}].class_id must be an integer."

            cid = ann["class_id"]
            cid_key = str(cid)
            if cid_key not in class_map:
                return False, None, f"Line {lineno}: annotation[{i}] class_id {cid} missing from object-detection-metadata.class-map."

            boxes.append({
                "class_name": class_map[cid_key].strip(),
                "top": ann["top"],
                "left": ann["left"],
                "height": ann["height"],
                "width": ann["width"],
            })

        v1 = dict(v1_base)
        v1["labels"] = {"boxes": boxes}
        return False, v1, ""

    if label_type == "semantic-segmentation":
        mask_ref = obj.get("semantic-segmentation-ref")
        if not _is_valid_s3_uri(mask_ref):
            return False, None, f"Line {lineno}: 'semantic-segmentation-ref' must be a valid s3://bucket/key URI."
        ext = _s3_key_ext(mask_ref)
        if ext != ".png":
            return False, None, f"Line {lineno}: 'semantic-segmentation-ref' must end with .png (got '{ext or '<no extension>'}')."

        meta = obj.get("semantic-segmentation-ref-metadata")
        if not isinstance(meta, dict):
            return False, None, f"Line {lineno}: missing or invalid 'semantic-segmentation-ref-metadata' (must be an object)."
        icm = meta.get("internal-color-map")
        if not isinstance(icm, dict) or len(icm) == 0:
            return False, None, f"Line {lineno}: 'semantic-segmentation-ref-metadata.internal-color-map' must be a non-empty object."

        # Build v1 color_map: class_name -> [hex] (max 1 for semantic seg)
        color_map: Dict[str, List[str]] = {}
        for k, v in icm.items():
            if not isinstance(v, dict):
                return False, None, f"Line {lineno}: internal-color-map['{k}'] must be an object."
            cn = v.get("class-name")
            hc = v.get("hex-color")
            if not isinstance(cn, str) or cn.strip() == "":
                return False, None, f"Line {lineno}: internal-color-map['{k}'].class-name must be a non-empty string."
            if not isinstance(hc, str):
                return False, None, f"Line {lineno}: internal-color-map['{k}'].hex-color must be a string."
            _require_hex_rrggbb(hc, f"Line {lineno}: internal-color-map['{k}'].hex-color")

            cn = cn.strip()
            if cn in color_map:
                # semantic seg should not repeat classes with multiple colors
                return False, None, f"Line {lineno}: semantic internal-color-map repeats class-name '{cn}'."
            color_map[cn] = [hc]

        v1 = dict(v1_base)
        v1["mask_ref"] = mask_ref
        v1["color_map"] = color_map
        return False, v1, ""

    if label_type == "instance-segmentation":
        meta = obj.get("instance-segmentation-metadata")
        if not isinstance(meta, dict):
            return False, None, f"Line {lineno}: missing or invalid 'instance-segmentation-metadata' (must be an object)."
        wrr = meta.get("worker-response-ref")
        if not _is_valid_s3_uri(wrr):
            return False, None, f"Line {lineno}: 'instance-segmentation-metadata.worker-response-ref' must be a valid s3://bucket/key URI."
        ext = _s3_key_ext(wrr)
        if ext != ".json":
            return False, None, f"Line {lineno}: worker-response-ref should end with .json (got '{ext or '<no extension>'}')."

        v1 = dict(v1_base)
        v1["worker_response_ref"] = wrr
        return False, v1, ""

    return False, None, f"Line {lineno}: unsupported label_type '{label_type}'."

# --------------------------------
# Misc Helpers
# --------------------------------
def _make_formatted_manifest_path(original: Path) -> Path:
    base = original.with_suffix("")
    out = Path(str(base) + ".formatted.jsonl")
    if not out.exists():
        return out
    for i in range(1, 10_000):
        candidate = Path(str(base) + f".formatted.{i}.jsonl")
        if not candidate.exists():
            return candidate
    raise RuntimeError("Unable to choose a unique formatted output path.")

def _unique_sibling_path(original: Path, *, suffix: str) -> Path:
    base = original.with_suffix("")
    out = Path(str(base) + suffix)
    if not out.exists():
        return out
    for i in range(1, 10_000):
        candidate = Path(str(base) + f".{i}" + suffix)
        if not candidate.exists():
            return candidate
    raise RuntimeError("Unable to choose a unique converted output path.")

def _parse_json_object_line(*, line: str) -> Optional[Dict[str, Any]]:
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed

def _safe_unlink(p: Path) -> None:
    try:
        p.unlink(missing_ok=True)
    except Exception:
        pass

def _is_valid_s3_uri(uri: Any) -> bool:
    if not isinstance(uri, str):
        return False
    if not uri.startswith("s3://"):
        return False
    rest = uri[5:]
    if "/" not in rest:
        return False
    bucket, key = rest.split("/", 1)
    if not bucket or not key:
        return False
    if uri.strip() != uri:
        return False
    return True

def _s3_key_ext(uri: str) -> str:
    key = uri[5:].split("/", 1)[1]
    return Path(key).suffix.lower()

def _validate_source_ref(*, obj: Dict[str, Any], lineno: int) -> Tuple[bool, str]:
    if "source-ref" not in obj:
        return False, f"Line {lineno}: missing required key 'source-ref'."

    src = obj["source-ref"]
    if not _is_valid_s3_uri(src):
        return False, f"Line {lineno}: 'source-ref' must be a valid s3://bucket/key URI."

    ext = _s3_key_ext(src)
    if ext not in _ALLOWED_IMAGE_EXTS:
        return False, (
            f"Line {lineno}: 'source-ref' must end with one of {sorted(_ALLOWED_IMAGE_EXTS)} "
            f"(got '{ext or '<no extension>'}')."
        )
    return True, ""

def _require_hex_rrggbb(color: str, context: str) -> None:
    if not isinstance(color, str):
        raise ValueError(f"{context}: must be a string")
    if len(color) != 7 or not color.startswith("#"):
        raise ValueError(f"{context}: must be exactly '#RRGGBB' (7 chars)")
    hexpart = color[1:]
    if any(c not in "0123456789abcdefABCDEF" for c in hexpart):
        raise ValueError(f"{context}: invalid hex digits in '{color}'")

def _require_columns(reader: csv.DictReader, required: set) -> None:
    headers = set(reader.fieldnames or [])
    missing = required - headers
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}. Found: {sorted(headers)}")

def _parse_number_field(val: Any, line_no: int, field: str) -> float:
    if val is None:
        raise ValueError(f"CSV line {line_no}: missing '{field}'")
    s = str(val).strip()
    if s == "":
        raise ValueError(f"CSV line {line_no}: empty '{field}'")
    try:
        return float(s)
    except ValueError:
        raise ValueError(f"CSV line {line_no}: '{field}' must be a number (got '{s}')")

def _is_finite_number(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))
