'''
Enforces 'v1' formatted manifest uploads for the upload workflow that the validation batch job expects.
Class names are always lowercases with no trailing or leading spaces.

Shared fields:
    "schema": _V1_SCHEMA,
    "label_type": <string label type, e.g. object-detection>,
    "source_ref": <S3 URI of image>

single-label:
    "labels": <single-element list of class name, e.g. ['cat']>

multi-label:
    "labels": <one or many-element list of class name(s), e.g. ['cat'] or ['cat', 'feline'], always unique and alphabetically sorted a-z>

object-detection:
    "labels"."boxes" = [ {
                         "class_name": <string lowercase class name>,
                         "top": <float>,
                         "left": <float>,
                         "height": <float>,
                         "width": <float>
                         },
                        ...]

semantic-segmentation:
    "mask_ref": <S3 URI of PNG mask image>
    "color_map": {<string class name>: <single-element list of hex color in the mask>, ...}

instance-segmentation:
    "worker_response_ref": <valid S3 URI pointing to a .json file containing the PNG mask data>

CSV expected structures:
------------------------
For object-detection CSV input, rows must be grouped by source-ref (all boxes for an image contiguous).

single-label: "source-ref", "class-name"
multi-label: "source-ref", "labels"
object-detection: "source-ref", "class-name", "top", "left", "height", "width"
semantic-segmentation: "source-ref", "semantic-segmentation-ref", "color_map"
instance-segmentation: "source-ref", "worker-response-ref"
'''

import os
import re
import csv
import json

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

from workers.common.python.common.general_utils.class_normalizer import canonicalize_class_name

_V1_SCHEMA = "cvdms.manifest.v1"
_ALLOWED_EXTS_LC = {".jsonl", ".ndjson", ".manifest", ".csv"}
_ALLOWED_IMAGE_EXTS_LC = {".jpg", ".jpeg", ".png"}
_RESERVED_CLASS_NAMES_LC = {"bg", "background"}

_INT_STR_RE = re.compile(r"^[+-]?\d+$")
_INT_DOT_ZERO_STR_RE = re.compile(r"^[+-]?\d+\.0+$")
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")  # <=128 chars, starts w/ alnum

def validate_manifest(manifest_path: str, label_type: str) -> Dict[str, Any]:
    """
    Validates and filters manifest in O(1) memory and writes a normalized v1 JSONL.

    Inputs allowed:
      - Ground Truth style JSONL/NDJSON/.manifest with EXACT keys as specified
      - CSV (function converts this to GT-style JSONL, then normalized)

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
    in_extension_lc = os.path.splitext(manifest_path)[1].lower()
    if in_extension_lc not in _ALLOWED_EXTS_LC:
        return {
            "success": False,
            "error": f"Invalid extension '{in_extension_lc}'. Allowed: {sorted(_ALLOWED_EXTS_LC)}",
            "local_path": "",
        }

    # If CSV, convert to a GT-shaped JSONL first (exact GT keys), then normalize to v1 below.
    if in_extension_lc == ".csv":
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
    Input columns:
        single-label: source-ref, class-name
        multi-label: source-ref, labels (string, comma separated list,  e.g. "person,inanimate")
        object-detection: source-ref, class-name, top, left, height, width
        semantic-segmentation: source-ref, semantic-segmentation-ref, color_map (dict of class name to one-element list of hex color)
        instance-segmentation: source-ref, worker-response-ref

    Output: Converts CSV into GT-shaped JSONL using EXACT keys we expect later:
      - source-ref (all label types)
      - single-label-metadata.class-name for single label task
      - multi-label-metadata.class-map for multi-label task (any present values are labels, so map must not be dataset wide; scoped to image only)
      - object-detection.annotations + object-detection-metadata.class-map for bboxes
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
    if label_type == "single-label":
        _csv_to_gt_single_label(p, tmp_path)
    elif label_type == "multi-label":
        _csv_to_gt_multi_label(p, tmp_path)
    elif label_type == "object-detection":
        _csv_to_gt_object_detection(p, tmp_path)
    elif label_type == "semantic-segmentation":
        _csv_to_gt_semantic_seg(p, tmp_path)
    elif label_type == "instance-segmentation":
        _csv_to_gt_instance_seg(p, tmp_path)
    else:
        raise ValueError(f"Unsupported label_type '{label_type}'")

    os.replace(tmp_path, out_path)
    return str(out_path)

def _csv_to_gt_single_label(csv_file: Path, out_tmp: Path) -> None:
    with csv_file.open("r", newline="", encoding="utf-8-sig") as fin, out_tmp.open("w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        _require_columns(reader, {"source-ref", "class-name"})

        for i, row in enumerate(reader, start=2):
            src = (row.get("source-ref") or "").strip()
            cn_normed = canonicalize_class_name(row.get("class-name"), field_name="class-name", allow_background=False)

            if not src:
                raise ValueError(f"CSV line {i}: class name and source-ref cannot be empty")

            if not _is_valid_s3_uri(src):
                raise ValueError(f"CSV line {i}: source-ref must be a valid S3 URI")

            obj = {
                "source-ref": src,
                "single-label-metadata": {"class-name": cn_normed},
            }
            fout.write(json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n")

def _csv_to_gt_multi_label(csv_file: Path, out_tmp: Path) -> None:
    with csv_file.open("r", newline="", encoding="utf-8-sig") as fin, out_tmp.open("w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        _require_columns(reader, {"source-ref", "labels"})

        for i, row in enumerate(reader, start=2):
            src = (row.get("source-ref") or "").strip()
            labels_raw = (row.get("labels") or "").strip().lower()

            if not labels_raw or not src:
                raise ValueError(f"CSV line {i}: labels and source-ref field cannot be empty")

            if not _is_valid_s3_uri(src):
                raise ValueError(f"CSV line {i}: source-ref must be a valid S3 URI")

            labels_normed = [canonicalize_class_name(x, field_name="class-name", allow_background=False) for x in labels_raw.split(",") if x.strip()]

            if not labels_normed:
                raise ValueError(f"CSV line {i}: labels field cannot be empty")

            # stable ids: sort labels for determinism
            labels_normed = sorted(set(labels_normed))
            class_map = {str(idx): name for idx, name in enumerate(labels_normed)}

            obj = {
                "source-ref": src,
                "multi-label-metadata": {"class-map": class_map},
            }
            fout.write(json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n")

def _csv_to_gt_object_detection(csv_file: Path, out_tmp: Path) -> None:
    seen_srcs = set()
    with csv_file.open("r", newline="", encoding="utf-8-sig") as fin, out_tmp.open("w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        _require_columns(reader, {"source-ref", "class-name", "top", "left", "height", "width"})

        cur_src: Optional[str] = None
        cur_anns: List[Dict[str, Any]] = []

        def flush(src_nm: str, anns: List[Dict[str, Any]]) -> None:
            if not anns:
                return
            class_names = sorted({a["class_name"] for a in anns})
            class_name_to_int_id = {name: idx for idx, name in enumerate(class_names)}
            class_str_id_to_name = {str(idx): name for name, idx in class_name_to_int_id.items()}

            gt_anns = [{
                "class_id": class_name_to_int_id[a["class_name"]],
                "top": a["top"],
                "left": a["left"],
                "height": a["height"],
                "width": a["width"],
            } for a in anns]

            obj = {
                "source-ref": src_nm,
                "object-detection": {"annotations": gt_anns},
                "object-detection-metadata": {"class-map": class_str_id_to_name},
            }
            fout.write(json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n")

        for i, row in enumerate(reader, start=2):
            src = (row.get("source-ref") or "").strip()
            cn_normed = canonicalize_class_name(row.get("class-name"), field_name="class-name", allow_background=False)

            if not src:
                raise ValueError(f"CSV line {i}: object-detection class-name and source-ref cannot be empty")
            if not _is_valid_s3_uri(src):
                raise ValueError(f"CSV line {i}: source-ref must be a valid S3 URI")

            top = _parse_number_like(row.get("top"), i, "top")
            left = _parse_number_like(row.get("left"), i, "left")
            height = _parse_number_like(row.get("height"), i, "height")
            width = _parse_number_like(row.get("width"), i, "width")

            if top < 0 or left < 0:
                raise ValueError(f"CSV line {i}: top, left cannot be negative, got top={top}, left={left}")
            if height <= 0 or width <= 0:
                raise ValueError(f"CSV line {i}: height, width must be > 0, got height={height}, width={width}")

            if cur_src is None:
                cur_src = src

            # If source changes, flush previous image
            if src != cur_src:
                flush(cur_src, cur_anns)
                seen_srcs.add(cur_src)
                cur_src = src
                cur_anns = []

                if cur_src in seen_srcs:
                    raise ValueError(f"CSV line {i}: source-refs are not grouped properly: current source-ref {cur_src} has been seen earlier in the CSV. Group by source-ref and retry.")

            cur_anns.append({
                "class_name": cn_normed,
                "top": top,
                "left": left,
                "height": height,
                "width": width,
            })

        # final flush
        if cur_src is not None:
            flush(cur_src, cur_anns)
            seen_srcs.add(cur_src)

def _csv_to_gt_semantic_seg(csv_file: Path, out_tmp: Path) -> None:
    with csv_file.open("r", newline="", encoding="utf-8-sig") as fin, out_tmp.open("w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        _require_columns(reader, {"source-ref", "semantic-segmentation-ref", "color_map"})

        for i, row in enumerate(reader, start=2):
            src = (row.get("source-ref") or "").strip()
            mask = (row.get("semantic-segmentation-ref") or "").strip()
            cm_raw = (row.get("color_map") or "").strip()

            if not cm_raw or not mask or not src:
                raise ValueError(
                    f"CSV line {i}: color_map, source-ref, and semantic-segmentation-ref are required "
                    f"for semantic-segmentation"
                )

            if not _is_valid_s3_uri(src) or not _is_valid_s3_uri(mask):
                raise ValueError(
                    f"CSV line {i}: source-ref and semantic-segmentation-ref must be valid S3 URIs."
                )

            try:
                cm = _parse_json_object_line(line=cm_raw)
            except Exception as e:
                raise ValueError(f"CSV line {i}: color_map is not valid JSON: {e}")

            if cm is None or not cm:
                raise ValueError(f"CSV line {i}: color_map must be a non-empty JSON object")

            normalized_keys = [
                canonicalize_class_name(k, field_name="class-name", allow_background=True)
                for k in cm.keys()
            ]

            present_reserved_keys = [k for k in normalized_keys if k in _RESERVED_CLASS_NAMES_LC]
            present_bg_keys = [k for k in normalized_keys if k in {"bg", "background"}]

            if len(present_bg_keys) != 1:
                raise ValueError(
                    f"CSV line {i}: semantic color_map must include exactly one of 'bg' or 'background' class"
                )

            reserved_present_classes = set(present_reserved_keys) - set(present_bg_keys)
            if reserved_present_classes:
                raise ValueError(
                    f"CSV line {i}: semantic color_map contains reserved classes: {reserved_present_classes}"
                )

            icm: Dict[str, Dict[str, Any]] = {}
            seen_class_names = set()
            seen_hex_colors = set()

            k = 0
            for class_name, colors_list in cm.items():
                class_name_normed = canonicalize_class_name(
                    class_name,
                    field_name="class-name",
                    allow_background=True,
                )

                if class_name_normed in seen_class_names:
                    raise ValueError(
                        f"CSV line {i}: semantic color_map repeats class after normalization: {class_name_normed}"
                    )
                seen_class_names.add(class_name_normed)

                if not isinstance(colors_list, list):
                    raise ValueError(f"CSV line {i}: color_map[{class_name}] must be of type list")

                if len(colors_list) != 1:
                    raise ValueError(
                        f"CSV line {i}: semantic color_map[{class_name}] must have exactly 1 color "
                        f"(list of length 1)"
                    )

                color = colors_list[0]
                _require_hex_rrggbb(color, f"CSV line {i}: hex color: {color}")

                color_lc = color.lower()
                if color_lc in seen_hex_colors:
                    raise ValueError(
                        f"CSV line {i}: semantic color_map repeats hex color across classes: {color}"
                    )
                seen_hex_colors.add(color_lc)

                icm[str(k)] = {
                    "class-name": class_name_normed,
                    "hex-color": color_lc,
                }
                k += 1

            obj = {
                "source-ref": src,
                "semantic-segmentation-ref": mask,
                "semantic-segmentation-ref-metadata": {"internal-color-map": icm},
            }
            fout.write(json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n")

def _csv_to_gt_instance_seg(csv_file: Path, out_tmp: Path) -> None:
    with csv_file.open("r", newline="", encoding="utf-8-sig") as fin, out_tmp.open("w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        _require_columns(reader, {"source-ref", "worker-response-ref"})

        for i, row in enumerate(reader, start=2):
            src = (row.get("source-ref") or "").strip()
            wrr = (row.get("worker-response-ref") or "").strip()

            if not src or not wrr:
                raise ValueError(f"CSV line {i}: source-ref and worker-response-ref are required.")

            if not _is_valid_s3_uri(src) or not _is_valid_s3_uri(wrr):
                raise ValueError(f"CSV line {i}: source-ref and worker-response-ref must be valid S3 URIs.")

            obj = {
                "source-ref": src,
                "instance-segmentation-metadata": {"worker-response-ref": wrr},
            }
            fout.write(json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n")

# --------------------------------
# GT row -> v1 row
# --------------------------------
def _gt_row_to_v1(*, obj: Dict[str, Any], label_type: str, lineno: int) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Converts Ground Truth style jsonl row to my standardized, minimalized, v1 format. Permits skipping if line indicates a skipped image.

    Returns: (skip, v1_obj_or_none, error_string_if_any)
    """
    src = obj.get("source-ref")
    v1_base = {"schema": _V1_SCHEMA, "label_type": label_type, "source_ref": src}

    if label_type == "single-label":
        meta = obj.get("single-label-metadata")
        if not isinstance(meta, dict):
            return False, None, f"Line {lineno}: missing or invalid 'single-label-metadata' (must be a dict)."

        if not meta.get("class-name", "").strip().lower():
            return True, None, ""  # skip empty

        try:
            cn_normed = canonicalize_class_name(meta.get("class-name", ""), field_name="class-name", allow_background=False)
        except Exception as e:
            return False, None, f"Line {lineno}: 'single-label-metadata.class-name' is a reserved class name or invalid: {str(e)}"

        v1 = dict(v1_base)
        v1["labels"] = [cn_normed]
        return False, v1, ""

    if label_type == "multi-label":
        meta = obj.get("multi-label-metadata")
        if not isinstance(meta, dict):
            return False, None, f"Line {lineno}: missing or invalid 'multi-label-metadata' (must be a dict)."

        class_map = meta.get("class-map")
        if not isinstance(class_map, dict):
            return False, None, f"Line {lineno}: 'multi-label-metadata.class-map' must be a dict."

        # Skip when no class-map present for this image
        if len(class_map) == 0:
            return True, None, ""  # skip

        # Validate class-map values strings
        labels: List[str] = []
        for k, v in class_map.items():
            try:
                cn_normed = canonicalize_class_name(v, field_name="class-name", allow_background=False)
            except Exception as e:
                return False, None, f"Line {lineno}: 'multi-label-metadata.class-map' error: {str(e)}"

            labels.append(cn_normed)

        # Normalize: dedup + sort for determinism
        labels = sorted(list(set(labels)))
        v1 = dict(v1_base)
        v1["labels"] = labels
        return False, v1, ""

    if label_type == "object-detection":
        od = obj.get("object-detection")
        if not isinstance(od, dict):
            return False, None, f"Line {lineno}: missing or invalid 'object-detection' (must be a dict)."

        annotations = od.get("annotations")
        if not isinstance(annotations, list):
            return False, None, f"Line {lineno}: 'object-detection.annotations' must be a list."

        if len(annotations) == 0:
            return True, None, ""  # skip empty

        meta = obj.get("object-detection-metadata")
        if not isinstance(meta, dict):
            return False, None, f"Line {lineno}: missing or invalid 'object-detection-metadata' (must be a dict)."

        class_map = meta.get("class-map") # str class_id to str class name
        if not isinstance(class_map, dict):
            return False, None, f"Line {lineno}: 'object-detection-metadata.class-map' must be a dict."

        if len(class_map) == 0:
            return False, None, f"Line {lineno}: 'object-detection-metadata.class-map' must be a non-empty dict."

        for k, v in class_map.items():
            if not isinstance(v, str) or not v.strip() or (v.strip().lower() in _RESERVED_CLASS_NAMES_LC):
                return False, None, f"Line {lineno}: object-detection class-map values must be non-empty strings and not one of {_RESERVED_CLASS_NAMES_LC}."

            try:
                _parse_int_like(k, lineno, f"object-detection-metadata.class-map key {k}")
            except Exception as e:
                return False, None, f"Line {lineno}: object-detection class-map keys must be integer-like: {e}"

        boxes = []
        for i, ann in enumerate(annotations):
            if not isinstance(ann, dict):
                return False, None, f"Line {lineno}: annotation[{i}] must be a dict."

            for field in ("top", "left", "height", "width"):
                if field not in ann:
                    return False, None, f"Line {lineno}: annotation[{i}] missing required field '{field}'."

                try:
                    v = _parse_number_like(ann[field], lineno, f"annotation[{i}].{field}") # returns finite float or raises
                except ValueError as e:
                    return False, None, f"Line {lineno}: annotation[{i}] required field '{field}' is not float or int-like: {e}"

                if field in ["top", "left"] and v < 0:
                    return False, None, f"Line {lineno}: annotation[{i}].{field} must be non-negative."

                if field in ["height", "width"] and v <= 0:
                    return False, None, f"Line {lineno}: annotation[{i}].{field} must be a positive number."

                ann[field] = v

            if "class_id" not in ann:
                return False, None, f"Line {lineno}: annotation[{i}] missing required field 'class_id'."

            try:
                cid = _parse_int_like(ann["class_id"], lineno, f"annotation[{i}].class_id") # returns int or raises
            except Exception as e:
                return False, None, f"Line {lineno}: annotation[{i}].class_id must be integer-like: {e}"

            cid_key = str(cid)
            if cid_key not in class_map:
                return False, None, f"Line {lineno}: annotation[{i}] class_id {cid} missing from object-detection-metadata.class-map."

            try:
                cn_normed = canonicalize_class_name(class_map[cid_key], field_name=f"class_map[{cid_key}]", allow_background=False)
            except Exception as e:
                return False, None, f"Line {lineno}: class_map[cid_key] must be a valid class: {str(e)}"

            boxes.append({
                "class_name": cn_normed,
                "top": ann["top"], # floats
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

        ext = _s3_key_ext(mask_ref) # lowercase extension
        if ext != ".png":
            return False, None, f"Line {lineno}: 'semantic-segmentation-ref' must end with .png/.PNG (got '{ext or '<no extension>'}')."

        meta = obj.get("semantic-segmentation-ref-metadata")
        if not isinstance(meta, dict):
            return False, None, f"Line {lineno}: missing or invalid 'semantic-segmentation-ref-metadata' (must be a dict)."

        icm = meta.get("internal-color-map")
        if not isinstance(icm, dict) or len(icm) == 0:
            return False, None, f"Line {lineno}: 'semantic-segmentation-ref-metadata.internal-color-map' must be a non-empty dict"

        # Build v1 color_map: class_name -> [hex] (max 1 for semantic seg)
        color_map: Dict[str, List[str]] = {}
        saw_bg = False
        seen_cn = set()
        reserved_non_bg_classes = set(_RESERVED_CLASS_NAMES_LC) - {'bg', 'background'}
        for k, v in icm.items():
            if not isinstance(v, dict):
                return False, None, f"Line {lineno}: internal-color-map['{k}'] must be an object."

            try:
                cn_normed = canonicalize_class_name(v.get("class-name", ""), field_name="class-name", allow_background=True)
            except Exception as e:
                return False, None, f"Line {lineno}:class-name must be a valid class name: {str(e)}."

            if cn_normed in seen_cn:
                return False, None, f"Line {lineno}: semantic internal-color-map repeats class-name {cn_normed}"

            hc = v.get("hex-color", "")

            try:
                _require_hex_rrggbb(hc, f"Line {lineno}: internal-color-map['{k}'].hex-color")
            except Exception as e:
                return False, None, f"Line {lineno}: internal-color-map['{k}'].hex-color must be a valid hex color: {e}"

            if cn_normed in reserved_non_bg_classes:
                return False, None, f"Line {lineno}: semantic internal-color-map contains a reserved non-background class: {cn_normed}"

            if cn_normed in ['bg', 'background']:
                if saw_bg:
                    return False, None, f"Line {lineno}: semantic internal-color-map must include exactly one of 'bg' or 'background', not both"
                else:
                    saw_bg = True

                seen_cn.add(cn_normed)
                continue

            seen_cn.add(cn_normed)
            color_map[cn_normed] = [hc]

        if not saw_bg:
            return False, None, f"Line {lineno}: semantic internal-color-map must include 'bg' or 'background', case insensitive"

        if len(color_map) == 0:
            return False, None, f"Line {lineno}: semantic color_map has no non-background classes."

        v1 = dict(v1_base)
        v1["mask_ref"] = mask_ref
        v1["color_map"] = color_map
        return False, v1, ""

    if label_type == "instance-segmentation":
        meta = obj.get("instance-segmentation-metadata")

        if not isinstance(meta, dict):
            return False, None, f"Line {lineno}: missing or invalid 'instance-segmentation-metadata' (must be a dict)."

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
    def _bad_const(x: str):
        raise ValueError(f"Invalid JSON constant: {x}")  # NaN/Infinity
    try:
        parsed = json.loads(line, parse_constant=_bad_const)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None

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

def _s3_key_basename(uri: str) -> str:
    key = uri[5:].split("/", 1)[1]
    return key.rsplit("/", 1)[-1]

def _validate_source_ref(*, obj: Dict[str, Any], lineno: int) -> Tuple[bool, str]:
    if "source-ref" not in obj:
        return False, f"Line {lineno}: missing required key 'source-ref'."

    src = obj["source-ref"]
    if not _is_valid_s3_uri(src):
        return False, f"Line {lineno}: 'source-ref' must be a valid s3://bucket/key URI."

    ext = _s3_key_ext(src)
    if ext not in _ALLOWED_IMAGE_EXTS_LC:
        return False, (
            f"Line {lineno}: 'source-ref' must end with one of {sorted(_ALLOWED_IMAGE_EXTS_LC)} "
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

def _parse_number_like(val: Any, lineno: int, context: str) -> float:
    # Accept int/float or numeric strings; reject NaN/Infinity
    if isinstance(val, (int, float)):
        x = float(val)
    elif isinstance(val, str):
        s = val.strip()
        if s == "":
            raise ValueError(f"Line {lineno}: {context} is empty string.")
        try:
            x = float(s)
        except ValueError:
            raise ValueError(f"Line {lineno}: {context} must be numeric (got '{val}').")
    else:
        raise ValueError(f"Line {lineno}: {context} must be a number or numeric string (got {type(val).__name__}).")

    if not math.isfinite(x):
        raise ValueError(f"Line {lineno}: {context} must be finite (not NaN/Infinity).")

    return x

def _parse_int_like(val: Any, lineno: int, context: str) -> int:
    # real ints (but exclude bool)
    if isinstance(val, int) and not isinstance(val, bool):
        return val

    # floats: allow only exact integers (e.g., 1.0), reject non-finite
    if isinstance(val, float):
        if not math.isfinite(val):
            raise ValueError(f"Line {lineno}: {context} must be finite (not NaN/Infinity).")
        if val.is_integer():
            return int(val)
        raise ValueError(f"Line {lineno}: {context} must be integer-like (got {val}).")

    # strings: allow "1" or "1.0" / "1.00" only
    if isinstance(val, str):
        s = val.strip()
        if s == "":
            raise ValueError(f"Line {lineno}: {context} is empty string.")
        if _INT_STR_RE.match(s):
            return int(s)
        if _INT_DOT_ZERO_STR_RE.match(s):
            return int(s.split(".", 1)[0])
        raise ValueError(f"Line {lineno}: {context} must be integer-like (got '{val}').")

    raise ValueError(
        f"Line {lineno}: {context} must be an int, float-int, or integer-like string (got {type(val).__name__})."
    )

def validate_s3_key_prefix(prefix: str) -> str:
    if not isinstance(prefix, str):
        raise TypeError(f"path_prefix must be a string, got {type(prefix).__name__}")

    s = prefix.strip()
    if s == "":
        raise ValueError(f"path_prefix cannot be empty")

    # Disallow control chars (keeps logs/URLs/tools sane)
    if any(ord(ch) < 32 for ch in s):
        raise ValueError(f"path_prefix contains control characters (not allowed): {prefix!r}")

    if s.startswith("/") or s.endswith("/"):
        raise ValueError(f"path_prefix must not start or end with '/': {s!r}")

    if "//" in s:
        raise ValueError(f"path_prefix must not contain empty segments ('//'): {s!r}")

    parts = s.split("/")
    for p in parts:
        if p in (".", ".."):
            raise ValueError(f"path_prefix contains illegal segment {p!r}: {s!r}")
        if not _SEGMENT_RE.fullmatch(p):
            raise ValueError(
                f"path_prefix segment {p!r} is invalid. Allowed characters: "
                f"letters/digits/._- ; must start with letter/digit; max 128 chars per segment."
            )

    return s  # normalized (trimmed)