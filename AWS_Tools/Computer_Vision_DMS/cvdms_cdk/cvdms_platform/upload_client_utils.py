import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ALLOWED_EXTS = {".jsonl", ".ndjson", ".manifest"}
_ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

ALLOWED_LABEL_TYPES = {
    "single-label",
    "multi-label",
    "object-detection",
    "semantic-segmentation",
    "instance-segmentation"
}

def validate_manifest(manifest_path: str, label_type: str) -> Dict[str, Any]:
    """
    Validates and filters a Ground Truth–style JSON Lines manifest in O(1) memory.

    Strategy: always stream-write a filtered output file.
      - Read input line-by-line
      - Validate and decide skip
      - Immediately write kept lines to output
      - Track counters
      - If nothing was skipped, return original path (and delete the filtered file)

    Returns:
      {
        "success": bool,
        "error": str,
        "local_path": str,        # original path if no skips; else filtered path
        "skipped_count": int,
        "kept_count": int,
        "total_nonempty": int,
      }

    Filtering rules:
      - single-label: skip if class-name == ""
      - multi-label: skip if multi-label == [] OR class-map == {}
      - object-detection: skip if annotations == []
      - semantic-segmentation: no filtering (only validation)
      - instance-segmentation: no filtering (only validation)
    """
    try:
        p = Path(manifest_path)

        # ---- File exists + extension checks ----
        if not p.exists() or not p.is_file():
            return {"success": False, "error": f"File not found: {manifest_path}", "local_path": ""}

        if p.suffix.lower() not in _ALLOWED_EXTS:
            return {
                "success": False,
                "error": f"Invalid extension '{p.suffix}'. Allowed: {sorted(_ALLOWED_EXTS)}",
                "local_path": "",
            }

        # Always choose an output path up-front. We'll delete it if we end up not needing it.
        filtered_path = _make_filtered_path(p)

        skipped_count = 0
        kept_count = 0
        total_nonempty = 0

        # Stream read + stream write (O(1) memory)
        filtered_path.parent.mkdir(parents=True, exist_ok=True)
        with p.open("r", encoding="utf-8-sig") as fin, filtered_path.open("w", encoding="utf-8") as fout:
            for lineno, raw in enumerate(fin, start=1):
                line = raw.strip()
                if not line:
                    continue  # ignore blank/whitespace-only lines

                total_nonempty += 1

                obj = _parse_json_object_line(line=line)
                if obj is None:
                    # delete partial output to avoid leaving confusing artifacts
                    try:
                        filtered_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    return {
                        "success": False,
                        "error": f"Line {lineno}: not valid JSON (or not a single JSON object).",
                        "local_path": "",
                    }

                # validate required "source-ref"
                ok, err = _validate_source_ref(obj=obj, lineno=lineno)
                if not ok:
                    try:
                        filtered_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    return {"success": False, "error": err, "local_path": ""}

                # validate per label type; may return skip=True for filtering rules
                skip, err = _validate_per_label_type(obj=obj, label_type=label_type, lineno=lineno)
                if err:
                    try:
                        filtered_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    return {"success": False, "error": err, "local_path": ""}

                if skip:
                    skipped_count += 1
                    continue

                # Keep: write immediately.
                # Prefer writing the original (trimmed) JSON line to preserve fidelity and be fast.
                # Ensure newline termination.
                fout.write(line + "\n")
                kept_count += 1

        if total_nonempty == 0:
            try:
                filtered_path.unlink(missing_ok=True)
            except Exception:
                pass
            return {"success": False, "error": "Manifest has no JSON lines (file is empty or only blank lines).", "local_path": ""}

        # If nothing was skipped, we can keep the original file path and remove the filtered output.
        if skipped_count == 0:
            try:
                filtered_path.unlink(missing_ok=True)
            except Exception:
                # Not fatal; worst case we leave a duplicate filtered file behind
                pass
            return {
                "success": True,
                "error": "",
                "local_path": str(p),
                "skipped_count": 0,
                "kept_count": kept_count,
                "total_nonempty": total_nonempty,
            }

        # Otherwise return the filtered path
        return {
            "success": True,
            "error": "",
            "local_path": str(filtered_path),
            "skipped_count": skipped_count,
            "kept_count": kept_count,
            "total_nonempty": total_nonempty,
        }

    except Exception as e:
        # Catch-all so caller always gets a structured result
        return {"success": False, "error": f"Unexpected error: {type(e).__name__}: {e}", "local_path": ""}

# --------------------------------
# Helpers for manifest validation
# --------------------------------
def _parse_json_object_line(*, line: str) -> Optional[Dict[str, Any]]:
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed

def _is_valid_s3_uri(uri: Any) -> bool:
    if not isinstance(uri, str):
        return False
    if not uri.startswith("s3://"):
        return False
    rest = uri[5:]
    # require bucket/key
    if "/" not in rest:
        return False
    bucket, key = rest.split("/", 1)
    if not bucket or not key:
        return False
    # avoid obvious whitespace
    if uri.strip() != uri:
        return False
    return True

def _s3_key_ext(uri: str) -> str:
    # assumes _is_valid_s3_uri already checked
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

def _validate_per_label_type(*, obj: Dict[str, Any], label_type: str, lineno: int) -> Tuple[bool, str]:
    """
    Returns: (skip_this_line, error_string_if_any)
    """
    if label_type == "single-label":
        meta = obj.get("single-label-metadata")
        if not isinstance(meta, dict):
            return False, f"Line {lineno}: missing or invalid 'single-label-metadata' (must be an object)."

        if "class-name" not in meta:
            return False, f"Line {lineno}: 'single-label-metadata' missing required key 'class-name'."

        class_name = meta.get("class-name")
        if not isinstance(class_name, str):
            return False, f"Line {lineno}: 'single-label-metadata.class-name' must be a string (can be empty)."

        # filtering rule
        if class_name == "":
            return True, ""

        return False, ""

    if label_type == "multi-label":
        # require multi-label to exist and be a list
        ml = obj.get("multi-label")
        if not isinstance(ml, list):
            return False, f"Line {lineno}: missing or invalid 'multi-label' (must be a list)."

        meta = obj.get("multi-label-metadata")
        if not isinstance(meta, dict):
            return False, f"Line {lineno}: missing or invalid 'multi-label-metadata' (must be an object)."

        class_map = meta.get("class-map")
        if not isinstance(class_map, dict):
            return False, f"Line {lineno}: 'multi-label-metadata.class-map' must be an object/dict."

        # values must be strings (can be empty? you did not allow empty here; we’ll require non-empty strings)
        for k, v in class_map.items():
            if not isinstance(v, str):
                return False, f"Line {lineno}: 'multi-label-metadata.class-map' values must be strings."

        # filtering rules
        if len(ml) == 0:
            return True, ""

        if len(class_map) == 0:
            return True, ""

        return False, ""

    if label_type == "object-detection":
        od = obj.get("object-detection")
        if not isinstance(od, dict):
            return False, f"Line {lineno}: missing or invalid 'object-detection' (must be an object)."

        annotations = od.get("annotations")
        if not isinstance(annotations, list):
            return False, f"Line {lineno}: 'object-detection.annotations' must be a list."

        # filtering rule: blank image
        if len(annotations) == 0:
            return True, ""

        # validate each annotation has top/left/height/width (and ideally class_id)
        for i, ann in enumerate(annotations):
            if not isinstance(ann, dict):
                return False, f"Line {lineno}: annotation[{i}] must be an object."
            for field in ("top", "left", "height", "width"):
                if field not in ann:
                    return False, f"Line {lineno}: annotation[{i}] missing required field '{field}'."
                if not isinstance(ann[field], (int, float)):
                    return False, f"Line {lineno}: annotation[{i}].{field} must be a number."

            if "class_id" not in ann:
                return False, f"Line {lineno}: annotation[{i}] missing required field 'class_id'."
            if not isinstance(ann["class_id"], int):
                return False, f"Line {lineno}: annotation[{i}].class_id must be an integer."

        meta = obj.get("object-detection-metadata")
        if not isinstance(meta, dict):
            return False, f"Line {lineno}: missing or invalid 'object-detection-metadata' (must be an object)."

        class_map = meta.get("class-map")
        if not isinstance(class_map, dict):
            return False, f"Line {lineno}: 'object-detection-metadata.class-map' must be an object/dict."

        # class-map values must be strings
        for _, v in class_map.items():
            if not isinstance(v, str):
                return False, f"Line {lineno}: 'object-detection-metadata.class-map' values must be strings."

        return False, ""

    if label_type == "semantic-segmentation":
        mask_ref = obj.get("semantic-segmentation-ref")
        if not _is_valid_s3_uri(mask_ref):
            return False, f"Line {lineno}: 'semantic-segmentation-ref' must be a valid s3://bucket/key URI."

        ext = _s3_key_ext(mask_ref)
        if ext != ".png":
            return False, f"Line {lineno}: 'semantic-segmentation-ref' must end with .png (got '{ext or '<no extension>'}')."

        meta = obj.get("semantic-segmentation-ref-metadata")
        if not isinstance(meta, dict):
            return False, f"Line {lineno}: missing or invalid 'semantic-segmentation-ref-metadata' (must be an object)."

        icm = meta.get("internal-color-map")
        if not isinstance(icm, dict) or len(icm) == 0:
            return False, f"Line {lineno}: 'semantic-segmentation-ref-metadata.internal-color-map' must be a non-empty object."

        # Validate map entries
        for k, v in icm.items():
            if not isinstance(v, dict):
                return False, f"Line {lineno}: internal-color-map['{k}'] must be an object."
            cn = v.get("class-name")
            hc = v.get("hex-color")
            if not isinstance(cn, str):
                return False, f"Line {lineno}: internal-color-map['{k}'].class-name must be a string."
            if not isinstance(hc, str) or not hc.startswith("#") or len(hc) not in (4, 7):
                return False, f"Line {lineno}: internal-color-map['{k}'].hex-color must look like '#rgb' or '#rrggbb'."

        return False, ""

    if label_type == "instance-segmentation":
        meta = obj.get("instance-segmentation-metadata")
        if not isinstance(meta, dict):
            return False, f"Line {lineno}: missing or invalid 'instance-segmentation-metadata' (must be an object)."

        wrr = meta.get("worker-response-ref")
        if not _is_valid_s3_uri(wrr):
            return False, f"Line {lineno}: 'instance-segmentation-metadata.worker-response-ref' must be a valid s3://bucket/key URI."

        # not strictly required by you, but this is the expected shape
        ext = _s3_key_ext(wrr)
        if ext != ".json":
            return False, f"Line {lineno}: worker-response-ref should end with .json (got '{ext or '<no extension>'}')."

        return False, ""

    # Should never happen due to earlier validation
    return False, f"Line {lineno}: unsupported label_type '{label_type}'."

def _make_filtered_path(original: Path) -> Path:
    # Write a .jsonl output regardless of input extension
    base = original.with_suffix("")  # removes the last suffix
    out = Path(str(base) + ".filtered.jsonl")

    # Avoid overwriting an existing file by adding a numeric suffix if needed
    if not out.exists():
        return out

    for i in range(1, 10_000):
        candidate = Path(str(base) + f".filtered.{i}.jsonl")
        if not candidate.exists():
            return candidate

    raise RuntimeError("Unable to choose a unique filtered output path.")

def _write_jsonl(path: Path, objects: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in objects:
            f.write(json.dumps(obj, separators=(",", ":")) + "\n")