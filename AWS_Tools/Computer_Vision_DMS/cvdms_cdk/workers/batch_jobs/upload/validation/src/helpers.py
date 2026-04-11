import io
import json
import base64
import uuid
import hashlib
from typing import Optional, Dict

import boto3
from PIL import Image
import numpy as np
from numpy.typing import NDArray
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

from common.general_utils.s3_utils import write_s3_obj, parse_s3_uri, read_obj_with_retry
from common.general_utils.class_normalizer import canonicalize_class_name

TASK_NAME = "[VAL_JOB_DEF_HELPER]"

s3 = boto3.client("s3")

def stable_uuid5(seed: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))

def parse_json_object_line(line) -> Optional[Dict]:
    def _bad_const(x: str):
        raise ValueError(f"Invalid JSON constant: {x}")  # NaN/Infinity
    try:
        parsed = json.loads(line, parse_constant=_bad_const)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None

def normalize_numeric_4(v) -> str:
    try:
        d = Decimal(str(v))
    except (InvalidOperation, ValueError):
        raise ValueError(f"bad numeric: {v}")

    if not d.is_finite():
        raise ValueError("non-finite number")

    d = d.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    if d == 0:
        d = Decimal("0.0000")

    # ensure fixed 4 decimals
    s = format(d, "f")
    if "." not in s:
        s += ".0000"
    else:
        whole, frac = s.split(".", 1)
        s = whole + "." + (frac + "0000")[:4]
    return s

# Image feature calculation helpers
def infer_dtype(img) -> str:
    mode = img.mode
    if mode in ("L", "RGB", "RGBA", "CMYK", "YCbCr"):
        return "uint8"
    if mode in ("I;16", "I;16B", "I;16L"):
        return "uint16"
    if mode == "I":
        return "int32"
    if mode == "F":
        return "float32"
    if mode == "1":
        return "bool"
    return str(mode)  # fallback

def normalize_hex(h: str) -> str:
    h = h.strip().lower()

    if len(h) != 7:
        raise ValueError(f"{TASK_NAME} Bad hex color: {h}, length must be 7")

    if not h.startswith("#"):
        raise ValueError(f"Bad hex color: {h}, no # sign as first character")

    hexpart = h[1:]
    if any(c not in "0123456789abcdef" for c in hexpart):
        raise ValueError(f"Bad hex color: {h}, non hex digits")

    return h

def create_and_save_labels(line: dict,
                           label_type: str,
                           job_id: str,
                           file_bucket_name: str) -> tuple[list[str], list[str], Optional[str], Optional[str]]:
    try:
        if label_type == "single-label" or label_type == "multi-label":
            label_ls = line.get("labels", [])

            if not label_ls:
                return [], [], None, f"empty labels list for {label_type} type"
            elif not isinstance(label_ls, list):
                return [], [], None, f"no list labels list for {label_type} type"
            elif label_type == "single-label" and len(label_ls) != 1:
                return [], [], None, "labels list for single label type is not len 1"
            elif not all(isinstance(cn, str) and cn.strip() for cn in label_ls):
                return [], [], None, f"labels list values for label type {label_type} are not all non-empty strings"

            normed_labels = [cn.strip().lower() for cn in label_ls]
            labels = sorted(list(set(normed_labels)))
            return [], labels, None, None
        elif label_type == "object-detection":
            paths, classes_present, label_fingerprint = create_object_detection_label(line, job_id, file_bucket_name)
            return paths, classes_present, label_fingerprint, None
        elif label_type == "semantic-segmentation":
            paths, classes_present, label_fingerprint = create_semantic_segmentation_label(line, job_id, file_bucket_name)
            return paths, classes_present, label_fingerprint, None
        elif label_type == "instance-segmentation":
            paths, classes_present, label_fingerprint = create_instance_segmentation_label(line, job_id, file_bucket_name)
            return paths, classes_present, label_fingerprint, None

        return [], [], None, f"Unsupported label_type: {label_type}"

    except Exception as e:
        return [], [], None, f"{TASK_NAME} Error creating and saving label for label type {label_type}: {e}"

def create_object_detection_label(line: dict, job_id: str, file_bucket_name: str) -> tuple[list[str], list[str], str]:
    bboxes = line.get("labels", {}).get('boxes', [])

    if not isinstance(bboxes, list) or len(bboxes) == 0:
        raise ValueError(f"{TASK_NAME} object-detection annotations missing/empty: no bboxes list present")

    tuples_anns = []
    classes_present = []
    for ann in bboxes:
        if not isinstance(ann, dict):
            continue

        cn = ann.get("class_name")
        if not isinstance(cn, str) or not cn.strip():
            raise ValueError(f"{TASK_NAME} object-detection invalid class: {cn}")

        class_name = cn.strip().lower()
        if class_name in {"bg", "background"}:
            raise ValueError(f"{TASK_NAME} object: reserved class name used: {class_name}")

        classes_present.append(class_name)

        coords = {}
        for f in ("top", "left", "height", "width"):
            v = ann.get(f)

            if v is None or not isinstance(v, (int, float, str)):
                raise ValueError(f"{TASK_NAME} object-detection annotation bbox field {f} must be present and numeric")

            if isinstance(v, str) and not v.strip():
                raise ValueError(f"{TASK_NAME} object-detection annotation bbox field {f} must be present and numeric")

            v = normalize_numeric_4(v) # type str

            if f in ['height', 'width'] and float(v) <= 0:
                raise ValueError(f"{TASK_NAME} object-detection annotation bbox field {f} must be non-zero and positive")

            if f in ['top', 'left'] and float(v) < 0:
                raise ValueError(f"{TASK_NAME} object-detection annotation bbox field {f} must be non-negative")

            coords[f] = v

        tuples_anns.append((class_name, coords['top'], coords['left'], coords['height'], coords['width']))

    # make label fingerprint
    if not tuples_anns:
        raise ValueError(f"{TASK_NAME} object-detection annotations contained no valid annotation objects")

    tuples_anns.sort()
    payload = {"v": 1, "label_type": "object-detection", "boxes": [list(t) for t in tuples_anns]}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    label_fingerprint = hashlib.sha256(blob).hexdigest()

    # after building classes_present
    seen = set()
    uniq = []
    for c in classes_present:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    classes_present = uniq

    out_anns = [
        {"class_name": cn, "top": top, "left": left, "height": height, "width": width}
        for (cn, top, left, height, width) in tuples_anns
    ]

    key = f"temp/image-upload/{job_id}/object-detection/{label_fingerprint}.json"
    uri = write_s3_obj(file_bucket_name,
                          key,
                          json.dumps({"annotations": out_anns}),
                          "application/json",
                          TASK_NAME)

    return [uri], classes_present, label_fingerprint

def create_semantic_segmentation_label(line: dict, job_id: str, file_bucket_name: str) -> tuple[list[str], list[str], str]:
    """
    Color-invariant semantic fingerprinting:
    - Uses internal-color-map to map RGB colors -> class names
    - Assigns numeric IDs by CLASS NAME ONLY (bg=0, others sorted by class name)
    - Any pixel color not in internal-color-map is treated as background (0) by default
      (i.e., we do NOT raise on unknown colors).
    """
    mask_ref = line.get("mask_ref")
    color_map = line.get("color_map")

    if not isinstance(mask_ref, str) or not mask_ref.startswith("s3://"):
        raise ValueError(f"{TASK_NAME} semantic segmentation mask_ref missing/invalid")
    if not isinstance(color_map, dict) or len(color_map) == 0:
        raise ValueError(f"{TASK_NAME} semantic color_map missing/empty")

    # Build class_name -> set(hex_colors)
    class_to_hex: dict[str, str] = {}
    for cn, hc_ls in color_map.items():
        if not isinstance(cn, str) or not cn.strip():
            raise ValueError(f"{TASK_NAME} semantic-segmentation invalid class: {cn}")

        if not isinstance(hc_ls, list) or not hc_ls:
            raise ValueError(f"{TASK_NAME} semantic-segmentation invalid hex color list: {hc_ls}")

        if len(hc_ls) != 1:
            raise ValueError(f"{TASK_NAME} semantic color map has hex list of len greater than 1")

        cn = cn.strip().lower()

        if cn in {"bg", "background"}:
            raise ValueError(f"{TASK_NAME} semantic: v1 color_map must not include '{cn}' (reserved)")

        hc = normalize_hex(hc_ls[0])
        class_to_hex[cn] = hc

    hex_to_class: dict[str, str] = {}
    for cls, hexval in class_to_hex.items():
        if hexval in hex_to_class and hex_to_class[hexval] != cls:
            raise ValueError("duplicate hex-color assigned to multiple classes")
        hex_to_class[hexval] = cls


    # Deterministic class IDs by CLASS NAME ONLY (color-independent)
    all_classes = sorted([c for c in class_to_hex.keys() if c not in {"bg", "background"}])

    id_to_class: dict[str, str] = {"0": "bg"}
    class_to_id: dict[str, int] = {"bg": 0}

    next_id = 1
    for c in all_classes:
        if next_id > 255:
            raise ValueError(f"{TASK_NAME} Too many classes for uint8 mask")
        class_to_id[c] = next_id
        id_to_class[str(next_id)] = c
        next_id += 1

    # Load the RGB mask
    mb, mk = parse_s3_uri(mask_ref, TASK_NAME)
    resp = read_obj_with_retry(mb, mk, TASK_NAME)
    if resp is None:
        raise ValueError(f"{TASK_NAME} mask_ref unable to be loaded with retry: {mask_ref}, parsed bucket = {mb}, parsed key = {mk}")

    mask_bytes = resp["Body"].read()
    rgb = np.array(Image.open(io.BytesIO(mask_bytes)).convert("RGB"), dtype=np.uint8)

    # Convert to indexed mask (unknown colors remain 0 == background)
    h, w, _ = rgb.shape
    idx = np.zeros((h, w), dtype=np.uint8)

    packed = (
        (rgb[:, :, 0].astype(np.uint32) << 16)
        | (rgb[:, :, 1].astype(np.uint32) << 8)
        | rgb[:, :, 2].astype(np.uint32)
    )

    # Fill idx by class (union all colors that map to that class)
    for cls, hex_color in class_to_hex.items():
        pid = class_to_id.get(cls)
        if pid is None:
            continue  # shouldn't happen

        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        tgt = (np.uint32(r) << 16) | (np.uint32(g) << 8) | np.uint32(b)
        idx[packed == tgt] = np.uint8(pid)

    # Fingerprint: mapping (stable) + pixels (stable) => color-invariant across GT jobs
    pixel_bytes = idx.tobytes(order="C")
    mapping_pairs = sorted(id_to_class.items(), key=lambda kv: int(kv[0]))
    payload = {
        "v": 1,
        "label_type": "semantic-segmentation",
        "h": int(h),
        "w": int(w),
        "id_to_class": mapping_pairs,
    }
    meta_blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    label_fingerprint = hashlib.sha256(meta_blob + b"|" + pixel_bytes).hexdigest()

    # classes_present from ids present
    present_ids = np.unique(idx)
    classes_present = []
    for pid in present_ids:
        if int(pid) == 0:
            continue
        cn = id_to_class.get(str(int(pid)))
        if cn and cn != 'bg':
            classes_present.append(cn)

    # de-dupe preserve order (unique already, but keep pattern)
    seen = set()
    uniq = []
    for c in classes_present:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    classes_present = uniq

    if not classes_present:
        raise ValueError(f"{TASK_NAME} semantic mask contains only background")

    # Save PNG + meta JSON using fingerprint as key
    png_key = f"temp/image-upload/{job_id}/semantic-segmentation/{label_fingerprint}.png"
    meta_key = f"temp/image-upload/{job_id}/semantic-segmentation/{label_fingerprint}.json"

    out_img = Image.fromarray(idx, mode="L")
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")

    png_uri = write_s3_obj(file_bucket_name, png_key, buf.getvalue(), "image/png", TASK_NAME)
    meta_uri = write_s3_obj(file_bucket_name, meta_key, json.dumps({"id_to_class": id_to_class}), "application/json", TASK_NAME)

    return [png_uri, meta_uri], classes_present, label_fingerprint

def create_instance_segmentation_label(line: dict, job_id: str, file_bucket_name: str) -> tuple[list[str], list[str], str]:
    wrr = line.get("worker_response_ref")
    if not isinstance(wrr, str) or not wrr.startswith("s3://"):
        raise ValueError(f"{TASK_NAME} instance: worker_response_ref missing/invalid")

    wb, wk = parse_s3_uri(wrr, TASK_NAME)
    resp = read_obj_with_retry(wb, wk, TASK_NAME)
    if resp is None:
        raise ValueError(f"{TASK_NAME} worker_response_ref unable to be loaded with retry: {wrr}, parsed bucket = {wb}, parsed key = {wk}")

    wrr_bytes = resp["Body"].read()
    wrr_json = parse_json_object_line(wrr_bytes.decode("utf-8"))

    if wrr_json is None:
        raise ValueError(f"{TASK_NAME} instance: worker_response_ref is not valid strict JSON")

    answers = wrr_json.get("answers", [])
    if not isinstance(answers, list) or not answers:
        raise ValueError(f"{TASK_NAME} instance: worker response missing answers")

    ar = answers[0].get("answerContent", {}).get("annotatedResult", {})
    instances = ar.get("instances", [])
    png_b64 = ar.get("labeledImage", {}).get("pngImageData")

    if not isinstance(instances, list) or not instances:
        raise ValueError(f"{TASK_NAME} instance: worker response missing instances")
    if not isinstance(png_b64, str) or not png_b64.strip():
        raise ValueError(f"{TASK_NAME} instance: worker response missing labeledImage.pngImageData")

    # Parse instances: keep (hex_color, label)
    parsed = []
    seen_colors = set()
    for inst in instances:
        if not isinstance(inst, dict):
            continue
        hc = inst.get("color")
        lab = inst.get("label")
        if not isinstance(hc, str) or not isinstance(lab, str) or not lab.strip():
            continue
        hc = normalize_hex(hc)
        if hc in seen_colors:
            raise ValueError(f"{TASK_NAME} duplicate instance color in worker response: {hc}")
        seen_colors.add(hc)

        lab = canonicalize_class_name(lab, field_name="instance.label", allow_background=False)

        parsed.append((hc, lab))

    if not parsed:
        raise ValueError(f"{TASK_NAME} instance: no valid instances parsed from worker response")

    # decode and load RGB mask
    mask_bytes = base64.b64decode(png_b64)
    rgb = np.array(Image.open(io.BytesIO(mask_bytes)).convert("RGB"), dtype=np.uint8)
    h, w, _ = rgb.shape

    packed: NDArray[np.uint32] = (
        (rgb[:, :, 0].astype(np.uint32) << 16)
        | (rgb[:, :, 1].astype(np.uint32) << 8)
        | rgb[:, :, 2].astype(np.uint32)
    )

    # Build per-instance masks and stable ordering keys
    instance_masks = []
    for hc, lab in parsed:
        r = int(hc[1:3], 16)
        g = int(hc[3:5], 16)
        b = int(hc[5:7], 16)
        tgt = (np.uint32(r) << 16) | (np.uint32(g) << 8) | np.uint32(b)

        m: NDArray[np.bool_] = (packed == tgt)

        if not m.any():
            raise ValueError(f"{TASK_NAME} instance: instance color {hc} ({lab}) has 0 pixels in mask")

        area = int(m.sum())
        ys, xs = np.where(m)
        top = int(ys.min()); bottom = int(ys.max())
        left = int(xs.min()); right = int(xs.max())

        # Tie-breaker that’s independent of the color: hash the binary mask bytes
        # (We already hash idx bytes later; this is just for deterministic ordering.)
        mask_hash = hashlib.sha256(m.tobytes(order="C")).hexdigest()

        # Sort key: label + geometry + content hash
        sort_key = (lab, top, left, bottom, right, area, mask_hash)
        instance_masks.append((sort_key, lab, m))

    # Deterministic ordering independent of GT color choices
    instance_masks.sort(key=lambda t: t[0])

    # Build indexed mask + mapping
    idx = np.zeros((h, w), dtype=np.uint8)
    id_to_class = {"0": "bg"}

    next_id = 1
    for _, lab, m in instance_masks:
        if next_id > 255:
            raise ValueError(f"{TASK_NAME} too many instances for uint8 mask")

        if np.any(idx[m] != 0):
            raise ValueError(f"{TASK_NAME} instance: overlapping instance masks detected")

        idx[m] = np.uint8(next_id)
        id_to_class[str(next_id)] = lab
        next_id += 1

    # Fingerprint (same as you do)
    pixel_bytes = idx.tobytes(order="C")
    mapping_pairs = sorted(id_to_class.items(), key=lambda kv: int(kv[0]))
    payload = {"v": 1, "label_type": "instance-segmentation", "h": int(h), "w": int(w), "id_to_class": mapping_pairs}
    meta_blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    label_fingerprint = hashlib.sha256(meta_blob + b"|" + pixel_bytes).hexdigest()

    # classes_present from ids present
    present_ids = np.unique(idx)
    classes_present = []
    for pid in present_ids:
        if int(pid) == 0:
            continue
        cn = id_to_class.get(str(int(pid)))
        if cn and cn not in {"bg", "background"}:
            classes_present.append(cn)

    # de-dupe preserve order
    seen = set()
    uniq = []
    for c in classes_present:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    classes_present = uniq

    if not classes_present:
        raise ValueError(f"{TASK_NAME} instance segmentation mask contains only background")

    # Save outputs using fingerprint as key
    png_key = f"temp/image-upload/{job_id}/instance-segmentation/{label_fingerprint}.png"
    meta_key = f"temp/image-upload/{job_id}/instance-segmentation/{label_fingerprint}.json"

    out_img = Image.fromarray(idx, mode="L")
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")

    png_uri = write_s3_obj(file_bucket_name, png_key, buf.getvalue(), "image/png", TASK_NAME)
    meta_uri = write_s3_obj(file_bucket_name, meta_key, json.dumps({"id_to_class": id_to_class}), "application/json", TASK_NAME)

    return [png_uri, meta_uri], classes_present, label_fingerprint