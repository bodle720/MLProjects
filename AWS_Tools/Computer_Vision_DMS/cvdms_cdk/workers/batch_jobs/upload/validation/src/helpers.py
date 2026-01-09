import io
import json
import base64
import uuid
import hashlib
from typing import Optional

import boto3
from PIL import Image
import numpy as np

from common.s3_utils import write_s3_obj

s3 = boto3.client("s3")

LOWERCASE_BG_NAMES_POSSIBLE = ['bg', 'background']

def stable_uuid5(seed: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))

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
    if not h.startswith("#"):
        raise ValueError(f"Bad hex color: {h}")
    if len(h) == 4:  # #rgb
        return f"#{h[1]}{h[1]}{h[2]}{h[2]}{h[3]}{h[3]}"
    if len(h) == 7:
        return h
    raise ValueError(f"Bad hex color: {h}")

def create_and_save_labels(line: dict,
                           label_type: str,
                           job_id: str,
                           file_bucket_name: str) -> tuple[list[str], list[str], Optional[str], Optional[str]]:
    try:
        if label_type == "single-label":
            meta = line.get("single-label-metadata", {})
            cn = meta.get("class-name", "").strip().lower()
            if not cn:
                return [], [], None, "empty class-name for single label type"
            else:
                return [], [cn], None, None
        elif label_type == "multi-label":
            ml = line.get("multi-label", None)
            meta = line.get("multi-label-metadata", {})
            class_map = meta.get("class-map", None)

            if not ml or not isinstance(ml, list):
                return [], [], None, "multi-label missing/empty"
            if not class_map or not isinstance(class_map, dict):
                return [], [], None, "multi-label class-map missing/empty"

            out = []
            for idx in ml:
                key = str(idx)
                name = class_map.get(key)
                if isinstance(name, str) and name.strip():
                    out.append(name.strip().lower())

            # unique preserve order
            seen = set()
            classes_present = []
            for x in out:
                if x not in seen:
                    seen.add(x)
                    classes_present.append(x)

            if not classes_present:
                return [], [], None, "No classes are present for multi-label image"
            else:
                return [], classes_present, None, None
        elif label_type == "object-detection":
            paths, classes_present, label_fingerprint = _create_object_detection_label(line, job_id, file_bucket_name)
            return paths, classes_present, label_fingerprint, None
        elif label_type == "semantic-segmentation":
            paths, classes_present, label_fingerprint = _create_semantic_segmentation_label(line, job_id, file_bucket_name)
            return paths, classes_present, label_fingerprint, None
        elif label_type == "instance-segmentation":
            paths, classes_present, label_fingerprint = _create_instance_segmentation_label(line, job_id, file_bucket_name)
            return paths, classes_present, label_fingerprint, None

        return [], [], None, f"Unsupported label_type: {label_type}"

    except Exception as e:
        return [], [], None, f"[VAL_JOB_DEF] Error creating and saving label for label type {label_type}: {e}"

def _create_object_detection_label(line: dict, job_id: str, file_bucket_name: str) -> tuple[list[str], list[str], str]:
    od = line.get("object-detection", {})
    meta = line.get("object-detection-metadata", {})
    anns = od.get("annotations", None)
    class_map = meta.get("class-map", None)

    if not isinstance(anns, list) or len(anns) == 0:
        raise ValueError("object-detection.annotations missing/empty")
    if not isinstance(class_map, dict) or len(class_map) == 0:
        raise ValueError("object-detection-metadata.class-map missing/empty")

    tuples_anns = [] # out anns, but as a list of tuples
    classes_present = []
    for ann in anns:
        if not isinstance(ann, dict):
            continue

        cid = ann.get("class_id")

        if type(cid) == str:
            try:
                cid = int(cid)
            except:
                raise ValueError(f"class id in bbox annotation is not int or string version of int: {cid}")

        if not isinstance(cid, int):
            raise ValueError("object-detection annotation missing int class_id")

        class_name = class_map.get(str(cid))
        if not isinstance(class_name, str) or not class_name.strip():
            raise ValueError(f"class-map missing class for class_id={cid}")

        cn = class_name.strip().lower()
        classes_present.append(cn)

        coords = {}
        for f in ("top", "left", "height", "width"):
            v = ann.get(f)

            if type(v) == str:
                try:
                    v = int(v)
                except:
                    raise ValueError(f"coord {f} in bbox annotation is not int or string version of int: {v}")

            if not isinstance(v, int):
                raise ValueError(f"object-detection annotation.{f} must be number")

            coords[f] = v

        tuples_anns.append((cn, coords['top'], coords['left'], coords['height'], coords['width']))

    # make label fingerprint
    if not tuples_anns:
        raise ValueError("object-detection.annotations contained no valid annotation objects")

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
        {"class_name": cn, "coordinates": {"top": top, "left": left, "height": height, "width": width}}
        for (cn, top, left, height, width) in tuples_anns
    ]

    key = f"temp/image-upload/{job_id}/object-detection/{label_fingerprint}.json"
    uri = write_s3_obj(file_bucket_name,
                          key,
                          json.dumps({"annotations": out_anns}),
                          "application/json",
                          "[VAL_JOB_DEF]")

    return [uri], classes_present, label_fingerprint

def _create_semantic_segmentation_label(line: dict, job_id: str, file_bucket_name: str) -> tuple[list[str], list[str], str]:
    """
    Color-invariant semantic fingerprinting:
    - Uses internal-color-map to map RGB colors -> class names
    - Assigns numeric IDs by CLASS NAME ONLY (bg=0, others sorted by class name)
    - Any pixel color not in internal-color-map is treated as background (0) by default
      (i.e., we do NOT raise on unknown colors).
    """
    mask_ref = line.get("semantic-segmentation-ref")
    meta = line.get("semantic-segmentation-ref-metadata", {})
    icm = meta.get("internal-color-map", None)

    if not isinstance(mask_ref, str) or not mask_ref.startswith("s3://"):
        raise ValueError("semantic-segmentation-ref missing/invalid")
    if not isinstance(icm, dict) or len(icm) == 0:
        raise ValueError("semantic internal-color-map missing/empty")

    # Build class_name -> set(hex_colors), require a background class
    class_to_hexes: dict[str, set[str]] = {}
    bg_class: str | None = None

    for _, v in icm.items():
        if not isinstance(v, dict):
            continue

        cn = v.get("class-name")
        hc = v.get("hex-color")
        if not isinstance(cn, str) or not cn.strip():
            continue
        if not isinstance(hc, str) or not hc.strip():
            continue

        cn = cn.strip().lower()
        hc = normalize_hex(hc)

        class_to_hexes.setdefault(cn, set()).add(hc)

        if cn in LOWERCASE_BG_NAMES_POSSIBLE:
            # You can allow both "bg" and "background" to exist; we just need *a* bg class.
            if bg_class is None:
                bg_class = cn

    if bg_class is None:
        raise ValueError(
            "semantic error: internal-color-map must include class-name 'bg' or 'background' (case insensitive) for background"
        )

    # Deterministic class IDs by CLASS NAME ONLY (color-independent)
    non_bg_classes = sorted([c for c in class_to_hexes.keys() if c not in LOWERCASE_BG_NAMES_POSSIBLE])

    id_to_class: dict[str, str] = {"0": "bg"}
    class_to_id: dict[str, int] = {bg_class: 0}

    next_id = 1
    for c in non_bg_classes:
        if next_id > 255:
            raise ValueError("too many classes for uint8 mask")
        class_to_id[c] = next_id
        id_to_class[str(next_id)] = c
        next_id += 1

    # Load the RGB mask
    mb, mk = mask_ref[5:].split("/", 1)
    mask_bytes = s3.get_object(Bucket=mb, Key=mk)["Body"].read()
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
    for cls, hexes in class_to_hexes.items():
        pid = class_to_id.get(cls)
        if pid is None:
            continue  # shouldn't happen
        for hex_color in hexes:
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
        if cn and cn not in LOWERCASE_BG_NAMES_POSSIBLE:
            classes_present.append(cn)

    # de-dupe preserve order (unique already, but keep your pattern)
    seen = set()
    uniq = []
    for c in classes_present:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    classes_present = uniq

    if not classes_present:
        raise ValueError("semantic mask contains only background")

    # Save PNG + meta JSON using fingerprint as key
    png_key = f"temp/image-upload/{job_id}/semantic-segmentation/{label_fingerprint}.png"
    meta_key = f"temp/image-upload/{job_id}/semantic-segmentation/{label_fingerprint}.json"

    out_img = Image.fromarray(idx, mode="L")
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")

    png_uri = write_s3_obj(file_bucket_name, png_key, buf.getvalue(), "image/png", "[VAL_JOB_DEF]")

    meta_uri = write_s3_obj(file_bucket_name, meta_key, json.dumps({"id_to_class": id_to_class}), "application/json", "[VAL_JOB_DEF]")

    return [png_uri, meta_uri], classes_present, label_fingerprint

def _create_instance_segmentation_label(line: dict, job_id: str, file_bucket_name: str) -> tuple[list[str], list[str], str]:
    meta = line.get("instance-segmentation-metadata", {})
    wrr = meta.get("worker-response-ref")
    if not isinstance(wrr, str) or not wrr.startswith("s3://"):
        raise ValueError("instance: worker-response-ref missing/invalid")

    wb, wk = wrr[5:].split("/", 1)
    wrr_bytes = s3.get_object(Bucket=wb, Key=wk)["Body"].read()
    wrr_json = json.loads(wrr_bytes.decode("utf-8"))

    answers = wrr_json.get("answers", [])
    if not isinstance(answers, list) or not answers:
        raise ValueError("instance: worker response missing answers")

    ar = answers[0].get("answerContent", {}).get("annotatedResult", {})
    instances = ar.get("instances", [])
    png_b64 = ar.get("labeledImage", {}).get("pngImageData")

    if not isinstance(instances, list) or not instances:
        raise ValueError("instance: worker response missing instances")
    if not isinstance(png_b64, str) or not png_b64.strip():
        raise ValueError("instance: worker response missing labeledImage.pngImageData")

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
            raise ValueError(f"duplicate instance color in worker response: {hc}")
        seen_colors.add(hc)
        parsed.append((hc, lab.strip().lower()))

    if not parsed:
        raise ValueError("instance: no valid instances parsed from worker response")

    # decode and load RGB mask
    mask_bytes = base64.b64decode(png_b64)
    rgb = np.array(Image.open(io.BytesIO(mask_bytes)).convert("RGB"), dtype=np.uint8)
    h, w, _ = rgb.shape

    packed = (
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

        m = (packed == tgt)

        if not m.any():
            raise ValueError(f"instance: instance color {hc} ({lab}) has 0 pixels in mask")

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
            raise ValueError("too many instances for uint8 mask")
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
        if cn and cn not in LOWERCASE_BG_NAMES_POSSIBLE:
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
        raise ValueError("instance segmentation mask contains only background")

    # Save outputs using fingerprint as key
    png_key = f"temp/image-upload/{job_id}/instance-segmentation/{label_fingerprint}.png"
    meta_key = f"temp/image-upload/{job_id}/instance-segmentation/{label_fingerprint}.json"

    out_img = Image.fromarray(idx, mode="L")
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")

    png_uri = write_s3_obj(file_bucket_name, png_key, buf.getvalue(), "image/png", "[VAL_JOB_DEF]")
    meta_uri = write_s3_obj(file_bucket_name, meta_key, json.dumps({"id_to_class": id_to_class}), "application/json", "[VAL_JOB_DEF]")

    return [png_uri, meta_uri], classes_present, label_fingerprint