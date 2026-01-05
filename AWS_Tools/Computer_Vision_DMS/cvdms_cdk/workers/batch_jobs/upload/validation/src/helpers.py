import io
import time
import json
import base64
import uuid

import boto3
from botocore.exceptions import ClientError
from PIL import Image
import numpy as np

s3 = boto3.client("s3")

LOWERCASE_BG_NAMES_POSSIBLE = ['bg', 'background']

def stable_uuid5(seed: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))

def _label_uuid(stable_seed: str | None, fallback_prefix: str) -> str:
    """
    stable_seed: string that uniquely identifies the image/job/label_type.
    fallback_prefix: label subtype; included so different label kinds don't collide.
    """
    if stable_seed:
        return stable_uuid5(f"{stable_seed}|{fallback_prefix}")
    return str(uuid.uuid4())

def read_manifest_with_retry(bucket, key, retries=5, delay=2):
    for attempt in range(retries):
        try:
            return s3.get_object(Bucket=bucket, Key=key)
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                if attempt < retries - 1:
                    time.sleep(delay)
                    continue
            raise RuntimeError(f'Unknown exception in loading manifest file: {e}')

    return None

def _put_json(key: str, payload: dict, file_bucket_name: str) -> str:
    s3.put_object(
        Bucket=file_bucket_name,
        Key=key,
        Body=json.dumps(payload).encode("utf-8"),
        ContentType="application/json",
    )
    return f"s3://{file_bucket_name}/{key}"

# Image feature calculation helpers
def infer_dtype(img):
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

def _put_png(key: str, png_bytes: bytes, file_bucket_name: str) -> str:
    s3.put_object(
        Bucket=file_bucket_name,
        Key=key,
        Body=png_bytes,
        ContentType="image/png",
    )
    return f"s3://{file_bucket_name}/{key}"

def create_and_save_labels(line, label_type, job_id, file_bucket_name, stable_seed: str | None = None):
    try:
        if label_type == "single-label":
            meta = line.get("single-label-metadata", {})
            cn = meta.get("class-name", "").strip().lower()
            if not cn:
                return [], [], "empty class-name for single label type"
            else:
                return [], [cn], None
        elif label_type == "multi-label":
            ml = line.get("multi-label", None)
            meta = line.get("multi-label-metadata", {})
            class_map = meta.get("class-map", None)

            if not ml or not isinstance(ml, list):
                return [], [], "multi-label missing/empty"
            if not class_map or not isinstance(class_map, dict):
                return [], [], "multi-label class-map missing/empty"

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
                return [], [], "No classes are present for multi-label image"
            else:
                return [], classes_present, None
        elif label_type == "object-detection":
            paths, classes_present = _create_object_detection_label(line, job_id, file_bucket_name, stable_seed)
            return paths, classes_present, None
        elif label_type == "semantic-segmentation":
            paths, classes_present = _create_semantic_segmentation_label(line, job_id, file_bucket_name, stable_seed)
            return paths, classes_present, None
        elif label_type == "instance-segmentation":
            paths, classes_present = _create_instance_segmentation_label(line, job_id, file_bucket_name, stable_seed)
            return paths, classes_present, None

        return [], [], f"Unsupported label_type: {label_type}"

    except Exception as e:
        return [], [], f"Error creating and saving label for label type {label_type}: {e}"

def _create_object_detection_label(line, job_id, file_bucket_name, stable_seed: str | None = None) -> tuple[list[str], list[str]]:
    od = line.get("object-detection", {})
    meta = line.get("object-detection-metadata", {})
    anns = od.get("annotations", None)
    class_map = meta.get("class-map", None)

    if not isinstance(anns, list) or len(anns) == 0:
        raise ValueError("object-detection.annotations missing/empty")
    if not isinstance(class_map, dict) or len(class_map) == 0:
        raise ValueError("object-detection-metadata.class-map missing/empty")

    out_anns = []
    classes_present = []
    for ann in anns:
        if not isinstance(ann, dict):
            continue
        cid = ann.get("class_id")
        if not isinstance(cid, int):
            raise ValueError("object-detection annotation missing int class_id")
        class_name = class_map.get(str(cid))
        if not isinstance(class_name, str) or not class_name.strip():
            raise ValueError(f"class-map missing class for class_id={cid}")

        classes_present.append(class_name.strip().lower())

        coords = {}
        for f in ("top", "left", "height", "width"):
            v = ann.get(f)
            if not isinstance(v, (int, float)):
                raise ValueError(f"object-detection annotation.{f} must be number")
            coords[f] = float(v)

        out_anns.append({"class_name": class_name.strip().lower(), "coordinates": coords})

    # after building classes_present
    seen = set()
    uniq = []
    for c in classes_present:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    classes_present = uniq

    label_uuid = _label_uuid(stable_seed, "object-detection")
    key = f"temp/image-upload/{job_id}/object-detection/{label_uuid}.json"
    uri = _put_json(key, {"annotations": out_anns}, file_bucket_name)
    return [uri], classes_present

def _create_semantic_segmentation_label(line, job_id, file_bucket_name, stable_seed: str | None = None) -> tuple[list[str], list[str]]:
    mask_ref = line.get("semantic-segmentation-ref")
    meta = line.get("semantic-segmentation-ref-metadata", {})
    icm = meta.get("internal-color-map", None)

    if not isinstance(mask_ref, str) or not mask_ref.startswith("s3://"):
        raise ValueError("semantic-segmentation-ref missing/invalid")
    if not isinstance(icm, dict) or len(icm) == 0:
        raise ValueError("semantic internal-color-map missing/empty")

    # Build hex -> class map and require bg
    hex_to_class = {}
    bg_hex = None
    for _, v in icm.items():
        if not isinstance(v, dict):
            continue
        cn = v.get("class-name")
        hc = v.get("hex-color")
        if not isinstance(cn, str) or not isinstance(hc, str):
            continue
        cn = cn.strip()
        hc = normalize_hex(hc)
        hex_to_class[hc] = cn
        if cn.lower() in LOWERCASE_BG_NAMES_POSSIBLE:
            bg_hex = hc

    if bg_hex is None:
        raise ValueError("semantic error: internal-color-map must include class-name 'bg' or 'background' (case insensitive) for background")

    # Assign IDs: bg=0, then 1..N
    non_bg = [(h, c) for h, c in hex_to_class.items() if c.lower() not in LOWERCASE_BG_NAMES_POSSIBLE]
    non_bg.sort(key=lambda t: (t[1], t[0]))  # deterministic
    hex_to_id = {bg_hex: 0}
    id_to_class = {"0": "bg"}

    next_id = 1
    for h, c in non_bg:
        if h == bg_hex:
            continue
        hex_to_id[h] = next_id
        id_to_class[str(next_id)] = c.lower()
        next_id += 1

    # Load the RGB mask
    mb, mk = mask_ref[5:].split("/", 1)
    mask_bytes = s3.get_object(Bucket=mb, Key=mk)["Body"].read()
    rgb = np.array(Image.open(io.BytesIO(mask_bytes)).convert("RGB"), dtype=np.uint8)

    # Convert to indexed mask
    h, w, _ = rgb.shape
    idx = np.zeros((h, w), dtype=np.uint8)

    packed = (rgb[:, :, 0].astype(np.uint32) << 16) | (rgb[:, :, 1].astype(np.uint32) << 8) | rgb[:, :, 2].astype(np.uint32)

    known = np.zeros_like(packed, dtype=bool)
    for hex_color, pid in hex_to_id.items():
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        tgt = (np.uint32(r) << 16) | (np.uint32(g) << 8) | np.uint32(b)
        idx[packed == tgt] = np.uint8(pid)
        known |= (packed == tgt)

    if not np.all(known):
        # compute how many pixels are unknown (cheap)
        unknown_count = int((~known).sum())
        raise ValueError(f"semantic mask contains {unknown_count} pixels with colors not in internal-color-map")

    present_ids = np.unique(idx)
    classes_present = []
    for pid in present_ids:
        if int(pid) == 0:
            continue
        cn = id_to_class.get(str(int(pid)))
        if cn and cn.lower() not in LOWERCASE_BG_NAMES_POSSIBLE:
            classes_present.append(cn.lower())

    # de-dupe preserve order
    seen = set()
    uniq = []
    for c in classes_present:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    classes_present = uniq

    # Save PNG + meta JSON under same uuid
    label_uuid = _label_uuid(stable_seed, "semantic-segmentation")
    png_key = f"temp/image-upload/{job_id}/semantic-segmentation/{label_uuid}.png"
    meta_key = f"temp/image-upload/{job_id}/semantic-segmentation/{label_uuid}.json"

    out_img = Image.fromarray(idx, mode="L")
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    png_uri = _put_png(png_key, buf.getvalue(), file_bucket_name)
    meta_uri = _put_json(meta_key, {"id_to_class": id_to_class}, file_bucket_name)

    if not classes_present:
        raise ValueError("semantic mask contains only background")

    return [png_uri, meta_uri], classes_present

def _create_instance_segmentation_label(line, job_id, file_bucket_name, stable_seed: str | None = None) -> tuple[list[str], list[str]]:
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

    # Each instance gets its own id (0 bg always)
    id_to_class = {"0": "bg"}
    hex_to_id = {}
    next_id = 1

    for inst in instances:
        if not isinstance(inst, dict):
            continue
        hc = inst.get("color")
        lab = inst.get("label")
        if not isinstance(hc, str) or not isinstance(lab, str) or not lab.strip():
            continue

        hc = normalize_hex(hc)
        lab = lab.strip()

        # IMPORTANT: each instance gets a new ID even if label repeats
        if hc in hex_to_id:
            raise ValueError(f"duplicate instance color in worker response: {hc}")

        hex_to_id[hc] = next_id
        id_to_class[str(next_id)] = lab.lower()
        next_id += 1

    # decode and load RGB mask
    mask_bytes = base64.b64decode(png_b64)
    rgb = np.array(Image.open(io.BytesIO(mask_bytes)).convert("RGB"), dtype=np.uint8)

    # Convert to indexed mask (unknown colors -> 0 bg)
    h, w, _ = rgb.shape
    idx = np.zeros((h, w), dtype=np.uint8)
    packed = (rgb[:, :, 0].astype(np.uint32) << 16) | (rgb[:, :, 1].astype(np.uint32) << 8) | rgb[:, :, 2].astype(np.uint32)

    known = np.zeros_like(packed, dtype=bool)
    for hex_color, pid in hex_to_id.items():
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        tgt = (np.uint32(r) << 16) | (np.uint32(g) << 8) | np.uint32(b)
        idx[packed == tgt] = np.uint8(pid)
        known |= (packed == tgt)

    if not np.all(known):
        # compute how many pixels are unknown (cheap)
        unknown_count = int((~known).sum())
        raise ValueError(f"instance mask contains {unknown_count} pixels with colors not present in worker-response instances")

    present_ids = np.unique(idx)
    classes_present = []
    for pid in present_ids:
        if int(pid) == 0:
            continue
        cn = id_to_class.get(str(int(pid)))
        if cn and cn.lower() not in LOWERCASE_BG_NAMES_POSSIBLE:
            classes_present.append(cn.lower())

    # de-dupe preserve order
    seen = set()
    uniq = []
    for c in classes_present:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    classes_present = uniq

    label_uuid = _label_uuid(stable_seed, "instance-segmentation")
    png_key = f"temp/image-upload/{job_id}/instance-segmentation/{label_uuid}.png"
    meta_key = f"temp/image-upload/{job_id}/instance-segmentation/{label_uuid}.json"

    out_img = Image.fromarray(idx, mode="L")
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    png_uri = _put_png(png_key, buf.getvalue(), file_bucket_name)
    meta_uri = _put_json(meta_key, {"id_to_class": id_to_class}, file_bucket_name)

    if not classes_present:
        raise ValueError("instance segmentation mask contains only background")

    return [png_uri, meta_uri], classes_present