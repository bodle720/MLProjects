from typing import Any, Dict, List, Optional, Tuple
import hashlib

from common.s3_utils import (
    make_s3_uri,
    parse_s3_uri,
    get_key_basename,
    s3_copy_with_retry
)

TASK_NAME = "[REG_JOB_DEF_HELPER]"

def split_ext(filename: str) -> Tuple[str, str]:
    if "." not in filename:
        return filename, ""
    stem, ext = filename.rsplit(".", 1)
    return stem, ext.lower()


def copy_objects_or_raise(copy_plan: List[Tuple[str, str, str, str]]) -> None:
    """
    copy_plan: list of (src_bucket, src_key, dst_bucket, dst_key)
    """
    for src_bucket, src_key, dst_bucket, dst_key in copy_plan:
        s3_copy_with_retry(src_bucket, src_key, dst_bucket, dst_key, TASK_NAME)


def build_canonical_image_dest(file_bucket: str,
                               image_id: str,
                               temp_image_uri: str,
                               data_source: str,
                               path_prefix: str,
                               is_video: bool) -> Tuple[str, str]:
    _, temp_key = parse_s3_uri(temp_image_uri, TASK_NAME)
    fname = get_key_basename(temp_key)
    _, ext = split_ext(fname)
    if ext not in ("png", "jpg", "jpeg"):
        raise RuntimeError(f"{TASK_NAME} Unsupported or missing image extension for temp_source_ref={temp_image_uri}")
    category = 'videos' if is_video else 'non-videos'
    dst_key = f"canonical/{category}/{data_source}/{path_prefix}/{image_id}.{ext}"
    return dst_key, make_s3_uri(file_bucket, dst_key)


def build_canonical_label_dests_by_fingerprint(
    *,
    file_bucket: str,
    label_type: str,
    fingerprint: str,
    temp_bbox_meta_uri: Optional[str],
    temp_semantic_png_uri: Optional[str],
    temp_semantic_meta_uri: Optional[str],
    temp_instance_png_uri: Optional[str],
    temp_instance_meta_uri: Optional[str],
) -> Tuple[List[str], List[str], List[Tuple[str, str, str, str]]]:
    """
    Returns:
      - dst_keys
      - dst_uris
      - copy_plan: list of (src_bucket, src_key, dst_bucket, dst_key)
    """
    dst_keys: List[str] = []
    dst_uris: List[str] = []
    copy_plan: List[Tuple[str, str, str, str]] = []

    if label_type == "object-detection":
        if not temp_bbox_meta_uri:
            raise RuntimeError(f"{TASK_NAME} Missing temp_source_ref_bbox_meta for object-detection")
        src_b, src_k = parse_s3_uri(temp_bbox_meta_uri, TASK_NAME)

        dst_key = f"canonical/bounding-boxes/{fingerprint}.json"
        dst_keys.append(dst_key)
        dst_uris.append(make_s3_uri(file_bucket, dst_key))
        copy_plan.append((src_b, src_k, file_bucket, dst_key))
        return dst_keys, dst_uris, copy_plan

    if label_type == "semantic-segmentation":
        if not (temp_semantic_png_uri and temp_semantic_meta_uri):
            raise RuntimeError(f"{TASK_NAME} Missing semantic png/meta temp refs for semantic-segmentation")

        src_b1, src_k1 = parse_s3_uri(temp_semantic_png_uri, TASK_NAME)
        src_b2, src_k2 = parse_s3_uri(temp_semantic_meta_uri, TASK_NAME)

        dst_png = f"canonical/semantic-masks/{fingerprint}.png"
        dst_meta = f"canonical/semantic-masks/{fingerprint}.json"
        dst_keys.extend([dst_png, dst_meta])
        dst_uris.extend([make_s3_uri(file_bucket, dst_png), make_s3_uri(file_bucket, dst_meta)])

        copy_plan.append((src_b1, src_k1, file_bucket, dst_png))
        copy_plan.append((src_b2, src_k2, file_bucket, dst_meta))
        return dst_keys, dst_uris, copy_plan

    if label_type == "instance-segmentation":
        if not (temp_instance_png_uri and temp_instance_meta_uri):
            raise RuntimeError(f"{TASK_NAME} Missing instance png/meta temp refs for instance-segmentation")

        src_b1, src_k1 = parse_s3_uri(temp_instance_png_uri, TASK_NAME)
        src_b2, src_k2 = parse_s3_uri(temp_instance_meta_uri, TASK_NAME)

        dst_png = f"canonical/instance-annotations/{fingerprint}.png"
        dst_meta = f"canonical/instance-annotations/{fingerprint}.json"
        dst_keys.extend([dst_png, dst_meta])
        dst_uris.extend([make_s3_uri(file_bucket, dst_png), make_s3_uri(file_bucket, dst_meta)])

        copy_plan.append((src_b1, src_k1, file_bucket, dst_png))
        copy_plan.append((src_b2, src_k2, file_bucket, dst_meta))
        return dst_keys, dst_uris, copy_plan

    # single-label / multi-label: no label files
    return [], [], []


def build_canonical_imagery_row(*, row: Dict[str, Any], canonical_image_uri: str, registration_time: str) -> Dict[str, Any]:
    image_id = row.get("image_id")
    if not image_id:
        raise RuntimeError(f"{TASK_NAME} Missing image_id in upload_staging row")

    return {
        "image_id": image_id,
        "source_ref": canonical_image_uri,
        "img_type": row.get("img_type"),
        "img_height": row.get("img_height"),
        "img_width": row.get("img_width"),
        "num_channels": row.get("num_channels"),
        "dtype": row.get("dtype"),
        "file_size_mb": row.get("file_size_mb"),
        "uploaded_at": registration_time,
        "data_source": row.get("data_source"),
        "sha256_hash": row.get("sha256_hash"),
        "luma_mean": row.get("luma_mean"),
        "luma_p10": row.get("luma_p10"),
        "luma_p90": row.get("luma_p90"),
        "dark_frac": row.get("dark_frac"),
        "bright_frac": row.get("bright_frac"),
        "contrast_luma_std": row.get("contrast_luma_std"),
        "contrast_luma_p90_p10": row.get("contrast_luma_p90_p10"),
        "blur_laplacian_var": row.get("blur_laplacian_var"),
        "sat_mean": row.get("sat_mean"),
        "colorfulness": row.get("colorfulness"),
        "lighting_bucket": row.get("lighting_bucket"),
        "blur_bucket": row.get("blur_bucket"),
        "contrast_bucket": row.get("contrast_bucket"),
        "color_bucket": row.get("color_bucket")
    }


def build_canonical_label_table_row(
    *,
    label_type: str,
    fingerprint: str,
    canonical_label_uris: List[str],
    classes_present: Optional[List[str]],
) -> Optional[Dict[str, Any]]:
    """
    Returns a dict with routing field __table plus the correct columns for that table.
    NOTE: model A => no image_id column in canonical label tables.
    """
    if label_type == "object-detection":
        if len(canonical_label_uris) != 1:
            raise RuntimeError(f"{TASK_NAME} object-detection expected exactly 1 canonical label uri")
        return {
            "__table": "canonical_bounding_boxes",
            "bbox_annotation_id": fingerprint,
            "source_ref_meta": canonical_label_uris[0],
            "classes_present": classes_present,
        }

    if label_type == "semantic-segmentation":
        if len(canonical_label_uris) != 2:
            raise RuntimeError(f"{TASK_NAME} semantic-segmentation expected exactly 2 canonical label uris")
        png_uri = next(u for u in canonical_label_uris if u.endswith(".png"))
        meta_uri = next(u for u in canonical_label_uris if u.endswith(".json"))
        return {
            "__table": "canonical_semantic_masks",
            "semantic_mask_id": fingerprint,
            "source_ref_png": png_uri,
            "source_ref_meta": meta_uri,
            "classes_present": classes_present,
        }

    if label_type == "instance-segmentation":
        if len(canonical_label_uris) != 2:
            raise RuntimeError(f"{TASK_NAME} instance-segmentation expected exactly 2 canonical label uris")
        png_uri = next(u for u in canonical_label_uris if u.endswith(".png"))
        meta_uri = next(u for u in canonical_label_uris if u.endswith(".json"))
        return {
            "__table": "canonical_instance_annotations",
            "instance_annotation_id": fingerprint,
            "source_ref_png": png_uri,
            "source_ref_meta": meta_uri,
            "classes_present": classes_present,
        }

    return None


def build_image_label_rows(
    *,
    job_label_type: str,
    target_image_id: str,
    string_labels: Optional[List[str]],
    fingerprint: Optional[str],
) -> List[Dict[str, str]]:
    """
    image_labels schema: (image_id, label_id, label_type)
    - For single-label and multi-label jobs, label_type MUST be "string-label" and label_id is the lowercase string label.
    - For OD/semantic/instance jobs, label_type is the job_label_type and label_id is fingerprint.
    """
    out: List[Dict[str, str]] = []

    if job_label_type in ("single-label", "multi-label"):
        if not isinstance(string_labels, list) or not string_labels:
            return []
        for lab in string_labels:
            if isinstance(lab, str) and lab.strip():
                out.append(
                    {"image_id": target_image_id, "label_id": lab.strip().lower(), "label_type": "string-label"}
                )
        return out

    if job_label_type in ("object-detection", "semantic-segmentation", "instance-segmentation"):
        if not fingerprint:
            return []
        out.append({"image_id": target_image_id, "label_id": fingerprint, "label_type": job_label_type})
        return out

    return []


def fingerprint_owner_shard_id(fingerprint: str, num_shards: int) -> str:
    """
    Deterministically map a fingerprint -> owner shard id.

    - Prefers treating fingerprint as hex (sha256) and using the first 8 hex chars.
    - Falls back to md5 for non-hex fingerprints.
    Returns a zero-padded 6-char string like "000123".
    """
    if not isinstance(num_shards, int) or num_shards <= 0:
        raise ValueError(f"{TASK_NAME} num_shards must be a positive int, got {num_shards}")

    fp = (fingerprint or "").strip()
    if not fp:
        raise RuntimeError(f"{TASK_NAME} fingerprint is empty")

    try:
        # sha256 hex path
        v = int(fp[:8], 16)
    except Exception:
        # fallback path
        v = int(hashlib.md5(fp.encode("utf-8")).hexdigest()[:8], 16)

    shard = v % num_shards
    return str(shard).rjust(6, "0")


def build_owner_label_output_key(*, processed_prefix: str, owner_shard_id: str, source_target_shard: str) -> str:
    """
    Where the worker writes canonical label table rows routed by fingerprint-owner shard.

    Many workers can write files under the same owner shard prefix.
    Pre-ingest should group by owner_shard_id and treat the union as one logical shard.
    """
    owner = (owner_shard_id or "").strip()
    if not owner:
        raise RuntimeError(f"{TASK_NAME} owner_shard_id is empty")
    src = (source_target_shard or "shard").strip()
    return f"{processed_prefix}/canonical_labels_by_fingerprint/owner-{owner}/part-{src}.jsonl"
