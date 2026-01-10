from typing import Any, Dict, List, Optional, Tuple

from common.s3_utils import make_s3_uri, parse_s3_uri, get_key_basename, s3_copy_with_retry, s3_delete_best_effort

TASK_NAME = "[REG_JOB_DEF_HELPER]"

def split_ext(filename: str) -> Tuple[str, str]:
    # returns (stem, ext_without_dot) or (filename, "")
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

def cleanup_copied_best_effort(dst_bucket: str, dst_keys: List[str]) -> None:
    for k in dst_keys:
        s3_delete_best_effort(dst_bucket, k)

def build_canonical_image_dest(file_bucket: str, data_source: str, image_id: str, temp_image_uri: str) -> Tuple[str, str]:
    _, temp_key = parse_s3_uri(temp_image_uri, TASK_NAME)
    fname = get_key_basename(temp_key)
    _, ext = split_ext(fname)
    if ext not in ("png", "jpg", "jpeg"):
        raise RuntimeError(f"{TASK_NAME} Unsupported or missing image extension for temp_source_ref={temp_image_uri}")
    dst_key = f"canonical/imagery/{data_source}/{image_id}.{ext}"
    return dst_key, make_s3_uri(file_bucket, dst_key)

def build_canonical_label_dests(file_bucket: str,
                               label_type: str,
                               bbox_meta_uri: Optional[str],
                               semantic_png_uri: Optional[str],
                               semantic_meta_uri: Optional[str],
                               instance_png_uri: Optional[str],
                               instance_meta_uri: Optional[str]) -> Tuple[List[str], List[str], Optional[str]]:
    """
    Returns: (dst_keys, dst_uris, label_uuid)
    """
    dst_keys: List[str] = []
    dst_uris: List[str] = []
    label_uuid: Optional[str] = None

    if label_type == "object-detection":
        if not bbox_meta_uri:
            raise RuntimeError(f"{TASK_NAME} Missing temp_source_ref_bbox_meta for object-detection")
        _, src_key = parse_s3_uri(bbox_meta_uri, TASK_NAME)
        stem, ext = split_ext(get_key_basename(src_key))
        if ext != "json":
            raise RuntimeError(f"{TASK_NAME} Expected .json bbox meta, got {bbox_meta_uri}")
        label_uuid = stem
        dst_key = f"canonical/labels/object-detection/{label_uuid}.json"
        dst_keys.append(dst_key)
        dst_uris.append(make_s3_uri(file_bucket, dst_key))
        return dst_keys, dst_uris, label_uuid

    if label_type == "semantic-segmentation":
        if not (semantic_png_uri and semantic_meta_uri):
            raise RuntimeError(f"{TASK_NAME} Missing semantic png/meta temp refs for semantic-segmentation")
        _, png_key = parse_s3_uri(semantic_png_uri, TASK_NAME)
        _, meta_key = parse_s3_uri(semantic_meta_uri, TASK_NAME)
        png_stem, png_ext = split_ext(get_key_basename(png_key))
        meta_stem, meta_ext = split_ext(get_key_basename(meta_key))
        if png_ext != "png" or meta_ext != "json":
            raise RuntimeError(f"{TASK_NAME} Expected semantic .png and .json, got png={semantic_png_uri}, meta={semantic_meta_uri}")
        if png_stem != meta_stem:
            raise RuntimeError(f"{TASK_NAME} Semantic png/meta label UUID mismatch: {png_stem} vs {meta_stem}")
        label_uuid = png_stem
        dst_png = f"canonical/labels/semantic-segmentation/{label_uuid}.png"
        dst_meta = f"canonical/labels/semantic-segmentation/{label_uuid}.json"
        dst_keys.extend([dst_png, dst_meta])
        dst_uris.extend([make_s3_uri(file_bucket, dst_png), make_s3_uri(file_bucket, dst_meta)])
        return dst_keys, dst_uris, label_uuid

    if label_type == "instance-segmentation":
        if not (instance_png_uri and instance_meta_uri):
            raise RuntimeError(f"{TASK_NAME} Missing instance png/meta temp refs for instance-segmentation")
        _, png_key = parse_s3_uri(instance_png_uri, TASK_NAME)
        _, meta_key = parse_s3_uri(instance_meta_uri, TASK_NAME)
        png_stem, png_ext = split_ext(get_key_basename(png_key))
        meta_stem, meta_ext = split_ext(get_key_basename(meta_key))
        if png_ext != "png" or meta_ext != "json":
            raise RuntimeError(f"{TASK_NAME} Expected instance .png and .json, got png={instance_png_uri}, meta={instance_meta_uri}")
        if png_stem != meta_stem:
            raise RuntimeError(f"{TASK_NAME} Instance png/meta label UUID mismatch: {png_stem} vs {meta_stem}")
        label_uuid = png_stem
        dst_png = f"canonical/labels/instance-segmentation/{label_uuid}.png"
        dst_meta = f"canonical/labels/instance-segmentation/{label_uuid}.json"
        dst_keys.extend([dst_png, dst_meta])
        dst_uris.extend([make_s3_uri(file_bucket, dst_png), make_s3_uri(file_bucket, dst_meta)])
        return dst_keys, dst_uris, label_uuid

    # single-label / multi-label: no label files
    return [], [], None

def build_canonical_imagery_row(data_source: str,
                                row: Dict[str, Any],
                                canonical_image_uri: str,
                                label_type: str,
                                registration_time: str,
                                label_uuid: Optional[str]) -> Dict[str, Any]:
    image_id = row.get("image_id")
    if not image_id:
        raise RuntimeError(f"{TASK_NAME} Missing image_id in upload_staging row")

    out = {
        "image_id": image_id,
        "source_ref": canonical_image_uri,
        "img_type": row.get("img_type"),
        "img_height": row.get("img_height"),
        "img_width": row.get("img_width"),
        "num_channels": row.get("num_channels"),
        "dtype": row.get("dtype"),
        "file_size_mb": row.get("file_size_mb"),
        "uploaded_at": registration_time,
        "data_source": data_source,
        "sha256_hash": row.get("sha256_hash"),
        "string_labels": None,
        "bbox_annotation_ids": None,
        "semantic_mask_ids": None,
        "instance_annotation_ids": None,
    }

    if label_type in ("single-label", "multi-label"):
        out["string_labels"] = row.get("string_labels")
    elif label_type == "object-detection":
        out["bbox_annotation_ids"] = [label_uuid] if label_uuid else None
    elif label_type == "semantic-segmentation":
        out["semantic_mask_ids"] = [label_uuid] if label_uuid else None
    elif label_type == "instance-segmentation":
        out["instance_annotation_ids"] = [label_uuid] if label_uuid else None
    else:
        raise RuntimeError(f"{TASK_NAME} Unknown label_type: {label_type}")

    return out

def build_label_table_row(file_bucket: str,
                          label_type: str,
                          image_id: str,
                          label_uuid: str,
                          canonical_label_uris: List[str],
                          classes_present: Optional[List[str]]) -> Optional[Dict[str, Any]]:
    """
    Returns a dict with a routing field __table plus the correct columns for that table.
    """
    if label_type == "object-detection":
        # only one uri (json)
        if len(canonical_label_uris) != 1:
            raise RuntimeError(f"{TASK_NAME} object-detection expected exactly 1 canonical label uri")
        return {
            "__table": "canonical_bounding_boxes",
            "bbox_annotation_id": label_uuid,
            "image_id": image_id,
            "source_ref_meta": canonical_label_uris[0],
            "classes_present": classes_present,
        }

    if label_type == "semantic-segmentation":
        # two uris: png + json
        if len(canonical_label_uris) != 2:
            raise RuntimeError(f"{TASK_NAME} semantic-segmentation expected exactly 2 canonical label uris")
        png_uri = next(u for u in canonical_label_uris if u.endswith(".png"))
        meta_uri = next(u for u in canonical_label_uris if u.endswith(".json"))
        return {
            "__table": "canonical_semantic_masks",
            "semantic_mask_id": label_uuid,
            "image_id": image_id,
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
            "instance_annotation_id": label_uuid,
            "image_id": image_id,
            "source_ref_png": png_uri,
            "source_ref_meta": meta_uri,
            "classes_present": classes_present,
        }

    return None