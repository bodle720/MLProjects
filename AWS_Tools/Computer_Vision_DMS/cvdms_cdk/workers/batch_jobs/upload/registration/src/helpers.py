import math
import time
from datetime import datetime, date
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import boto3
from botocore.exceptions import ClientError

s3 = boto3.client("s3")

def to_jsonable(v: Any) -> Any:
    if v is None:
        return None

    # pyarrow scalar -> python value
    if hasattr(v, "as_py"):
        v = v.as_py()

    # pandas Timestamp-like
    if hasattr(v, "to_pydatetime"):
        v = v.to_pydatetime()

    if isinstance(v, datetime):
        # store as naive "YYYY-MM-DD HH:MM:SS" for your Athena insert helper
        return v.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")

    if isinstance(v, date):
        return v.isoformat()

    if isinstance(v, Decimal):
        f = float(v)
        return None if not math.isfinite(f) else f

    if isinstance(v, float):
        return None if not math.isfinite(v) else v

    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", errors="replace")

    return v

def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in list(row.items()):
        row[k] = to_jsonable(v)
    return row

def parse_s3_uri(uri: str) -> Tuple[str, str]:
    # expects s3://bucket/key
    if not uri or not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {uri!r}")
    bucket_key = uri[len("s3://"):]
    bucket, key = bucket_key.split("/", 1)
    return bucket, key

def s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"

def s3_key_basename(key: str) -> str:
    return key.rsplit("/", 1)[-1]

def split_ext(filename: str) -> Tuple[str, str]:
    # returns (stem, ext_without_dot) or (filename, "")
    if "." not in filename:
        return filename, ""
    stem, ext = filename.rsplit(".", 1)
    return stem, ext.lower()

def s3_copy_with_retry(src_bucket: str, src_key: str, dst_bucket: str, dst_key: str,
                       retries: int = 6, base_delay: float = 0.5) -> None:
    last_err = None
    for attempt in range(retries):
        try:
            s3.copy_object(
                Bucket=dst_bucket,
                Key=dst_key,
                CopySource={"Bucket": src_bucket, "Key": src_key},
            )
            return
        except ClientError as e:
            last_err = e
            code = e.response.get("Error", {}).get("Code", "")
            # fail fast on auth-ish issues
            if code in ("AccessDenied", "AccessDeniedException", "InvalidAccessKeyId", "SignatureDoesNotMatch"):
                raise
            time.sleep(min(base_delay * (2 ** attempt), 5.0))
    raise RuntimeError(f"S3 copy failed after retries: s3://{src_bucket}/{src_key} -> s3://{dst_bucket}/{dst_key}: {last_err}")

def s3_delete_best_effort(bucket: str, key: str) -> None:
    try:
        s3.delete_object(Bucket=bucket, Key=key)
    except Exception:
        # best effort: swallow
        pass

def build_canonical_image_dest(file_bucket: str, data_source: str, image_id: str, temp_image_uri: str) -> Tuple[str, str]:
    _, temp_key = parse_s3_uri(temp_image_uri)
    fname = s3_key_basename(temp_key)
    _, ext = split_ext(fname)
    if ext not in ("png", "jpg", "jpeg"):
        raise RuntimeError(f"Unsupported or missing image extension for temp_source_ref={temp_image_uri}")
    dst_key = f"canonical/imagery/{data_source}/{image_id}.{ext}"
    return dst_key, s3_uri(file_bucket, dst_key)

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
            raise RuntimeError("Missing temp_source_ref_bbox_meta for object-detection")
        _, src_key = parse_s3_uri(bbox_meta_uri)
        stem, ext = split_ext(s3_key_basename(src_key))
        if ext != "json":
            raise RuntimeError(f"Expected .json bbox meta, got {bbox_meta_uri}")
        label_uuid = stem
        dst_key = f"canonical/labels/object-detection/{label_uuid}.json"
        dst_keys.append(dst_key)
        dst_uris.append(s3_uri(file_bucket, dst_key))
        return dst_keys, dst_uris, label_uuid

    if label_type == "semantic-segmentation":
        if not (semantic_png_uri and semantic_meta_uri):
            raise RuntimeError("Missing semantic png/meta temp refs for semantic-segmentation")
        _, png_key = parse_s3_uri(semantic_png_uri)
        _, meta_key = parse_s3_uri(semantic_meta_uri)
        png_stem, png_ext = split_ext(s3_key_basename(png_key))
        meta_stem, meta_ext = split_ext(s3_key_basename(meta_key))
        if png_ext != "png" or meta_ext != "json":
            raise RuntimeError(f"Expected semantic .png and .json, got png={semantic_png_uri}, meta={semantic_meta_uri}")
        if png_stem != meta_stem:
            raise RuntimeError(f"Semantic png/meta label UUID mismatch: {png_stem} vs {meta_stem}")
        label_uuid = png_stem
        dst_png = f"canonical/labels/semantic-segmentation/{label_uuid}.png"
        dst_meta = f"canonical/labels/semantic-segmentation/{label_uuid}.json"
        dst_keys.extend([dst_png, dst_meta])
        dst_uris.extend([s3_uri(file_bucket, dst_png), s3_uri(file_bucket, dst_meta)])
        return dst_keys, dst_uris, label_uuid

    if label_type == "instance-segmentation":
        if not (instance_png_uri and instance_meta_uri):
            raise RuntimeError("Missing instance png/meta temp refs for instance-segmentation")
        _, png_key = parse_s3_uri(instance_png_uri)
        _, meta_key = parse_s3_uri(instance_meta_uri)
        png_stem, png_ext = split_ext(s3_key_basename(png_key))
        meta_stem, meta_ext = split_ext(s3_key_basename(meta_key))
        if png_ext != "png" or meta_ext != "json":
            raise RuntimeError(f"Expected instance .png and .json, got png={instance_png_uri}, meta={instance_meta_uri}")
        if png_stem != meta_stem:
            raise RuntimeError(f"Instance png/meta label UUID mismatch: {png_stem} vs {meta_stem}")
        label_uuid = png_stem
        dst_png = f"canonical/labels/instance-segmentation/{label_uuid}.png"
        dst_meta = f"canonical/labels/instance-segmentation/{label_uuid}.json"
        dst_keys.extend([dst_png, dst_meta])
        dst_uris.extend([s3_uri(file_bucket, dst_png), s3_uri(file_bucket, dst_meta)])
        return dst_keys, dst_uris, label_uuid

    # single-label / multi-label: no label files
    return [], [], None

def copy_objects_or_raise(copy_plan: List[Tuple[str, str, str, str]]) -> None:
    """
    copy_plan: list of (src_bucket, src_key, dst_bucket, dst_key)
    """
    for src_bucket, src_key, dst_bucket, dst_key in copy_plan:
        s3_copy_with_retry(src_bucket, src_key, dst_bucket, dst_key)

def cleanup_copied_best_effort(dst_bucket: str, dst_keys: List[str]) -> None:
    for k in dst_keys:
        s3_delete_best_effort(dst_bucket, k)

def build_canonical_imagery_row(data_source: str,
                                row: Dict[str, Any],
                                canonical_image_uri: str,
                                label_type: str,
                                registration_time: str,
                                label_uuid: Optional[str]) -> Dict[str, Any]:
    image_id = row.get("image_id")
    if not image_id:
        raise RuntimeError("Missing image_id in upload_staging row")

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
        raise RuntimeError(f"Unknown label_type: {label_type}")

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
            raise RuntimeError("object-detection expected exactly 1 canonical label uri")
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
            raise RuntimeError("semantic-segmentation expected exactly 2 canonical label uris")
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
            raise RuntimeError("instance-segmentation expected exactly 2 canonical label uris")
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