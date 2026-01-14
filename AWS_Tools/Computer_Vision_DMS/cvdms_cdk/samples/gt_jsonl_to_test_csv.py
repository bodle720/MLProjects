#!/usr/bin/env python3
"""
gt_jsonl_to_test_csv.py

For TESTING ONLY:
Convert an AWS Ground Truth-style JSONL/NDJSON/.manifest file (one JSON object per line)
into a CSV that matches the "expected CSV formats" we assumed for each label type.

Usage:
  python samples/gt_jsonl_to_test_csv.py --label-type single-label --in samples/single_label/single_label.jsonl --out samples/single_label/single_label.csv
  python samples/gt_jsonl_to_test_csv.py --label-type multi-label --in samples/multi_label/multi_label.jsonl --out samples/multi_label/multi_label.csv
  python samples/gt_jsonl_to_test_csv.py --label-type object-detection --in samples/object_detection/object_detection.jsonl --out samples/object_detection/object_detection.csv
  python samples/gt_jsonl_to_test_csv.py --label-type semantic-segmentation --in samples/semantic_segmentation/semantic_segmentation.jsonl --out samples/semantic_segmentation/semantic_segmentation.csv
  python samples/gt_jsonl_to_test_csv.py --label-type instance-segmentation --in samples/instance_segmentation/instance_segmentation.jsonl --out samples/instance_segmentation/instance_segmentation.csv

CSV schemas produced (as assumed):
  single-label:
    columns: source-ref, class-name

  multi-label:
    columns: source-ref, labels
    where labels is comma-separated class strings (deduped, sorted)

  object-detection:
    columns: source-ref, class-name, top, left, height, width
    one row per bounding box

  semantic-segmentation:
    columns: source-ref, semantic-segmentation-ref, color_map
    where color_map is JSON string: { "CLASS": ["#RRGGBB"], ... } (list-of-strings, one per class)

  instance-segmentation:
    columns: source-ref, worker-response-ref

Notes:
- This script expects the EXACT GT keys you specified:
  source-ref, single-label-metadata.class-name, multi-label + multi-label-metadata.class-map,
  object-detection.annotations + object-detection-metadata.class-map, semantic-segmentation-ref + internal-color-map,
  instance-segmentation-metadata.worker-response-ref
- It ignores empty/blank lines.
- It does NOT try to validate image existence; only structure.

"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


ALLOWED_LABEL_TYPES = {
    "single-label",
    "multi-label",
    "object-detection",
    "semantic-segmentation",
    "instance-segmentation",
}


def parse_json_object_line(line: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-type", required=True, choices=sorted(ALLOWED_LABEL_TYPES))
    ap.add_argument("--in", dest="in_path", required=True, help="Input GT JSONL/NDJSON/.manifest file")
    ap.add_argument("--out", dest="out_path", required=True, help="Output CSV path")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    if not in_path.exists() or not in_path.is_file():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    # Ensure output folder exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    label_type = args.label_type

    if label_type == "single-label":
        write_single_label_csv(in_path, out_path)
    elif label_type == "multi-label":
        write_multi_label_csv(in_path, out_path)
    elif label_type == "object-detection":
        write_object_detection_csv(in_path, out_path)
    elif label_type == "semantic-segmentation":
        write_semantic_seg_csv(in_path, out_path)
    elif label_type == "instance-segmentation":
        write_instance_seg_csv(in_path, out_path)
    else:
        raise ValueError(f"Unsupported label_type: {label_type}")

    print(f"Wrote CSV: {out_path}")


def write_single_label_csv(in_path: Path, out_path: Path) -> None:
    fieldnames = ["source-ref", "class-name"]
    with in_path.open("r", encoding="utf-8-sig") as fin, out_path.open("w", newline="", encoding="utf-8") as fout:
        w = csv.DictWriter(fout, fieldnames=fieldnames)
        w.writeheader()

        for lineno, raw in enumerate(fin, start=1):
            line = raw.strip()
            if not line:
                continue
            obj = parse_json_object_line(line)
            if obj is None:
                raise ValueError(f"Line {lineno}: invalid JSON object")

            src = obj.get("source-ref")
            meta = obj.get("single-label-metadata")
            if not isinstance(meta, dict):
                raise ValueError(f"Line {lineno}: missing/invalid single-label-metadata")
            cn = meta.get("class-name")
            if not isinstance(cn, str):
                raise ValueError(f"Line {lineno}: single-label-metadata.class-name must be string")

            w.writerow({"source-ref": src, "class-name": cn})


def write_multi_label_csv(in_path: Path, out_path: Path) -> None:
    fieldnames = ["source-ref", "labels"]
    with in_path.open("r", encoding="utf-8-sig") as fin, out_path.open("w", newline="", encoding="utf-8") as fout:
        w = csv.DictWriter(fout, fieldnames=fieldnames)
        w.writeheader()

        for lineno, raw in enumerate(fin, start=1):
            line = raw.strip()
            if not line:
                continue
            obj = parse_json_object_line(line)
            if obj is None:
                raise ValueError(f"Line {lineno}: invalid JSON object")

            src = obj.get("source-ref")
            ml = obj.get("multi-label")
            meta = obj.get("multi-label-metadata")

            if not isinstance(ml, list):
                raise ValueError(f"Line {lineno}: missing/invalid multi-label (must be list)")
            if not isinstance(meta, dict):
                raise ValueError(f"Line {lineno}: missing/invalid multi-label-metadata")
            class_map = meta.get("class-map")
            if not isinstance(class_map, dict):
                raise ValueError(f"Line {lineno}: missing/invalid multi-label-metadata.class-map")

            # Convert ids -> class strings using class-map
            labels: List[str] = []
            for idx, class_id in enumerate(ml):
                if not isinstance(class_id, int):
                    raise ValueError(f"Line {lineno}: multi-label[{idx}] must be int")
                key = str(class_id)
                if key not in class_map:
                    raise ValueError(f"Line {lineno}: class_id {class_id} not found in class-map")
                v = class_map[key]
                if not isinstance(v, str):
                    raise ValueError(f"Line {lineno}: class-map[{key}] must be str")
                v = v.strip()
                if v:
                    labels.append(v)

            # dedup + sort then join by comma
            labels_str = ",".join(sorted(set(labels)))
            w.writerow({"source-ref": src, "labels": labels_str})


def write_object_detection_csv(in_path: Path, out_path: Path) -> None:
    fieldnames = ["source-ref", "class-name", "top", "left", "height", "width"]
    with in_path.open("r", encoding="utf-8-sig") as fin, out_path.open("w", newline="", encoding="utf-8") as fout:
        w = csv.DictWriter(fout, fieldnames=fieldnames)
        w.writeheader()

        for lineno, raw in enumerate(fin, start=1):
            line = raw.strip()
            if not line:
                continue
            obj = parse_json_object_line(line)
            if obj is None:
                raise ValueError(f"Line {lineno}: invalid JSON object")

            src = obj.get("source-ref")
            od = obj.get("object-detection")
            meta = obj.get("object-detection-metadata")

            if not isinstance(od, dict):
                raise ValueError(f"Line {lineno}: missing/invalid object-detection")
            anns = od.get("annotations")
            if not isinstance(anns, list):
                raise ValueError(f"Line {lineno}: object-detection.annotations must be list")

            if not isinstance(meta, dict):
                raise ValueError(f"Line {lineno}: missing/invalid object-detection-metadata")
            class_map = meta.get("class-map")
            if not isinstance(class_map, dict):
                raise ValueError(f"Line {lineno}: object-detection-metadata.class-map must be dict")

            for i, ann in enumerate(anns):
                if not isinstance(ann, dict):
                    raise ValueError(f"Line {lineno}: annotation[{i}] must be dict")
                cid = ann.get("class_id")
                if not isinstance(cid, int):
                    raise ValueError(f"Line {lineno}: annotation[{i}].class_id must be int")
                cname = class_map.get(str(cid))
                if not isinstance(cname, str):
                    raise ValueError(f"Line {lineno}: class-map missing class_id {cid}")

                # Write one row per annotation
                w.writerow(
                    {
                        "source-ref": src,
                        "class-name": cname,
                        "top": ann.get("top", ""),
                        "left": ann.get("left", ""),
                        "height": ann.get("height", ""),
                        "width": ann.get("width", ""),
                    }
                )


def write_semantic_seg_csv(in_path: Path, out_path: Path) -> None:
    fieldnames = ["source-ref", "semantic-segmentation-ref", "color_map"]
    with in_path.open("r", encoding="utf-8-sig") as fin, out_path.open("w", newline="", encoding="utf-8") as fout:
        w = csv.DictWriter(fout, fieldnames=fieldnames)
        w.writeheader()

        for lineno, raw in enumerate(fin, start=1):
            line = raw.strip()
            if not line:
                continue
            obj = parse_json_object_line(line)
            if obj is None:
                raise ValueError(f"Line {lineno}: invalid JSON object")

            src = obj.get("source-ref")
            mask_ref = obj.get("semantic-segmentation-ref")
            meta = obj.get("semantic-segmentation-ref-metadata")
            if not isinstance(meta, dict):
                raise ValueError(f"Line {lineno}: missing/invalid semantic-segmentation-ref-metadata")
            icm = meta.get("internal-color-map")
            if not isinstance(icm, dict):
                raise ValueError(f"Line {lineno}: internal-color-map must be dict")

            # Build class_name -> ["#RRGGBB"] mapping (list-of-strings)
            cm: Dict[str, List[str]] = {}
            for _, v in icm.items():
                if not isinstance(v, dict):
                    continue
                cn = v.get("class-name")
                hc = v.get("hex-color")
                if isinstance(cn, str) and isinstance(hc, str):
                    cm[cn] = [hc]

            w.writerow(
                {
                    "source-ref": src,
                    "semantic-segmentation-ref": mask_ref,
                    "color_map": json.dumps(cm, separators=(",", ":"), ensure_ascii=False),
                }
            )


def write_instance_seg_csv(in_path: Path, out_path: Path) -> None:
    fieldnames = ["source-ref", "worker-response-ref"]
    with in_path.open("r", encoding="utf-8-sig") as fin, out_path.open("w", newline="", encoding="utf-8") as fout:
        w = csv.DictWriter(fout, fieldnames=fieldnames)
        w.writeheader()

        for lineno, raw in enumerate(fin, start=1):
            line = raw.strip()
            if not line:
                continue
            obj = parse_json_object_line(line)
            if obj is None:
                raise ValueError(f"Line {lineno}: invalid JSON object")

            src = obj.get("source-ref")
            meta = obj.get("instance-segmentation-metadata")
            if not isinstance(meta, dict):
                raise ValueError(f"Line {lineno}: missing/invalid instance-segmentation-metadata")
            wrr = meta.get("worker-response-ref")
            if not isinstance(wrr, str):
                raise ValueError(f"Line {lineno}: worker-response-ref must be string")

            w.writerow({"source-ref": src, "worker-response-ref": wrr})


if __name__ == "__main__":
    main()
