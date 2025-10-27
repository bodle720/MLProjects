# -*- coding: utf-8 -*-
"""
Splitting logic

the stratified train/val/test manifest generator (split.py with scikit‑learn, pandas).
"""

import argparse
import os
import json
import boto3
import pandas as pd
from sklearn.model_selection import train_test_split
from io import StringIO

s3 = boto3.client("s3")

def write_s3(bucket, key, body):
    s3.put_object(Bucket=bucket, Key=key, Body=body.encode("utf-8"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", required=True)
    parser.add_argument("--task_type", required=True, choices=["classification", "bbox"])
    parser.add_argument("--bucket", required=True)
    args = parser.parse_args()

    dataset_id = args.dataset_id
    task_type = args.task_type
    bucket = args.bucket

    # --- Step 1: Load features/labels ---
    # For demo, we simulate with a DataFrame. In practice, query Athena or load from S3.
    # Example schema: uuid, image_s3_uri, label
    data = pd.DataFrame({
        "uuid": [f"img{i}" for i in range(100)],
        "image_s3_uri": [f"s3://{bucket}/images/img{i}.jpg" for i in range(100)],
        "label": ["cat" if i % 2 == 0 else "dog" for i in range(100)]
    })

    # --- Step 2: Stratified split ---
    train, temp = train_test_split(data, test_size=0.3, stratify=data["label"], random_state=42)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp["label"], random_state=42)

    splits = {"train": train, "val": val, "test": test}

    # --- Step 3: Write manifests ---
    base_prefix = f"dataset-manifests/{dataset_id}/"

    # JSON Lines manifest
    json_lines = []
    for split_name, df in splits.items():
        for _, row in df.iterrows():
            if task_type == "classification":
                entry = {
                    "source-ref": row["image_s3_uri"],
                    "label": row["label"],
                    "split": split_name
                }
            elif task_type == "bbox":
                # Example bbox entry (replace with real bbox schema)
                entry = {
                    "source-ref": row["image_s3_uri"],
                    "bounding-box": {"annotations": []},
                    "split": split_name
                }
            json_lines.append(json.dumps(entry))

    write_s3(bucket, base_prefix + "manifest.manifest", "\n".join(json_lines))

    # CSV manifest
    csv_buf = StringIO()
    all_rows = pd.concat([df.assign(split=split) for split, df in splits.items()])
    all_rows.to_csv(csv_buf, index=False)
    write_s3(bucket, base_prefix + "manifest.csv", csv_buf.getvalue())

    print(f"✅ Wrote manifests to s3://{bucket}/{base_prefix}")

if __name__ == "__main__":
    main()
