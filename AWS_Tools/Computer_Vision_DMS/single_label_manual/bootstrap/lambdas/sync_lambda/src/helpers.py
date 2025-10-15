# -*- coding: utf-8 -*-
"""
Sync helpers
"""

import os
import io
import json
import csv
import boto3
import numpy as np
import pandas as pd
from skimage.io import imread
from scipy.stats import skew, kurtosis
from skimage.feature import greycomatrix, greycoprops, canny
from skimage.filters import sobel
from sklearn.cluster import KMeans
from sklearn.preprocessing import KBinsDiscretizer
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

AWS_REGION = os.environ["AWS_REGION"]
S3_BUCKET_NAME = os.environ["S3_BUCKET_NAME"]
S3_DATASETS_ROOT = os.environ["S3_DATASETS_ROOT"]
DDB_IMAGERY_TABLE = os.environ["DDB_IMAGERY_TABLE"]

s3 = boto3.client("s3", region_name=AWS_REGION)
ddb = boto3.client("dynamodb", region_name=AWS_REGION)

# --- 1. Extract features from an imagery row ---
def extract_features_from_item(item):
    """Return features dict from a DynamoDB imagery row if present."""
    if not item:
        return None
    features_str = item.get("features", {}).get("S")
    if not features_str:
        return None
    try:
        return json.loads(features_str)
    except Exception:
        return None

# --- 2. Calculate and store features ---
def calculate_and_store_features(phash, dataset_id):
    """Download image, compute features, update imagery row, return features dict."""
    key = f"{S3_DATASETS_ROOT}/images/{phash}.png"
    try:
        resp = s3.get_object(Bucket=S3_BUCKET_NAME, Key=key)
        img_bytes = resp["Body"].read()
        # Save to temp file in memory
        img_path = f"/tmp/{phash}.png"
        with open(img_path, "wb") as f:
            f.write(img_bytes)

        features = extract_image_features(img_path)

        # Update DDB row
        dataset_phash = f"{dataset_id}#{phash}"
        ddb.update_item(
            TableName=DDB_IMAGERY_TABLE,
            Key={"dataset_phash": {"S": dataset_phash}},
            UpdateExpression="SET features = :f",
            ExpressionAttributeValues={":f": {"S": json.dumps(features)}}
        )
        return features
    except Exception as e:
        logger.error(f"[calculate_and_store_features] Failed for {phash}: {e}")
        raise

# --- Feature extraction logic (based on your script) ---
def extract_image_features(image_path):
    img = imread(image_path)
    if img.ndim == 2:
        bands = img[np.newaxis, ...]
    else:
        bands = np.transpose(img, (2, 0, 1))

    features = {}
    for b_idx, band in enumerate(bands, start=1):
        band = np.nan_to_num(band, copy=False)
        suffix = f"_b{b_idx}"

        features[f"mean{suffix}"]   = float(np.mean(band))
        features[f"std{suffix}"]    = float(np.std(band))
        features[f"min{suffix}"]    = float(np.min(band))
        features[f"max{suffix}"]    = float(np.max(band))
        features[f"median{suffix}"] = float(np.median(band))
        features[f"p25{suffix}"]    = float(np.percentile(band, 25))
        features[f"p75{suffix}"]    = float(np.percentile(band, 75))

        max_val = band.max() or 1
        band_q = np.clip((band / max_val) * 255, 0, 255).astype(np.uint8)

        glcm = greycomatrix(band_q, distances=[1], angles=[0], levels=256,
                            symmetric=True, normed=True)
        for prop in ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy']:
            features[f"{prop}{suffix}"] = float(greycoprops(glcm, prop)[0, 0])

        hist, _ = np.histogram(band_q, bins=256, range=(0, 255), density=True)
        hist += 1e-12
        features[f"hist_entropy{suffix}"] = float(-np.sum(hist * np.log2(hist)))

        flat = band_q.ravel()
        features[f"skewness{suffix}"] = float(skew(flat))
        features[f"kurtosis{suffix}"] = float(kurtosis(flat))

        grad_mag = sobel(band.astype(float))
        features[f"sobel_mean{suffix}"] = float(np.mean(grad_mag))
        features[f"sobel_std{suffix}"]  = float(np.std(grad_mag))

        edges = canny(band_q, sigma=1)
        features[f"edge_density{suffix}"] = float(edges.sum() / edges.size)

    return features

# --- 3. Assign splits ---
def assign_splits(enriched, splits={"train":0.7, "validation":0.15, "test":0.15}, n_clusters=20, n_bins=10, random_state=42):
    """Assign train/val/test splits balancing class and feature distributions."""
    df = pd.DataFrame(enriched)
    feature_cols = [c for c in df.columns if c.startswith(("mean","std","contrast","hist_entropy","sobel","edge_density"))]

    if not feature_cols:
        raise Exception("No feature columns available for split assignment.")

    km = KMeans(n_clusters=n_clusters, random_state=random_state)
    df['cluster_id'] = km.fit_predict(df[feature_cols])

    all_feats = feature_cols + ['cluster_id']
    bin_counts = [n_bins] * len(feature_cols) + [n_clusters]
    discretizer = KBinsDiscretizer(n_bins=bin_counts, encode='ordinal', strategy='quantile')
    binned = discretizer.fit_transform(df[all_feats]).astype(int)

    N = len(df)
    desired_total = {s: int(frac * N) for s, frac in splits.items()}
    classes = df['label'].unique()
    class_counts = df['label'].value_counts().to_dict()
    desired_class = {s: {c: int(class_counts[c] * frac) for c in classes} for s, frac in splits.items()}

    desired_bin = {}
    for s, frac in splits.items():
        desired_bin[s] = {}
        for feat_idx, feat in enumerate(all_feats):
            vals, counts = np.unique(binned[:, feat_idx], return_counts=True)
            per_bin = {int(val): int(count * frac) for val, count in zip(vals, counts)}
            desired_bin[s][feat] = per_bin

    current = {s: {'total':0, 'class':{c:0 for c in classes}, 'bin':{feat:{b:0 for b in desired_bin[s][feat]} for feat in all_feats}} for s in splits}

    rng = np.random.RandomState(random_state)
    indices = df.index.to_list()
    rng.shuffle(indices)
    assignment = {}

    for idx in indices:
        lbl = df.at[idx, 'label']
        row_bins = binned[df.index.get_indexer([idx])[0]]

        best_split, best_score = None, float('inf')
        for s in splits:
            score = abs((current[s]['total']+1) - desired_total[s])
            score += abs((current[s]['class'][lbl]+1) - desired_class[s][lbl])
            for feat_idx, feat in enumerate(all_feats):
                bin_val = int(row_bins[feat_idx])
                score += abs((current[s]['bin'][feat].get(bin_val,0)+1) - desired_bin[s][feat].get(bin_val,0))
            if score < best_score:
                best_score, best_split = score, s

        assignment[idx] = best_split
        current[best_split]['total'] += 1
        current[best_split]['class'][lbl] += 1
        for feat_idx, feat in enumerate(all_feats):
            bin_val = int(row_bins[feat_idx])
            current[best_split]['bin'][feat][bin_val] += 1

    df['split'] = df.index.map(assignment)
    return df.to_dict(orient="records")

# --- 4. Build CSV ---
def build_csv(enriched):
    """Convert enriched list of dicts into CSV string."""
    if not enriched:
        return ""

    # Collect all keys across all dicts to ensure wide schema
    fieldnames = sorted({k for row in enriched for k in row.keys()})

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in enriched:
        writer.writerow(row)

    return output.getvalue()

