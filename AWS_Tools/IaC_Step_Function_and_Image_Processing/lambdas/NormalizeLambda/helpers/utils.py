# -*- coding: utf-8 -*-
"""
Helpers for normalizing an image.
"""

import os

def parse_s3_uri(uri):
    assert uri.startswith("s3://"), f"[NormalizeLambda]: Invalid S3 URI: {uri}"
    parts = uri[5:].split("/", 1)
    return parts[0], parts[1]

def get_extension(key):
    ext = os.path.splitext(key)[1].lower()
    if ext not in ['.png', '.jpg', '.jpeg']:
        raise ValueError(f"[NormalizeLambda]: Unsupported image extension: {ext}")
    return ext
