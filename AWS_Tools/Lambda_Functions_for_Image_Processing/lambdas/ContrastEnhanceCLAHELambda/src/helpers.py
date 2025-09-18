# -*- coding: utf-8 -*-
"""
S3 I/O utilities for ContrastEnhanceClaheLambda.
"""

import os
from io import BytesIO
import numpy as np
from skimage import io as skio

def parse_s3_uri(uri):
    assert uri.startswith("s3://"), f"[ContrastEnhanceClaheLambda]: Invalid S3 URI: {uri}"
    parts = uri[5:].split("/", 1)
    return parts[0], parts[1]

def get_extension(key):
    ext = os.path.splitext(key)[1].lower()
    return ext

def load_numpy_from_s3(s3_client, bucket_name, object_key):
    """
    Downloads a .npy file from S3 and loads it into a NumPy array.
    Preserves original shape and data type.

    Parameters:
    - s3_client: boto3 S3 client
    - bucket_name: source S3 bucket name
    - object_key: S3 object key (e.g., 'folder/array.npy')

    Returns:
    - NumPy array with original shape and dtype
    """
    # Validate file extension
    if not object_key.endswith('.npy'):
        raise ValueError(f"Expected a .npy file, got: {object_key}")

    # Download object from S3
    response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
    buffer = BytesIO(response['Body'].read())

    # Load NumPy array
    array = np.load(buffer)

    return array

def load_png_jpg_jpeg_image_from_s3(s3_client, bucket_name, object_key):
    """
    Downloads a PNG or JPEG image from S3 and returns it as a NumPy array.

    Parameters:
    - s3_client: boto3 S3 client
    - bucket_name: source S3 bucket name
    - object_key: S3 object key (e.g., 'images/sample.png')

    Returns:
    - NumPy array representing the image
    """
    # Validate file extension
    valid_extensions = ('.png', '.jpg', '.jpeg')
    if not object_key.lower().endswith(valid_extensions):
        raise ValueError(f"Expected image file with one of {valid_extensions}, got: {object_key}")

    # Download image from S3
    response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
    buffer = BytesIO(response['Body'].read())

    # Load image using skimage.io
    image_array = skio.imread(buffer)
        
    return image_array

def upload_numpy_to_s3(s3_client, nparray, bucket_name, object_key):
    """
    Serializes a NumPy array and uploads it to S3 as a .npy file.
    Preserves shape and data type.
    
    Parameters:
    - s3_client: boto3 S3 client
    - nparray: NumPy array of any shape
    - bucket_name: target S3 bucket name
    - object_key: target S3 object key (e.g., 'folder/array.npy')
    """
    # Serialize array to in-memory buffer
    buffer = BytesIO()
    np.save(buffer, nparray)
    buffer.seek(0)

    # Upload to S3
    s3_client.put_object(
        Bucket=bucket_name,
        Key=object_key,
        Body=buffer,
        ContentType='application/octet-stream' # raw binary data 
    )