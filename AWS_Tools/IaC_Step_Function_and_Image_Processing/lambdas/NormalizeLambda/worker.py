# -*- coding: utf-8 -*-
"""
Worker function for the NormalizeLambda function.

Description of behavior:
    It may be the case earlier steps in the state machine were skipped, and we 
    just wanted to normalize an image. In that case, s3_raw_uri exists, but 
    s3_output_uri does not.
    Check and make sure s3_raw_urialways exists.
    If s3_output_uri does not exist, load in s3_raw_uri and save output to s3_output_uri.
    if s3_output_uri exists, load in s3_output_uri and save to s3_output_uri, because it's
    assumed to be the next step in a pipeline.
    Ths keeps the lambda modular and reusable in different applications (ad hoc or part of a state machine workflow)
"""

import io
import logging
import boto3
import numpy as np
from PIL import Image

from helpers.utils import parse_s3_uri, get_extension

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    """
    NormalizeLambda: Image normalization (global if RGB) Lambda function.

    This function normalizes an image loaded from S3 to either:
        - [0.0, 1.0] float range ('0to1')
        - [0, 255] uint8 range ('0to255')

    Behavior:
        - s3_raw_uri must exist in S3 and is used as the input if s3_output_uri does not exist.
        - If s3_output_uri exists in S3, it is used as the input instead of s3_raw_uri.
        - The normalized image is saved to s3_output_uri, preserving the original file extension.
        - Only .png, .jpg, .jpeg extensions are allowed (case-insensitive).
        - Supports grayscale (H,W), single-band (H,W,1), and RGB (H,W,3) image formats.
        - Uses min-max normalization for both modes, without assuming 0–255 input range.

    Parameters in `event`:
        - s3_raw_uri (str): S3 URI of the original image (required and must exist)
        - s3_output_uri (str): S3 URI to save the normalized image (required)
        - normalization (str): '0to1' or '0to255' (required)

    Returns:
        None. Image is saved to S3.
    """

    s3 = boto3.client('s3')

    s3_raw_uri = event.get('s3_raw_uri')
    s3_output_uri = event.get('s3_output_uri')
    normalization = event.get('normalization')

    if not s3_raw_uri or not s3_output_uri:
        raise ValueError("[NormalizeLambda]: Both 's3_raw_uri' and 's3_output_uri' are required.")

    if normalization not in ['0to1', '0to255']:
        raise ValueError(f"[NormalizeLambda]: Invalid normalization value: {normalization}")

    # Check if s3_output_uri exists
    out_bucket, out_key = parse_s3_uri(s3_output_uri)

    try:
        s3.head_object(Bucket=out_bucket, Key=out_key)
        input_uri = s3_output_uri
        logger.info(f"[NormalizeLambda]: Using existing output image as input: {input_uri}")
    except s3.exceptions.ClientError:
        input_uri = s3_raw_uri
        logger.info(f"[NormalizeLambda]: Output image not found. Using raw image as input: {input_uri}")

    in_bucket, in_key = parse_s3_uri(input_uri)
    ext = get_extension(in_key)

    try:
        response = s3.get_object(Bucket=in_bucket, Key=in_key)
    except s3.exceptions.NoSuchKey:
        raise FileNotFoundError(f"[NormalizeLambda]: Input image not found in S3: {input_uri}")

    image_bytes = response['Body'].read()
    image = Image.open(io.BytesIO(image_bytes))

    image_array = np.array(image)

    # Handle image shape
    if image_array.ndim == 2:
        logger.info("[NormalizeLambda]: Detected grayscale image (H, W)")
    elif image_array.ndim == 3 and image_array.shape[2] == 1:
        image_array = image_array.squeeze(-1)
        logger.info("[NormalizeLambda]: Detected single-band image (H, W, 1) → squeezed to (H, W)")
    elif image_array.ndim == 3 and image_array.shape[2] == 3:
        logger.info("[NormalizeLambda]: Detected RGB image (H, W, 3)")
    else:
        raise ValueError(f"[NormalizeLambda]: Unsupported image shape: {image_array.shape}")

    # Normalize using min-max scaling. image_array is either (H, W) or (H, W, 3)
    min_val = image_array.min()
    max_val = image_array.max()
    if max_val == min_val:
        raise ValueError("[NormalizeLambda]: Cannot normalize image with constant pixel values.")

    scaled = (image_array.astype(np.float32) - min_val) / (max_val - min_val)

    if normalization == '0to1':
        save_array = scaled
        logger.info("[NormalizeLambda]: Normalized image to [0.0, 1.0] float range")
    else:  # '0to255'
        save_array = (scaled * 255).astype(np.uint8)
        logger.info("[NormalizeLambda]: Normalized image to [0, 255] uint8 range")

    # Convert back to image
    output_image = Image.fromarray(save_array)
    output_buffer = io.BytesIO()
    output_image.save(output_buffer, format=ext.strip('.').upper())
    output_buffer.seek(0)

    s3.put_object(Bucket=out_bucket, Key=out_key, Body=output_buffer, ContentType=f"image/{ext.strip('.')}")
    logger.info(f"[NormalizeLambda]: Normalized image saved to S3: {s3_output_uri}")
