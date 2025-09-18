# -*- coding: utf-8 -*-
"""
Worker function for the NormalizeLambda function.

Description of behavior:
    It may be the case earlier steps in the state machine were skipped, and we 
    just wanted to normalize an image. In that case, s3_raw_uri exists, but 
    s3_output_uri does not.
    Check and make sure s3_raw_uri always exists.
    If s3_output_uri does not exist, load in s3_raw_uri and save output to s3_output_uri.
    if s3_output_uri exists, load in s3_output_uri and save to s3_output_uri, because it's
    assumed to be the next step in a pipeline.
    This keeps the Lambda modular and reusable in different applications (ad hoc or part of a state machine workflow)
"""

import logging
import boto3
import numpy as np

from helpers.utils import parse_s3_uri, get_extension, load_numpy_from_s3, \
                          load_png_jpg_jpeg_image_from_s3, upload_numpy_to_s3

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
        - The normalized image is saved to s3_output_uri.
        - Only .png, .jpg, .jpeg and .npy extensions are allowed (case-insensitive).
        - Supports grayscale (H,W), single-band (H,W,1), and RGB (H,W,3) image formats.
        - Uses min-max normalization for both modes, without assuming 0–255 input range.

    Parameters in `event`:
        - s3_raw_uri (str): S3 URI of the original image (required and must exist)
        - s3_output_uri (str): S3 URI to save the normalized image (required)
        - normalization (str): '0to1' or '0to255' (required)

    Returns:
        None. Image is saved to S3.
    """

    s3_client = boto3.client('s3')

    s3_raw_uri = event.get('s3_raw_uri')
    s3_output_uri = event.get('s3_output_uri')
    normalization = event.get('normalization')

    if not s3_raw_uri or not s3_output_uri:
        raise ValueError("[NormalizeLambda]: Both 's3_raw_uri' and 's3_output_uri' are required.")

    if normalization not in ['0to1', '0to255']:
        raise ValueError(f"[NormalizeLambda]: Invalid normalization value: {normalization}")

    # Check if s3_output_uri exists
    out_bucket, out_key = parse_s3_uri(s3_output_uri)
    output_ext = get_extension(out_key) # must be '.npy'
    
    if output_ext != '.npy':
        raise ValueError(f"[NormalizeLambda]: Invalid output extension, must be .npy, not: {output_ext}")

    try:
        s3_client.head_object(Bucket=out_bucket, Key=out_key)
        input_uri = s3_output_uri
        logger.info(f"[NormalizeLambda]: Using existing output image as input: {input_uri}")
    except s3_client.exceptions.ClientError:
        input_uri = s3_raw_uri
        logger.info(f"[NormalizeLambda]: Output image not found. Using raw image as input: {input_uri}")

    in_bucket, in_key = parse_s3_uri(input_uri)
    ext = get_extension(in_key) # ext is in ['.png', '.jpg', '.jpeg', '.npy']
    
    try:
        s3_client.get_object(Bucket=in_bucket, Key=in_key)
    except s3_client.exceptions.NoSuchKey:
        raise FileNotFoundError(f"[NormalizeLambda]: Input image not found in S3: {input_uri}")


    if ext in ['.png', '.jpg', '.jpeg']:
        image_array = load_png_jpg_jpeg_image_from_s3(s3_client, in_bucket, in_key)
    else:
        image_array = load_numpy_from_s3(s3_client, in_bucket, in_key)
        
    H, W = image_array.shape[:2]
    
    # Log image dimensions.
    if image_array.ndim == 2:
        logger.info(f"[NormalizeLambda]: Detected grayscale image ({H}, {W})")
    elif image_array.ndim == 3 and image_array.shape[2] == 1:
        logger.info(f"[NormalizeLambda]: Detected single-band image ({H}, {W}, 1)")
    elif image_array.ndim == 3 and image_array.shape[2] == 3:
        logger.info(f"[NormalizeLambda]: Detected RGB image ({H}, {W}, 3)")
    else:
        raise ValueError(f"[NormalizeLambda]: Unsupported image shape: {image_array.shape}")

    # Normalize using min-max scaling. image_array is either (H, W) or (H, W, 3)
    min_val = image_array.min()
    max_val = image_array.max()
    if max_val == min_val:
        raise ValueError(f"[NormalizeLambda]: Cannot normalize image with constant pixel values: min = max = {max_val}")

    scaled = (image_array.astype(np.float32) - min_val) / (max_val - min_val)

    if normalization == '0to1':
        save_array = scaled
        logger.info("[NormalizeLambda]: Normalized image to [0.0, 1.0] float range")
    else:  # '0to255'
        save_array = (scaled * 255).astype(np.uint8)
        logger.info("[NormalizeLambda]: Normalized image to [0, 255] uint8 range")

    # Save as .npy file to S3.
    upload_numpy_to_s3(s3_client, save_array, out_bucket, out_key)

    logger.info(f"[NormalizeLambda]: Normalized image saved to S3: {s3_output_uri}")