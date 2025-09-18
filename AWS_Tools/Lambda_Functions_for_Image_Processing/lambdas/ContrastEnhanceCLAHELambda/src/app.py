# -*- coding: utf-8 -*-
"""
This is the code to use the CLAHE technique for enhancing the contrast of an image.
"""

import boto3
import logging
import numpy as np
from skimage import exposure, color

from helpers import parse_s3_uri, get_extension, load_png_jpg_jpeg_image_from_s3, \
                    load_numpy_from_s3 , upload_numpy_to_s3

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    """
    ContrastEnhanceClaheLambda: Image contrast enhancement using CLAHE technique.
    This function enhances the contrast of an image loaded from S3.

    Behavior:
        - s3_input_uri must exist in S3 and is used as the input image data.
        - The enhanced image is saved to s3_output_uri, possibly overwriting existing data.
        - Only .png, .jpg, .jpeg and .npy extensions are allowed (case-insensitive).
        - Supports grayscale (H,W), single-band (H,W,1), and RGB (H,W,3) image formats.

    Parameters in `event`:
        - s3_input_uri (str): S3 URI of the original image (required and must exist)
        - s3_output_uri (str): S3 URI to save the normalized image to (required to be .npy file)
        - kernel_size: The kernel size of the filter. E.g., if 8, then kernel is an (8,8) grid.
        - clip_limit: A float 0 to 1 indicating the level of contrast.

    Returns:
        None. Image is saved to S3.
    """

    s3_client = boto3.client('s3')

    s3_input_uri = event.get('s3_input_uri')
    s3_output_uri = event.get('s3_output_uri')
    kernel_size = int(event.get('kernel_size'))
    clip_limit = float(event.get('clip_limit'))

    if not s3_input_uri or not s3_output_uri:
        raise ValueError("[ContrastEnhanceClaheLambda]: Both 's3_input_uri' and 's3_output_uri' are required.")

    if clip_limit <= 0 or clip_limit >= 1:
        raise ValueError(f"[ContrastEnhanceClaheLambda]: Invalid clip limit value, must be between 0 and 1: {clip_limit}")

    if kernel_size <= 0:
        raise ValueError(f"[ContrastEnhanceClaheLambda]: Invalid kernel_size value, must be positive: {kernel_size}")

    out_bucket, out_key = parse_s3_uri(s3_output_uri)
    output_ext = get_extension(out_key) # must be '.npy'
    
    if output_ext != '.npy':
        raise ValueError(f"[ContrastEnhanceClaheLambda]: Invalid output extension, must be .npy, not: {output_ext}")
 
    in_bucket, in_key = parse_s3_uri(s3_input_uri)
    
    try:
        s3_client.get_object(Bucket=in_bucket, Key=in_key)
    except s3_client.exceptions.NoSuchKey:
        raise FileNotFoundError(f"[ContrastEnhanceClaheLambda]: Input image not found in S3: {s3_input_uri}")

    input_ext = get_extension(in_key)
    
    if input_ext not in ['.png', '.jpg', '.jpeg', '.npy']:
        raise ValueError(f"[ContrastEnhanceClaheLambda]: Invalid input extension, must be one of ['.png', '.jpg', '.jpeg', '.npy'], not: {input_ext}")

    if input_ext in ['.png', '.jpg', '.jpeg']:
        image_array = load_png_jpg_jpeg_image_from_s3(s3_client, in_bucket, in_key)
    else:
        image_array = load_numpy_from_s3(s3_client, in_bucket, in_key)
        
    H, W = image_array.shape[:2]
    
    # Log image dimensions.
    if image_array.ndim == 2:
        logger.info(f"[ContrastEnhanceClaheLambda]: Detected grayscale image ({H}, {W})")
    elif image_array.ndim == 3 and image_array.shape[2] == 1:
        logger.info(f"[ContrastEnhanceClaheLambda]: Detected single-band image ({H}, {W}, 1)")
    elif image_array.ndim == 3 and image_array.shape[2] == 3:
        logger.info(f"[ContrastEnhanceClaheLambda]: Detected RGB image ({H}, {W}, 3)")
    else:
        raise ValueError(f"[ContrastEnhanceClaheLambda]: Unsupported image shape: {image_array.shape}")

    orig_dtype = image_array.dtype 
    
    # Enhance the contrast. Normalize dtype to float in [0,1] for CLAHE
    image = image_array.astype(np.float32)
    orig_min = image.min()
    orig_max = image.max()
    image_norm = (image - orig_min) / (orig_max - orig_min + 1e-8)  # avoid div by zero
    
    logger.info(f"[ContrastEnhanceClaheLambda]: Applying CLAHE with clip_limit={clip_limit}, kernel_size=({kernel_size}, {kernel_size})")
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        # Grayscale or single-band
        image_gray = image_norm.squeeze()  # (H,W,1) -> (H,W)
        enhanced = exposure.equalize_adapthist(image_gray, clip_limit=clip_limit, kernel_size=(kernel_size, kernel_size))
    
    elif image.ndim == 3 and image.shape[2] == 3:
        # RGB -> LAB -> CE -> RGB
        lab = color.rgb2lab(image_norm)
        
        L = lab[..., 0]
        L_min, L_max = L.min(), L.max()
        L_norm = (L - L_min) / (L_max - L_min + 1e-8)
        
        L_enhanced = exposure.equalize_adapthist(L_norm, clip_limit=clip_limit, kernel_size=(kernel_size, kernel_size))
        
        # Restore original L range
        lab[..., 0] = L_enhanced * (L_max - L_min) + L_min

        enhanced = color.lab2rgb(lab)
    else:
        raise ValueError(f"[ContrastEnhanceClaheLambda]: Unsupported image shape for CLAHE: {image.shape}")

    # Restore original range
    save_array = enhanced * (orig_max - orig_min) + orig_min
    
    if orig_dtype == np.uint8:
        save_array = np.clip(save_array, 0, 255).astype(np.uint8)

    # Save as .npy file to S3.
    upload_numpy_to_s3(s3_client, save_array, out_bucket, out_key)

    logger.info(f"[ContrastEnhanceClaheLambda]: Contrast-enhanced image saved to S3: {s3_output_uri}")