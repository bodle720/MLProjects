# -*- coding: utf-8 -*-
"""
Lambda function to resize an image give the imae URI in S3, and a new height and width.
"""

import boto3
import os
import io
import re
from skimage import io as skio, transform, img_as_ubyte
from PIL import Image

def lambda_handler(event, context):
    # Extract and validate inputs
    try:
        # Input image parameters
        new_height = int(event.get("new_height"))
        new_width = int(event.get("new_width"))
        input_image_s3_uri = event.get("input_image_s3_uri")
        
        # Resizing parameters
        resizing_order = int(event.get("resizing_order"))
        preserve_range_str = str(event.get("preserve_range")) # Just in case.
        preserve_range = preserve_range_str.lower() == "true"


        if not input_image_s3_uri or not re.match(r'^s3://([^/]+)/(.+)$', input_image_s3_uri):
            raise ValueError("Invalid or missing S3 URI")

        bucket, key = re.match(r'^s3://([^/]+)/(.+)$', input_image_s3_uri).groups()
        ext = os.path.splitext(key)[1].lower()

        if ext.lower() not in ['.jpg', '.jpeg', '.png']:
            raise ValueError(f"Unsupported image format: {ext}")

        # Download image from S3
        s3 = boto3.client('s3')
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch image from S3: {str(e)}")

        img_bytes = obj['Body'].read()

        # Load image using skimage
        image = skio.imread(io.BytesIO(img_bytes)) # numpy.ndarray of shape (height, width, num_bands), uint8 datatype

        old_height, old_width = image.shape[:2]
        
        # Resize image
        resized = transform.resize(image,
                                   (new_height, new_width),
                                   order = resizing_order,
                                   preserve_range = preserve_range,
                                   anti_aliasing=True) # uint8 datatype if preserve_range = True
        
        resized_uint8 = img_as_ubyte(resized) # Ensures uint8 datatype

        # Convert to bytes using PIL for upload
        pil_img = Image.fromarray(resized_uint8)
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG' if ext == '.png' else 'JPEG')
        buffer.seek(0)

        # Construct new key
        filename = os.path.basename(key)
        new_key = f"resized_imgs/{filename}"

        # Upload resized image
        s3.put_object(
            Bucket=bucket,
            Key=new_key,
            Body=buffer,
            ContentType='image/jpeg' if ext in ['.jpg', '.jpeg'] else 'image/png'
        )

        # Return structured output for Step Functions
        return {
            "status": "success",
            "resized_s3_uri": f"s3://{bucket}/{new_key}",
            "original_s3_uri": input_image_s3_uri,
            "old_dimensions": {
                "height": old_height,
                "width": old_width
            },
            "new_dimensions": {
                "height": new_height,
                "width": new_width
            },
            "format": ext.lstrip('.')
        }

    except Exception as e:
        # Fail the state machine with a clear error
        raise RuntimeError(f"Lambda failed: {str(e)}")

