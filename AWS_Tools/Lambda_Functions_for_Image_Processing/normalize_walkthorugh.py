# -*- coding: utf-8 -*-
"""
This script will call on the helper functions in helpers/ to create the NormalizeLambda function.

The lambda function, NormalizeLambda, will only require NumPy and Pillow. This will be light enough
to avoid needing Docker. We will choose the strategy of creating a layer composed of NumPy and Pillow
and attach it to the lambda function itself so it has it at runtime. An alternate approach is to pip
install the two packages and their dependencies directly into lambdas/NormalizeLambda. Then, zip everything
up and utilize that to create the lambda using the helper helpers.lambda_helpers.create_lambda_function.

Using a layer makes the dependencies more modular so they can be reused by other lambdas rather than reinstalling
them each time.

Ensure you're authenticated via the AWS CLI before running this script.
"""

#%% Imports and variable definitions.

import json
import boto3
import time
import io
from skimage import io as skio
import matplotlib.pyplot as plt

from helpers.s3_helpers import upload_local_file_to_s3, load_numpy_from_s3
from helpers.lambda_helpers import publish_new_lambda_layer, zip_folder_of_lambda_function_contents, \
                                    create_lambda_function, describe_a_lambda

AWS_REGION = 'us-east-1' # Replace with your region
account_id = boto3.client("sts").get_caller_identity()["Account"]
lambda_client = boto3.client("lambda", region_name=AWS_REGION)
s3_client = boto3.client('s3', region_name=AWS_REGION)

#%% This step installs the required dependencies and publishes a reusable Lambda layer containing NumPy and Pillow.

print('Publishing a new layer, it may take a few minutes...')

layer_name = "numpy-pillow-layer"
requirements = ["numpy", "pillow"]
runtime = "python3.12"
description="Numpy and Pillow packages for image processing workflows."

layer_arn = publish_new_lambda_layer(
                                    lambda_client=lambda_client,
                                    layer_name=layer_name,
                                    requirements = requirements, # boto3 is included by default by AWS.
                                    description=description,
                                    runtime = runtime
                                )

print(f"Published layer ARN: {layer_arn}")

#%% Since the NormalizeLambda function is a Zip Lambda, I've created a helper function to streamline
# the process. Of course, you can do this manually in File Explorer as well. Note: do not zip the 
# folder itself, just the contents of the lambdas/NormalizeLambda/ folder. This code will create  the
# normalize_lambda_zipped.zip file, which contains the definition of our lambda.
# We will use the file lambdas/normalize_lambda_zipped.zip to define the lambda function.

source_dir = "lambdas/NormalizeLambda"
save_to = "lambdas/normalize_lambda_zipped.zip"

zip_folder_of_lambda_function_contents(source_dir,
                                       save_to)

#%% Create the lambda function NormalizeLambda; primary code is inside lambdas/NormalizeLambda/worker.py
# We will use the .zip file created above. Again, a helper function I made does this cleanly.
# This function will also attach the above layer to the function so it can correctly import NumPy and Pillow.

from_docker = False
code_source = save_to
function_name = "NormalizeLambda"
lambda_role_arn = f"arn:aws:iam::{account_id}:role/ImgNormalizationLambdaRole" # Replace with your IAM role ARN
handler = "worker.lambda_handler" # Required for zip lambdas
env_vars = None # See function docstring for explanation
runtime = 'python3.12' # Matches the layer we made above
timeout = 30 # Seconds, very generous, depends on task.
memory_size = 1024 # If too small and images too big, triggers OOM Error.
layers_to_attach = [layer_arn] # Attach the layer we made containing the dependencies the lambda will import.
description = "A Lambda function created from a ZIP image that normalizes an input S3 URI .jpg, .jpeg, .png or .npy file according to input parameters."

response_NormalizeLambda = create_lambda_function(lambda_client,
                                                 from_docker,
                                                 code_source,
                                                 function_name,
                                                 lambda_role_arn,  
                                                 handler=handler, 
                                                 env_vars=env_vars,
                                                 runtime=runtime, 
                                                 timeout=timeout, 
                                                 memory_size=memory_size,
                                                 layers_to_attach=layers_to_attach,
                                                 description=description)

#%% Describe some features of the lambda.

describe_a_lambda(lambda_client, function_name)

#%% The NormalizeLambda function is now fully deployed and ready for testing. Let's test it out with a sample image.

# Upload the curious cat image to an S3 bucket.
path_to_local_img = "imgs/curious_cat.jpg"
bucket_name = "scratchwork-temp-001" # Replace with your actual bucket name.
upload_success = upload_local_file_to_s3(s3_client, path_to_local_img, bucket_name, object_name = "temp/curious_cat.jpg")
print('Was the upload successful? ', 'Yes!' if upload_success else 'No.')

#%% Let's load in the local image and do a little histogram exploration.

image_array_local = skio.imread(path_to_local_img)
print('Local Image is of type = ', type(image_array_local)) # <class 'numpy.ndarray'>
print('Local Image has shape: ', image_array_local.shape) # (RGB) -> (H, W, 3) -> (2688, 1920, 3)
print('Local Image has data type: ', image_array_local.dtype) #  uint8
print(f'Local Image pixel values range from {image_array_local.min()} to {image_array_local.max()}') # 0 to 255

local_pixel_values = image_array_local.flatten()
plt.figure(figsize=(8, 4))
plt.hist(local_pixel_values, bins=256, color='gray', edgecolor='black')
plt.title("Local Curious Cat Pixel Intensity Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

#%% Let's load in the S3 image and do a sanity check. They should be identical.

response = s3_client.get_object(Bucket=bucket_name, Key="temp/curious_cat.jpg")
image_bytes = response["Body"].read()
buffer = io.BytesIO(image_bytes)
image_array_s3_raw = skio.imread(buffer)

print('S3 Raw Image is of type = ', type(image_array_s3_raw))
print('S3 Raw Image has shape: ', image_array_s3_raw.shape) # (RGB)
print('S3 Raw Image has data type: ', image_array_s3_raw.dtype) #  uint8
print(f'S3 Raw Image pixel values range from {image_array_s3_raw.min()} to {image_array_s3_raw.max()}')

s3_raw_pixel_values = image_array_s3_raw.flatten()
plt.figure(figsize=(8, 4))
plt.hist(s3_raw_pixel_values, bins=256, color='gray', edgecolor='black')
plt.title("S3 Raw Curious Cat Pixel Intensity Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

#%% Let's invoke the lambda function NormalizeLambda and normalize the image from 0 to 1 and 0 to 255
# and save each result

responses = []
payloads = [{
        "s3_raw_uri": f"s3://{bucket_name}/temp/curious_cat.jpg",
        "s3_output_uri": f"s3://{bucket_name}/normalized_images/curious_cat_normalized_0to1.npy", 
        "normalization": "0to1" 
    },
    {
        "s3_raw_uri": f"s3://{bucket_name}/temp/curious_cat.jpg",
        "s3_output_uri": f"s3://{bucket_name}/normalized_images/curious_cat_normalized_0to255.npy", 
        "normalization": "0to255" 
    }]


for payload in payloads:
    print(f"Invoking NormalizeLambda with normalization={payload['normalization']}")
    resp = lambda_client.invoke(
        FunctionName="NormalizeLambda",
        InvocationType="RequestResponse",  # Use 'Event' for async
        Payload=json.dumps(payload).encode("utf-8")
    )
    time.sleep(0.5)
    responses.append(resp)

print(responses)

#%% Let's load the normalized numpy array from S3 that was normalized from 0 to 1 and inspect.

img_0to1 = load_numpy_from_s3(s3_client, bucket_name, "normalized_images/curious_cat_normalized_0to1.npy")

print('Loaded in Image is of type = ', type(img_0to1)) # <class 'numpy.ndarray'>
print('Loaded in Image has shape: ', img_0to1.shape) # (RGB) -> (H, W, 3) -> (2688, 1920, 3)
print('Loaded in Image has data type: ', img_0to1.dtype) #  float32
print(f'Loaded in Image pixel values range from {img_0to1.min()} to {img_0to1.max()}') # 0 to 1

loaded_pixel_values = img_0to1.flatten()
plt.figure(figsize=(8, 4))
plt.hist(loaded_pixel_values, bins=256, color='gray', edgecolor='black')
plt.title("Loaded from S3 0 to 1 Curious Cat Pixel Intensity Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

#%% Let's load the normalized numpy array from S3 that was normalized from 0 to 255 and compare.

img_0to255 = load_numpy_from_s3(s3_client, bucket_name, "normalized_images/curious_cat_normalized_0to255.npy")

print('Loaded in Image is of type = ', type(img_0to255)) # <class 'numpy.ndarray'>
print('Loaded in Image has shape: ', img_0to255.shape) # (RGB) -> (H, W, 3) -> (2688, 1920, 3) 
print('Loaded in Image has data type: ', img_0to255.dtype) #  uint8
print(f'Loaded in Image pixel values range from {img_0to255.min()} to {img_0to255.max()}') # 0 to 255

loaded_pixel_values = img_0to255.flatten()
plt.figure(figsize=(8, 4))
plt.hist(loaded_pixel_values, bins=256, color='gray', edgecolor='black')
plt.title("Loaded from S3 0 to 255 Curious Cat Pixel Intensity Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()