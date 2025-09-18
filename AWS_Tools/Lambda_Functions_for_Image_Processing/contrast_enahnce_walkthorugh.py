# -*- coding: utf-8 -*-
"""
This script will call on the helper functions in helpers/ to create the ContrastEnhanceCLAHELambda function.
Unlike NormalizeLambda, this will be a docker based lambda fuction.

With large packages like scikit-image, you cannot zip it and its required dependencies
because they break the 70 MB limit. 

You could package each dependency as a separate Lambda layer, which improves modularity and reuse across
functions. However, this adds complexity, so we’ll use a Docker image to encapsulate the full environment.

Be sure to sign in thorugh the AWS CLI.
"""

#%% Imports and variable definitions.

import json
import boto3
import time
from skimage import io as skio
import matplotlib.pyplot as plt

from helpers.s3_helpers import upload_local_file_to_s3, load_numpy_from_s3
from helpers.ecr_docker_helpers import build_and_push_docker_image_to_ecr
from helpers.lambda_helpers import create_lambda_function
                                    
AWS_REGION = 'us-east-1' # Replace with your region
account_id = boto3.client("sts").get_caller_identity()["Account"]
lambda_client = boto3.client("lambda", region_name=AWS_REGION)
s3_client = boto3.client('s3', region_name=AWS_REGION)
ecr_client = boto3.client('ecr', region_name=AWS_REGION)

#%% We need to register the Docker image in ECR first, AWS's ECR Registry (like a Docker Hub repo)

image_name = "apply-clahe-lambda"
path_to_folder_containing_dockerfile = "lambdas/ContrastEnhanceCLAHELambda"
local_tag = ecr_tag = 'latest'
for_lambda_fn = True

ecr_image_uri = build_and_push_docker_image_to_ecr(ecr_client,
                                                   AWS_REGION, 
                                                   account_id,
                                                   image_name,
                                                   path_to_folder_containing_dockerfile,
                                                   local_tag,
                                                   ecr_tag,
                                                   for_lambda_fn)

#%% This will make the lambda function.

from_docker = True
code_source = ecr_image_uri
function_name = 'ContrastEnhanceClaheLambda'
lambda_role_arn = f"arn:aws:iam::{account_id}:role/LambdaBasicRole" # Replace with your role.

response = create_lambda_function(lambda_client,
                                from_docker,
                                code_source,
                                function_name,
                                lambda_role_arn,
                                handler=None,
                                env_vars=None,
                                runtime='python3.12',
                                timeout=30,
                                memory_size=1024,
                                description="A Lambda function created from a Docker image in ECR that Enhances the contrast of an image using CLAHE.")


print(response)

#%% The lambda function should be made now. Let's try it out. We will use the cat image in imgs folder

# Upload the curious cat image to an S3 bucket.
path_to_local_img = "imgs/curious_cat.jpg"
bucket_name = "your-bucket-name-here-xxxxx" # Replace with your actual bucket name.
upload_success = upload_local_file_to_s3(s3_client, path_to_local_img, bucket_name, object_name = "temp/curious_cat.jpg")
print('Was upload a success?: ', 'Yes!' if upload_success else 'No.')

#%% View the histogram of the local image.

image_array_local = skio.imread(path_to_local_img)
print('Local Image is of type = ', type(image_array_local)) # <class 'numpy.ndarray'>
print('Local Image has shape: ', image_array_local.shape) # (RGB) -> (H, W, 3) -> (2688, 1920, 3)
print('Local Image has data type: ', image_array_local.dtype) #  uint8
print(f'Local Image pixel values range from {image_array_local.min()} to {image_array_local.max()}') # 0 to 255

local_pixel_values = image_array_local.flatten()
plt.figure(figsize=(8, 4))
plt.hist(local_pixel_values, bins=256, color='gray', edgecolor='black')
plt.title("Original Curious Cat Pixel Intensity Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("imgs/histogram_original.png")
plt.show()

#%% Invoke the lambda on this S3 image with various settings, and save them to S3. The paramters are:
    # s3_input_uri <-- fixed, the curious cat image
    # s3_output_uri <-- where we save the three outputs
    # kernel_size <-- fixing this to 256
    # clip_limit <-- experimenting with this: 0.03, 0.05, and 0.1

responses = []
payloads = [{
        "s3_input_uri": f"s3://{bucket_name}/temp/curious_cat.jpg",
        "s3_output_uri": f"s3://{bucket_name}/enhanced_images/curious_cat_cl03_kernel256.npy", 
        "kernel_size": "256",
        "clip_limit": "0.03" 
    },
    {
        "s3_input_uri": f"s3://{bucket_name}/temp/curious_cat.jpg",
        "s3_output_uri": f"s3://{bucket_name}/enhanced_images/curious_cat_cl05_kernel256.npy", 
        "kernel_size": "256" ,
        "clip_limit": "0.05" 
    },
    {
        "s3_input_uri": f"s3://{bucket_name}/temp/curious_cat.jpg",
        "s3_output_uri": f"s3://{bucket_name}/enhanced_images/curious_cat_cl1_kernel256.npy", 
        "kernel_size": "256",
        "clip_limit": "0.1" 
    }
   ]


for payload in payloads:
    print(f"Invoking Lambda with clip_limit={payload['clip_limit']}")
    resp = lambda_client.invoke(
        FunctionName=function_name,
        InvocationType="RequestResponse",  # Use 'Event' for async
        Payload=json.dumps(payload).encode("utf-8")
    )
    time.sleep(0.5)
    responses.append(resp)
    
#%% Wait a moment and verify the files exist in S3, then load them in and view the histograms.

img_array_03 = load_numpy_from_s3(s3_client, bucket_name, "enhanced_images/curious_cat_cl03_kernel256.npy")
img_array_05 = load_numpy_from_s3(s3_client, bucket_name, "enhanced_images/curious_cat_cl05_kernel256.npy")
img_array_1 = load_numpy_from_s3(s3_client, bucket_name, "enhanced_images/curious_cat_cl1_kernel256.npy")

image_arrays = [img_array_03, img_array_05, img_array_1]
names = ["Clip Limit = 0.03", "Clip Limit = 0.05", "Clip Limit = 0.10"]
limits = ['03', '05', '10']

for img_arr, name, limit in zip(image_arrays, names, limits):
    print(f'Image with {name} is of type = ', type(img_arr))
    print(f'Image with {name} has shape: ', img_arr.shape) 
    print(f'Image with {name} has data type: ', img_arr.dtype)
    print(f'Image with {name} has pixel values range from {img_arr.min()} to {img_arr.max()}')
    print('-'*50)

    pixel_values = img_arr.flatten()
    plt.figure(figsize=(8, 4))
    plt.hist(pixel_values, bins=256, color='gray', edgecolor='black')
    plt.title(f"Histogram After CLAHE ({name})")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"imgs/histogram_cl_{limit}.png")
    plt.show()
    
#%% Let's save each of these locally to view. Open on your local system to view.

for img_arr, limit in zip(image_arrays, limits):
    skio.imsave(f"imgs/curious_cat_enhanced_cl_{limit}.jpg", img_arr)