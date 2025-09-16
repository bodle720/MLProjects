# -*- coding: utf-8 -*-
"""
This script will call on the helper functions in helpers/ to create the four lambda functions
indicated in README_preprocessing_state_machine.md

1. ResizeLambda (Docker based)
2. ContrastEnhanceCLAHELambda (Docker based)
3. MakeGrayscaleLambda (Docker based)
4. NormalizeLambda (Zip based, for example and educational reasons only)

With large packages like scikit-image, you cannot zip it and its required dependencies
because they break the 70 MB limit. You could break each dependency into separate pieces, zip them,
and uplod as separate layers and attach to the necessary function that way. This has the benefit of modularity
and reusability by other lambdas. However, if size permits, you can zip you lambda code and all dependencies
into one zip file and upload the entire lambda function and its packages as one unit.

The last lambda function, NormalizeLambda, will only require NumPy and Pillow. This will be light enough
to avoid needing Docker. We will choose the strategy of creating a layer composed of NumPy and Pillow
and attach it to th lambda function itself so it has it at runtime, though you can probably upload it all as one unit
in one zip file if desired.

In the best tradition of sci-fi, we will start by building the fourth lambda function: NormalizeLambda

todo change normalize role name to palcejholder
"""

#%% Imports and variable definitions.

import json
import boto3
import io
from skimage import io as skio
import matplotlib.pyplot as plt

from helpers.s3_helpers import upload_local_file_to_s3
from helpers.lambda_helpers import publish_new_lambda_layer, zip_folder_of_lambda_function_contents, \
                                    create_lambda_function

AWS_REGION = 'us-east-1' # Replace with your region
account_id = boto3.client("sts").get_caller_identity()["Account"]
lambda_client = boto3.client("lambda", region_name=AWS_REGION)
s3_client = boto3.client('s3', region_name=AWS_REGION)

#%% This will install all the dependencies you need for a layer we will attach to our NormalizeLambda function
# The following automatically installs the dependencies and published the layer to AWS. It is a hleper function
# I created to streamline the process and make the code cleaner. See the heleprs/ folder for the code.

print('Publishing a new layer, it may take a few minutes...')
layer_arn = publish_new_lambda_layer(
                                    lambda_client=lambda_client,
                                    layer_name="numpy-pillow-layer",
                                    requirements = ["numpy", "pillow"], # boto3 is included by default by AWS.
                                    description="Numpy and Pillow packages for image processing workflows.",
                                    runtime = "python3.12"
                                )

print(f"Published layer ARN: {layer_arn}")

#%% Since the NormalizeLambda function is a Zip Lambda, I've created a helper function to streamline
# the process. Of course, you can do this manually in File Explorer as well.

zip_folder_of_lambda_function_contents("lambdas/NormalizeLambda",
                                       "lambdas/normalize_lambda_zipped.zip")

print('Use the file lambdas/normalize_lambda_zipped.zip to define the lambda function.')

#%% Create the lambda function NormalizeLambda; primary code is inside lambdas/NormalizeLambda/worker.py
# We will use the .zip file created above. Again, a helper function I made does this cleanly.
# This function will also attatch the above layer fo us.

response = create_lambda_function(lambda_client,
                                 False,
                                 "lambdas/normalize_lambda_zipped.zip",
                                 "NormalizeLambda",
                                 f"arn:aws:iam::{account_id}:role/ImgNormalizationLambdaRole",  # Replace with your IAM role ARN
                                 handler="worker.lambda_handler", # Required for zip lambdas
                                 env_vars=None,
                                 runtime='python3.12', # Matches the layer we made above
                                 timeout=30, # Very generous.
                                 memory_size=1024,
                                 layers_to_attach = [layer_arn],
                                 description="A Lambda function created from a ZIP image that normalizes an input S3 URI .jpg, .jpeg, or .png file according to input parameters.")

#%% We are completely done with the NormalizeLambda function. Let's test it out with a sample image.

# Upload the curious cat image to an S3 bucket.
path_to_local_img = "imgs/curious_cat.jpg"
bucket_name = "scratchwork-temp-001"
upload_success = upload_local_file_to_s3(s3_client, path_to_local_img, bucket_name, object_name = "temp/curious_cat.jpg")
print('Was upload a success?: ', 'Yes!' if upload_success else 'No.')

#%% Let's load in the local image and do a little exploration.

image_array_local = skio.imread(path_to_local_img)
print('Local Image is of type = ', type(image_array_local))
print('Local Image has shape: ', image_array_local.shape) # (RGB)
print(f'Local Image pixel values range from {image_array_local.min()} to {image_array_local.max()}')

local_pixel_values = image_array_local.flatten()
plt.figure(figsize=(8, 4))
plt.hist(local_pixel_values, bins=256, color='gray', edgecolor='black')
plt.title("Local Curious Cat Pixel Intensity Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

#%% Let's load in the S3 image and do a sanity check. They are identical. Sanity check passed.

response = s3_client.get_object(Bucket=bucket_name, Key="temp/curious_cat.jpg")
image_bytes = response["Body"].read()
buffer = io.BytesIO(image_bytes)
image_array_s3_raw = skio.imread(buffer)

print('S3 Raw Image is of type = ', type(image_array_s3_raw))
print('S3 Raw Image has shape: ', image_array_s3_raw.shape) # (RGB)
print(f'S3 Raw Image pixel values range from {image_array_s3_raw.min()} to {image_array_s3_raw.max()}')

s3_raw_pixel_values = image_array_s3_raw.flatten()
plt.figure(figsize=(8, 4))
plt.hist(s3_raw_pixel_values, bins=256, color='gray', edgecolor='black')
plt.title("S3 Raw Curious Cat Pixel Intensity Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

#%% Let's invoke the lambda function NormalizeLambda and normalize it from 0 to 1.

payload = {
    "s3_raw_uri": f"s3://{bucket_name}/temp/curious_cat.jpg",
    "s3_output_uri": f"s3://{bucket_name}/temp/curious_cat_normalized.jpg",
    "normalization": "0to1"
}

response = lambda_client.invoke(
    FunctionName="NormalizeLambda",
    InvocationType="RequestResponse",  # Use 'Event' for async
    Payload=json.dumps(payload).encode("utf-8")
)

#%% Read the response
response_payload = json.load(response['Payload'])
print(response_payload)


#%%
import boto3
import json
import time
from lambda_helpers import make_lambda_function
from ecr_docker_helpers import build_and_push_docker_image_to_ecr
#%% Make the lambda functions

region = 'us-east-1'
ecr_client = boto3.client('ecr', region_name=region)
account_id = boto3.client("sts").get_caller_identity()["Account"]
image_name = "step-lambda-task-2"
path_to_folder_containing_dockerfile = "lambdas/SimpleTask2_docker"
local_tag = ecr_tag = 'latest'
lambda_client = boto3.client('lambda', region_name=region)
for_lambda_fn = True

ecr_image_uri = build_and_push_docker_image_to_ecr(ecr_client,
                                                   region, 
                                                   account_id,
                                                   image_name,
                                                   path_to_folder_containing_dockerfile,
                                                   local_tag,
                                                   ecr_tag,
                                                   for_lambda_fn)

print(f'Successfully made ECR image: {ecr_image_uri}')


#%% Make the zipped up lambda function.

lambda_1_create_out = make_lambda_function(lambda_client,
                                             False,
                                             r"lambdas\SimpleTask1_zip/my_worker.zip",
                                             'lambda-worker-1',
                                             f"arn:aws:iam::{account_id}:role/LambdaBasicRole",
                                             handler="my_worker.do_stuff",
                                             env_vars={'Variables':{'some_val': "10"}},
                                             runtime='python3.12',
                                             timeout=20,
                                             description="A Lambda function created from a ZIP file.")

lambda_1_arn = lambda_1_create_out['FunctionArn']
#%% Make the docker lambda, one based on the above ecr image.

lambda_2_create_out = make_lambda_function(lambda_client,
                                             True,
                                             ecr_image_uri,
                                             'lambda-worker-2',
                                             f"arn:aws:iam::{account_id}:role/LambdaBasicRole",
                                             handler=None,
                                             env_vars=None,
                                             runtime='python3.12',
                                             timeout=20,
                                             description="A Lambda function created from a Docker image.")

#%%
lambda_2_arn = lambda_2_create_out['FunctionArn']



#%%
step_client = boto3.client('stepfunctions')

state_machine_name = ""
lambda1_arn = ""
lambda2_arn = ""
state_machine_role_arn = ""

definition = {
    "Comment": "Sequential Lambda execution",
    "StartAt": "Lambda1",
    "States": {
        "Lambda1": {
            "Type": "Task",
            "Resource": lambda1_arn,
            "Next": "Lambda2"
        },
        "Lambda2": {
            "Type": "Task",
            "Resource": lambda2_arn,
            "End": True
        }
    }
}

response = step_client.create_state_machine(
    name=state_machine_name,
    definition=json.dumps(definition),
    roleArn=state_machine_role_arn,
    type='STANDARD'
)

print(response['stateMachineArn'])