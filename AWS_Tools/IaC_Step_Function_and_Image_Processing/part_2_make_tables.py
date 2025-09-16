# -*- coding: utf-8 -*-
"""
Setup the DynamoDB 'images' and 'preprocessing_styles' tables.

The images table will store metadata about every image (see Part 1).
The preprocessing_styles table will store pre-defined paramaters for how
we want an image to be preprocessed for model training. These parameters acts as input into
the PreprocessorStateMachine (later).
"""

import boto3
import json
import uuid
from botocore.exceptions import ClientError

# Set some paramters ad variables.
AWS_REGION = 'us-east-1'
dynamodb_client = boto3.client('dynamodb', region_name=AWS_REGION)
dynamodb_resource = boto3.resource('dynamodb', region_name=AWS_REGION)

#%%
# CREATE TABLE
TABLE_NAME_IMAGES = 'images'

response = dynamodb_client.create_table(
                    TableName=TABLE_NAME_IMAGES,
                    AttributeDefinitions=[
                        {'AttributeName': 'dataset_name#label', 'AttributeType': 'S'},
                        {'AttributeName': 'split#basename', 'AttributeType': 'S'},
                    ],
                    KeySchema=[
                        {'AttributeName': 'dataset_name#label', 'KeyType': 'HASH'},  # Partition Key
                        {'AttributeName': 'split#basename', 'KeyType': 'RANGE'},     # Sort Key
                    ],
                    BillingMode='PAY_PER_REQUEST'  # On-demand pricing
                )

print("Table creation initiated.")

#%% Grab the images table we made and load in the manifest from earlier.

images_table = dynamodb_resource.Table(TABLE_NAME_IMAGES)

with open('cifar10_manifest.json', 'r') as f:
    manifest = json.load(f)
    
#%% Place each row in the table.

num_records = len(manifest)

for ix, item in enumerate(manifest):
    if ix == 0 or (ix + 1)%100 == 0:
        print(f'On {ix+1} of {num_records}...')
    try:
        images_table.put_item(
                    Item=item,
                    ConditionExpression='attribute_not_exists(#pk) AND attribute_not_exists(#sk)',
                    ExpressionAttributeNames={
                        '#pk': 'dataset_name#label',
                        '#sk': 'split#basename'
                    }
                )
    except ClientError as e:
        if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
            print(f"Item already exists: {item['image_id']}")
        else:
            raise
    
print("Done.")

#%% Now that the images table is made, let's make a table for storing different named
# styles of preprocessing an image. Afterwards, we can simply pick a name and the state machine
# will apply the steps appropriately.

TABLE_NAME_PREPROCESSING = 'preprocessing_styles'

response = dynamodb_client.create_table(
                    TableName=TABLE_NAME_PREPROCESSING,
                    AttributeDefinitions=[
                        {'AttributeName': 'style_name', 'AttributeType': 'S'}
                    ],
                    KeySchema=[
                        {'AttributeName': 'style_name', 'KeyType': 'HASH'}  # Partition Key only
                    ],
                    BillingMode='PAY_PER_REQUEST'  # Flexible, no need to set throughput
                )

print("Table creation initiated.")

#%% Grab the table, then add a few custom styles we can utilize later with predefined settings.
# The partition key will be 'style_name' and there will be no sort key. So,
# 'style_name' is the primary key alone and must all be unique.

table = dynamodb_resource.Table('preprocessing_styles')

#%% Define a few styles to be used in the preprocessing state machine.

styles = [
    {
        'style_name': f"resize256_norm0to255_{uuid.uuid4().hex[:8]}",
        'resize': True,
        'resize_params': {
            'method': 'force',
            'scale': '0',
            'new_width': 256,
            'new_height': 256,
            'resize_order': 3
        },
        'contrast_enhance': False,
        'clahe_contrast_enhance_params': {
            'kernel_size': 8,
            'clip_limit': "0.05"
        },
        'make_grayscale': False,
        'normalization': '0to255'
    },
    {
        'style_name': f"resize128_norm0to1_Grayscale_{uuid.uuid4().hex[:8]}",
        'resize': True,
        'resize_params': {
            'method': 'force',
            'scale': '0',
            'new_width': 128,
            'new_height': 128,
            'resize_order': 3
        },
        'contrast_enhance': False,
        'clahe_contrast_enhance_params': {
            'kernel_size': 8,
            'clip_limit': "0.05"
        },
        'make_grayscale': True,
        'normalization': '0to1'
    },
    {
        'style_name': f"resize512_norm0to1_CE_{uuid.uuid4().hex[:8]}",
        'resize': True,
        'resize_params': {
            'method': 'force',
            'scale': '0',
            'new_width': 512,
            'new_height': 512,
            'resize_order': 4
        },
        'contrast_enhance': True,
        'clahe_contrast_enhance_params': {
            'kernel_size': 8,
            'clip_limit': "0.05"
        },
        'make_grayscale': False,
        'normalization': '0to1'
    },
    {
        'style_name': f"scalingUp20Perc_norm0to1_CE_{uuid.uuid4().hex[:8]}",
        'resize': True,
        'resize_params': {
            'method': 'preserve_aspect_ratio',
            'scale': '1.20',
            'new_width': 0,
            'new_height': 0,
            'resize_order': 4
        },
        'contrast_enhance': True,
        'clahe_contrast_enhance_params': {
            'kernel_size': 8,
            'clip_limit': "0.08"
        },
        'make_grayscale': False,
        'normalization': '0to1'
    },
    {
        'style_name': f"CE_Grayscale_norm0to1{uuid.uuid4().hex[:8]}",
        'resize': False,
        'resize_params': {
            'method': 'preserve_aspect_ratio',
            'scale': '1.20',
            'new_width': 0,
            'new_height': 0,
            'resize_order': 4
        },
        'contrast_enhance': True,
        'clahe_contrast_enhance_params': {
            'kernel_size': 8,
            'clip_limit': "0.08"
        },
        'make_grayscale': True,
        'normalization': '0to1'
    }
]

#%% Add the rows to the table.

for style in styles:
    try:
        table.put_item(
            Item=style,
            ConditionExpression='attribute_not_exists(style_name)'  # Prevent overwrite
        )
        print(f"Inserted: {style['style_name']}")
    except ClientError as e:
        if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
            print(f"Skipped (already exists): {style['style_name']}")
        else:
            print(f"Error inserting {style['style_name']}: {e.response['Error']['Message']}")

print("Done.")
