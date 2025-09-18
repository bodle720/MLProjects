# -*- coding: utf-8 -*-
"""
This will go over some helper tools I have written for transferring data to and from S3, and is for
educational use.

Note: The upload functionalities will overwrite existing data, which will then be lost forever, so use with caution.
Further, be cautious with deleting buckets as delete_s3_bucket will erase all data in the indicated bucket.
"""

#%% Imports and defintions.

import boto3
import pandas as pd
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray

import s3_helpers as helpers

AWS_REGION = 'us-east-1'
s3_client = boto3.client('s3', region_name=AWS_REGION)
s3_resource = boto3.resource('s3', region_name=AWS_REGION)

#%% Retrieve the list of existing buckets and summarize. Repeat cell to see any changes.

for bucket in helpers.summarize_buckets(s3_client):
    bucket_name = bucket['Name']
    print(f"\nBucket: {bucket_name}")
    for key, value in bucket.items():
        if key != 'Name':
            print(f"  {key}: {value}")
   
#%% Create an S3 bucket using a random, globally unique name.

random_bucket_name = 'some_random_name'
helpers.create_bucket(random_bucket_name,
                      region = 'us-east-1')

#%% Delete the bucket you just made and everything in it. Careful.

helpers.delete_s3_bucket(s3_resource,
                         random_bucket_name,
                         force=True)

#%% Push a local file to S3 to a key you prefer (the object_name 'folder' structure.)

local_path_txt_sample = "sample_file_to_push.txt"
helpers.upload_local_file_to_s3(s3_client,
                                local_path_txt_sample,
                                bucket_name,
                                object_name = 'my/subfolderinS3/sample.txt')

#%% Download the file back to local file system.

helpers.download_s3_obj_to_local_file(s3_client,
                                      bucket_name,
                                      'my/subfolderinS3/sample.txt',
                                      'downloaded_file_from_s3.txt')

#%% Make and upload a Python dict to S3 as a json file.

data_dict = {
        "user_id": [0.2, 0.3, 0.5, 0.7, 0.8],
        "active": True,
        "score": 87.5,
        "tags": ["premium", "beta"],
        "profile": {
            "name": "Alice",
            "location": "NY",
            "preferences": {
                "newsletter": True,
                "notifications": ["email", "sms", 'apps', 'medias']
            }
        }
    }

helpers.upload_dict_to_s3(s3_client,
                          data_dict,
                          bucket_name,
                          'json_data_files/my_data.json')

#%% Upload Python text string to S3 as .txt file.

helpers.upload_text_to_s3(s3_client,
                          'This is a\nmultiline text file.\nThanks for reading,\nAuthor.',
                          bucket_name,
                          'text_data_files/my_data.txt')

#%% Make and upload a Pandas DataFrame to S3 as a CSV file.

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['NY', 'LA', 'Chicago']
})

helpers.upload_dataframe_to_s3(s3_client,
                               df,
                               bucket_name,
                               'csvs_data_files/my_data.csv',
                               encoding='utf-8',
                               index=False)

#%% Retrieve the dictionary we pushed to S3 earlier.

data_dict_pulled = helpers.get_dict_from_s3(s3_client,
                                            bucket_name,
                                            'json_data_files/my_data.json',
                                            encoding='utf-8')
print(data_dict_pulled)

#%% Get the Python text we pushed to S3 earlier.

my_text = helpers.get_text_from_s3(s3_client,
                                   bucket_name,
                                   'text_data_files/my_data.txt',
                                   encoding='utf-8')
print(my_text)


#%% Get the Pandas DataFrame we pushed to S3 earlier.

my_df = helpers.get_dataframe_from_s3(s3_client,
                                      bucket_name,
                                      'csvs_data_files/my_data.csv',
                                      encoding='utf-8',
                                      delimiter=',')
print(my_df.head())

#%% Now let's upload the df above as a .parquet file.

helpers.upload_df_or_dict_as_parquet_to_s3(s3_client,
                                           df,
                                           bucket_name,
                                           'parquet_data_files/my_data.parquet')

#%% Now upload the dict data_dict above as a .parquet file.

helpers.upload_df_or_dict_as_parquet_to_s3(s3_client,
                                           data_dict,
                                           bucket_name,
                                           'parquet_data_files/my_data_dict.parquet')

#%% Finally, retureve the .parquet files. First, the DataFrame, indicated as return_as_dict = False.

df_from_par = helpers.get_df_or_dict_parquet_from_s3(s3_client,
                                                     bucket_name,
                                                     'parquet_data_files/my_data.parquet',
                                                     return_as_dict=False)

#%% And do the same for the dict, using return_as_dict = True. This returns a list of dicts.

data_dict_from_par = helpers.get_df_or_dict_parquet_from_s3(s3_client,
                                                            bucket_name,
                                                            'parquet_data_files/my_data_dict.parquet',
                                                            return_as_dict=True)

#%% Let's load in an image in different formats as a numpy array and save to S3.

local_img_path = "imgs/boat.jpg"
image_array_rgb = imread(local_img_path)

print('RGB Image type: ', type(image_array_rgb)) 
print('RGB Image shape: ', image_array_rgb.shape) # (H, W, 3) = (image height, image width, num bands)
print('RGB Image dtype: ', image_array_rgb.dtype)  # uint8
print('-'*50)
# Convert to grayscale
image_array_gray = rgb2gray(image_array_rgb)
print('Grayscale Image type: ', type(image_array_gray))
print('Grayscale Image shape: ', image_array_gray.shape) # (H, W)
print('Grayscale Image dtype: ', image_array_gray.dtype)  # float64
print('-'*50)

# Make the Grayscale image have one band
image_array_gray3d = np.expand_dims(image_array_gray, axis=-1)
print('Grayscale Single Band Image type: ', type(image_array_gray3d))
print('Grayscale Single Band Image shape: ', image_array_gray3d.shape) # (H, W, 1)
print('Grayscale Single Band Image dtype: ', image_array_gray3d.dtype) # float64
print('-'*50)

#%% Save them to S3

helpers.upload_numpy_to_s3(s3_client, image_array_rgb, bucket_name, 'img_rgb.npy')
helpers.upload_numpy_to_s3(s3_client, image_array_gray, bucket_name, 'img_gray.npy')
helpers.upload_numpy_to_s3(s3_client, image_array_gray3d, bucket_name, 'img_gray3d.npy')

#%% Load them in and inspect. Preserves shape and data type.

img_rgb = helpers.load_numpy_from_s3(s3_client, bucket_name, 'img_rgb.npy')
img_gray = helpers.load_numpy_from_s3(s3_client, bucket_name, 'img_gray.npy')
img_gray3d = helpers.load_numpy_from_s3(s3_client, bucket_name, 'img_gray3d.npy')

print('Loaded in RGB Image type: ', type(img_rgb)) 
print('Loaded in RGB Image shape: ', img_rgb.shape) # (H, W, 3)
print('Loaded in RGB Image dtype: ', img_rgb.dtype)  # uint8
print('-'*50)

print('Loaded in Grayscale Image type: ', type(img_gray))
print('Loaded in Grayscale Image shape: ', img_gray.shape) # (H, W)
print('Loaded in Grayscale Image dtype: ', img_gray.dtype)  # float64
print('-'*50)

print('Loaded in Grayscale Single Band Image type: ', type(img_gray3d))
print('Loaded in Grayscale Single Band Image shape: ', img_gray3d.shape) # (H, W, 1)
print('Loaded in Grayscale Single Band Image dtype: ', img_gray3d.dtype) # float64

#%% Save the image file itself to S3 (as a .jpg)

helpers.upload_local_file_to_s3(s3_client, local_img_path, bucket_name, object_name = "imgs/boat.jpg")

#%% Load the .jpg back in as a numpy array.

loaded_img_arr = helpers.load_png_jpg_jpeg_image_from_s3(s3_client, bucket_name, "imgs/boat.jpg")

print('Loaded in Image type: ', type(loaded_img_arr))
print('Loaded in Image shape: ', loaded_img_arr.shape) # (H, W, 3)
print('Loaded in Image dtype: ', loaded_img_arr.dtype) # uint8










