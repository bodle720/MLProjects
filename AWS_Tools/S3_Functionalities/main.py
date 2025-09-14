# -*- coding: utf-8 -*-
"""
This will go over some helper tools I have written for transferring data to and from S3, and is for
educational use.

Note: The upload functionalities will overwrite existing data, which will then be lost forever, so use with caution.
Further, be cautious with deleting buckets as delete_s3_bucket will erase all data in the indicated bucket.
"""

#%% Imports

import pandas as pd
import helpers

#%% Retrieve the list of existing buckets and summarize. Repeat cell to see any changes.

for bucket in helpers.summarize_buckets():
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

helpers.delete_s3_bucket(random_bucket_name,
                         force=True)

#%% Push a local file to S3 to a key you prefer (the object_name 'folder' structure.)

local_path_txt_sample = "sample_file_to_push.txt"
helpers.upload_local_file_to_s3(local_path_txt_sample,
                                bucket_name,
                                object_name = 'my/subfolderinS3/sample.txt')

#%% Download the file back to local file system.

helpers.download_s3_obj_to_local_file(bucket_name,
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

helpers.upload_dict_to_s3(data_dict,
                          bucket_name,
                          'json_data_files/my_data.json')

#%% Upload Python text string to S3 as .txt file.

helpers.upload_text_to_s3('This is a\nmultiline text file.\nThanks for reading,\nAuthor.',
                          bucket_name,
                          'text_data_files/my_data.txt')

#%% Make and upload a Pandas DataFrame to S3 as a CSV file.

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['NY', 'LA', 'Chicago']
})

helpers.upload_dataframe_to_s3(df,
                               bucket_name,
                               'csvs_data_files/my_data.csv',
                               encoding='utf-8',
                               index=False)

#%% Retrieve the dictionary we pushed to S3 earlier.

data_dict_pulled = helpers.get_dict_from_s3(bucket_name,
                                            'json_data_files/my_data.json',
                                            encoding='utf-8')
print(data_dict_pulled)

#%% Get the Python text we pushed to S3 earlier.

my_text = helpers.get_text_from_s3(bucket_name,
                                   'text_data_files/my_data.txt',
                                   encoding='utf-8')
print(my_text)


#%% Get the Pandas DataFrame we pushed to S3 earlier.

my_df = helpers.get_dataframe_from_s3(bucket_name,
                                      'csvs_data_files/my_data.csv',
                                      encoding='utf-8',
                                      delimiter=',')
print(my_df.head())

#%% Now let's upload the df above as a .parquet file.

helpers.upload_df_or_dict_as_parquet_to_s3(df,
                                           bucket_name,
                                           'parquet_data_files/my_data.parquet')

#%% Now upload the dict data_dict above as a .parquet file.

helpers.upload_df_or_dict_as_parquet_to_s3(data_dict,
                                           bucket_name,
                                           'parquet_data_files/my_data_dict.parquet')

#%% Finally, retureve the .parquet files. First, the DataFrame, indicated as return_as_dict = False.

df_from_par = helpers.get_df_or_dict_parquet_from_s3(bucket_name,
                                                     'parquet_data_files/my_data.parquet',
                                                     return_as_dict=False)

#%% And do the same for the dict, using return_as_dict = True. This returns a list of dicts.

data_dict_from_par = helpers.get_df_or_dict_parquet_from_s3(bucket_name,
                                                            'parquet_data_files/my_data_dict.parquet',
                                                            return_as_dict=True)