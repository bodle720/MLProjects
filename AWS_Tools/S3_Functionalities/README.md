# AWS S3 Helper Functions (Educational)

These scripts provide a collection of Python functions using Boto3 to make interacting with Amazon S3 easier. It’s designed for educational purposes to help learners understand how to:

- List and summarize existing S3 buckets
- Create and delete buckets
- Upload and download files, text, dictionaries, and DataFrames
- Serialize and deserialize data to/from JSON, CSV, and Parquet formats

---

## What’s Included

- `summarize_buckets()`: Lists all buckets and summarizes metadata like region, versioning, and policy
- `create_bucket()`: Creates a bucket in a specified region
- `delete_s3_bucket()`: Deletes a bucket, optionally removing all objects and versions
- `upload_local_file_to_s3()`: Uploads a local file to S3
- `download_s3_obj_to_local_file()`: Downloads an object from S3 to local disk
- `upload_dict_to_s3()`, `upload_text_to_s3()`, `upload_dataframe_to_s3()`: Uploads structured data to S3
- `get_dict_from_s3()`, `get_text_from_s3()`, `get_dataframe_from_s3()`: Retrieves structured data from S3
- `upload_df_or_dict_as_parquet_to_s3()`, `get_df_or_dict_parquet_from_s3()`: Handles Parquet serialization

---

## Disclaimer

This code is intended for **educational use only**. It interacts with live AWS resources and may perform destructive operations (e.g., deleting buckets or overwriting objects). Please ensure:

- You have appropriate permissions before running any function
- You are working in a safe, non-production environment
- You do **not** expose AWS credentials, secrets, or sensitive data in your code or uploads

---

## AWS Authentication

These scripts use `boto3`, which automatically pulls credentials from your local AWS configuration. You must authenticate using one of the following methods:

### Option 1: AWS CLI with static credentials
```bash
aws configure
```
Stores your access key and secret key in `~/.aws/credentials`.

### Option 2: AWS SSO (recommended for organizations)
```bash
aws sso login --profile your-profile-name
```
Make sure your default or named profile is active when running the script.

---

## Required IAM Permissions

To run these functions successfully, your IAM or Identity Center user or role should have the following permissions:

```json
{
  "Effect": "Allow",
  "Action": [
    "s3:ListBucket",
    "s3:GetObject",
    "s3:PutObject",
    "s3:DeleteObject",
    "s3:CreateBucket",
    "s3:DeleteBucket",
    "s3:GetBucketPolicy",
    "s3:GetBucketVersioning",
    "s3:GetBucketLocation",
    "s3:GetBucketPublicAccessBlock"
  ],
  "Resource": "*"
}
```

For tighter security, you can scope `Resource` to specific buckets.

---

## Notes of Caution

- Bucket names must be globally unique and follow AWS naming rules
- Uploading to S3 with the same key will **overwrite** existing objects
- Parquet files only store the data you write—no credentials or environment info
- Avoid uploading `.aws/` or local data files to GitHub; use `.gitignore` to exclude them

---

Happy building—and feel free to fork, extend, or adapt for your own learning projects!
