# cvdms_platform/cvdms.py
from functools import lru_cache
from typing import Optional, Dict, Tuple, List
import os

import boto3
from botocore.exceptions import ClientError

from .clients import UploadClient

SSM_PREFIX_TEMPLATE = "/cvdms/{app}/"
REQUIRED_KEYS = ["job_table_name", "lock_table_name", "file_bucket_name"]

def _session_for_profile(profile_name: Optional[str]) -> boto3.Session:
    # prefer explicit profile; boto3 will use default if profile_name is None
    return boto3.Session(profile_name=profile_name) if profile_name else boto3.Session()

@lru_cache(maxsize=32)
def _load_ssm_params(session: boto3.Session, app_name: str, region_name: str) -> Dict[str, str]:
    ssm = session.client("ssm", region_name=region_name)
    prefix = SSM_PREFIX_TEMPLATE.format(app=app_name)
    out: Dict[str, str] = {}

    # validate region param exists and matches runtime region
    region_param = prefix + "region"
    try:
        resp = ssm.get_parameter(Name=region_param, WithDecryption=True)
        ssm_region = resp["Parameter"]["Value"]
        if ssm_region != region_name:
            raise RuntimeError(f"SSM region mismatch: {region_param}={ssm_region} but session region={region_name}")
    except ClientError as e:
        raise RuntimeError(f"Missing or unreadable SSM region param {region_param} in region {region_name}: {e}")

    # load required params
    for k in REQUIRED_KEYS:
        param = prefix + k
        try:
            r = ssm.get_parameter(Name=param, WithDecryption=True)
            out[k] = r["Parameter"]["Value"]
        except ClientError as e:
            raise RuntimeError(f"Missing required SSM param {param} in region {region_name}: {e}")

    out["_region"] = region_name
    return out

class CvdmsApp:
    """
    Minimal constructor: require only profile_name.
    Example:
      app = CvdmsApp(app_name="cvdmsv1", profile_name="abc")
      ok, out = app.upload_imagery("/tmp/files.csv")
    """

    def __init__(self, app_name: str, profile_name: Optional[str]):
        if not app_name:
            raise ValueError("app_name is required")
        if not profile_name:
            raise ValueError("profile_name is required")

        # create session for the given profile
        session = _session_for_profile(profile_name)

        # resolve region from profile or environment
        region = session.region_name or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
        if not region:
            # instruct user how to set region for profile
            msg = (
                f"Profile '{profile_name}' has no region configured. "
                f"Set region for that profile in ~/.aws/config (e.g. '[profile {profile_name}]\\nregion = us-west-2') "
                "or export AWS_DEFAULT_REGION before running."
            )
            raise RuntimeError(msg)

        # quick credentials check
        try:
            sts = session.client("sts", region_name=region)
            sts.get_caller_identity()
        except Exception as e:
            raise RuntimeError(f"Could not validate AWS credentials for profile '{profile_name}': {e}")

        # load and validate SSM params in resolved region
        cfg = _load_ssm_params(session=session, app_name=app_name, region_name=region)

        # map SSM keys to UploadClient args and validate presence
        file_bucket = cfg.get("file_bucket")
        job_table = cfg.get("job_table")
        lock_table = cfg.get("lock_table")
        if not (file_bucket and job_table and lock_table):
            missing = [k for k in ("file_bucket", "job_table", "lock_table") if not cfg.get(k)]
            raise RuntimeError(f"Missing required SSM params for app {app_name}: {missing}")

        # construct UploadClient; it will create its own boto3 clients using the region
        self._client = UploadClient(
            region_name=region,
            file_bucket=file_bucket,
            job_table=job_table,
            lock_table=lock_table,
        )

    def upload_imagery(self, csv_path: str, *, summary: str = "", dataset_id: Optional[str] = None) -> Tuple[bool, Dict]:
        return self._client.start_upload_job_from_csv(csv_path, summary=summary, dataset_id=dataset_id)

    def update_job_status(self, job_id: str, status: str, error_msg: Optional[str] = None) -> Tuple[bool, str]:
        return self._client.update_job_status(job_id, status, error_msg)