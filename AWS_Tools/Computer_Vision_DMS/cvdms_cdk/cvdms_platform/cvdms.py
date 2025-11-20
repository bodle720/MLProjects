from functools import lru_cache
from typing import Optional, Dict, Tuple, List
import os
import logging

import boto3
from botocore.exceptions import ClientError

from .clients import UploadClient

SSM_PREFIX_TEMPLATE = "/cvdms/{app}/"
REQUIRED_KEYS = ["storage/job_table_name", "storage/lock_table_name", "storage/file_bucket_name"]

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

        main_dir = os.path.dirname(__file__)
        logs_folder = os.path.join(main_dir, 'logs')
        os.makedirs(logs_folder, exist_ok=True)

        # Configure logging settings
        logging_save_to = os.path.join(logs_folder, 'logs.txt')
        logger = logging.getLogger()
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(logging_save_to)
        console_handler = logging.StreamHandler()

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logging.info('Instantiating user instance.')

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
            identity = sts.get_caller_identity()
            user = identity["Arn"].split("/")[-1]
        except Exception as e:
            raise RuntimeError(f"Could not validate AWS credentials for profile '{profile_name}': {e}")

        # load and validate SSM params in resolved region
        cfg = _load_ssm_params(session=session, app_name=app_name, region_name=region)
        logging.info(f"Using AWS profile: {profile_name}, user = {user}, region: {region}, passed config loading step.")

        # map SSM keys to UploadClient args and validate presence
        file_bucket_name = cfg.get("storage/file_bucket_name")
        job_table_name = cfg.get("storage/job_table_name")
        lock_table_name = cfg.get("storage/lock_table_name")
        if not (file_bucket_name and job_table_name and lock_table_name):
            missing = [k for k in ("storage/file_bucket_name", "storage/job_table_name", "storage/lock_table_name") if not cfg.get(k)]
            raise RuntimeError(f"Missing required SSM params for app {app_name}: {missing}")

        # construct UploadClient; it will create its own boto3 clients using the region
        self._upload_client = UploadClient(
            region_name=region,
            user=user,
            file_bucket_name=file_bucket_name,
            job_table_name=job_table_name,
            lock_table_name=lock_table_name,
        )

        logging.info('Instantiation complete.')

    def upload_imagery(self, csv_path: str, *, summary: str = "") -> Tuple[bool, Dict]:
        return self._upload_client.start_upload_job_from_csv(csv_path, summary=summary)