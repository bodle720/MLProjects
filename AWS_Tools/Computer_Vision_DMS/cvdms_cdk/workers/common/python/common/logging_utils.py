import json
import logging
from datetime import datetime, timezone

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

firehose = boto3.client("firehose")

def log(job_id, user, event_type, message, stream_name, warning=None, error=None, level="info"):
    entry = {
        "job_id": job_id,
        "user": user,
        "event_type": event_type,
        "message": message,
        "warning": warning,
        "error": error,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    }

    # CloudWatch log for operational visibility
    line = json.dumps(entry)
    if level.lower() == "error":
        logger.error(line)
    else:
        logger.info(line)

    # Firehose DirectPut (JSON line)
    try:
        firehose.put_record(
            DeliveryStreamName=stream_name,
            Record={"Data": (line + "\n").encode("utf-8")}
        )
    except Exception as e:
        # Do not fail the handler—the design prefers non-DLQ behavior.
        # Optionally log the failure; avoid recursion by not calling log() again.
        logger.error(json.dumps({
            "job_id": job_id,
            "user": user,
            "event_type": event_type,
            "message": "Failed to put log to Firehose",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        }))