import json
import logging
from typing import Literal, Union
from datetime import datetime, timezone

import boto3

logger = logging.getLogger(__name__)
firehose = boto3.client("firehose")

LogLevel = Literal["info", "warning", "error"]

def log(job_id: str,
        user: str,
        event_type: str,
        stream_name: str,
        message: str,
        level: Union[LogLevel, str] = "info") -> None:

    # Best-effort normalization + validation
    lvl = str(level).strip().lower()

    if lvl not in ("info", "warning", "error"):
        logger.error(
            json.dumps(
                {
                    "job_id": job_id,
                    "user": user,
                    "event_type": event_type,
                    "level": "error",
                    "message": message,
                    "error": f"Invalid log level: {level!r}",
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                },
                ensure_ascii=False,
            )
        )
        return

    entry = {
        "job_id": job_id,
        "user": user,
        "event_type": event_type,
        "level": lvl,
        "message": message,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    line = json.dumps(entry, ensure_ascii=False)

    if lvl == "error":
        logger.error(line)
    elif lvl == "warning":
        logger.warning(line)
    else:
        logger.info(line)

    try:
        firehose.put_record(
            DeliveryStreamName=stream_name,
            Record={"Data": (line + "\n").encode("utf-8")},
        )
    except Exception as e:
        logger.error(
            json.dumps(
                {
                    "job_id": job_id,
                    "user": user,
                    "event_type": event_type,
                    "level": "error",
                    "message": "Failed to put log to Firehose",
                    "stream_name": stream_name,
                    "firehose_error": str(e),
                    "original_level": lvl,
                    "original_message": message,
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                },
                ensure_ascii=False,
            )
        )
