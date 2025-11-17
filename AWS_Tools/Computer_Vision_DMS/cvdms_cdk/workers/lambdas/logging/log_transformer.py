import json
import base64
from datetime import datetime, timezone

def normalize_record(record_body):
    """
    Accept raw JSON or text and produce stable JSON structure:
    {
      "job_id": "<string>",
      "user": "<string>",
      "event_type": "<string>",
      "message": "<string>",
      "warning": "<string or null>",
      "error": "<string or null>",
      "timestamp": "<ISO8601 timestamp UTC>"
    }
    """
    try:
        obj = json.loads(record_body)
    except Exception:
        # not JSON, wrap raw text as message
        obj = {"message": record_body}

    # tolerant normalization
    out = {}
    out["job_id"] = str(obj.get("job_id")) if obj.get("job_id") is not None else None
    out["user"] = str(obj.get("user")) if obj.get("user") is not None else None
    out["event_type"] = obj.get("event_type") or obj.get("type") or "unknown"
    out["message"] = obj.get("message") or obj.get("msg") or ""
    out["warning"] = json.dumps(obj.get("warning")) if obj.get("warning") is not None else None
    out["error"] = json.dumps(obj.get("error")) if obj.get("error") is not None else None

    # timestamp handling: accept numeric epoch, ISO strings, or create now
    ts = obj.get("timestamp")
    if ts is None:
        out["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        out["timestamp"] = ts


    return out

def handler(event, context):
    """
    Firehose transformation Lambda entrypoint.
    Expects event records in the Firehose transform input format.
    Returns records in Firehose transform output format.
    """
    output = {"records": []}

    for rec in event.get("records", []):
        rec_id = rec.get("recordId")
        # Firehose gives base64 data
        try:
            raw = base64.b64decode(rec.get("data")).decode("utf-8")
        except Exception:
            raw = ""

        normalized = normalize_record(raw)
        # Firehose JSON->Parquet conversion expects JSON lines; send back a JSON string per record.
        out_data = json.dumps(normalized) + "\n"
        out_b64 = base64.b64encode(out_data.encode("utf-8")).decode("utf-8")

        output["records"].append({
            "recordId": rec_id,
            "result": "Ok",
            "data": out_b64
        })

    return output