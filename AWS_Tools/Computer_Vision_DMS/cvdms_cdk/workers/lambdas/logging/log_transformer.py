import json
import base64
from datetime import datetime, timezone

def normalize_record(record_body):
    """
    Accept raw JSON or text and produce stable JSON structure:
    {
      "job_id": "<string>",
      "event_type": "<string>",
      "message": "<string>",
      "warnings": "<string or null>",
      "errors": "<string or null>",
      "timestamp": "<ISO8601 UTC>"
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
    out["event_type"] = obj.get("event_type") or obj.get("type") or "unknown"
    out["message"] = obj.get("message") or obj.get("msg") or ""
    # ensure warnings/errors are strings or null
    out["warnings"] = json.dumps(obj.get("warnings")) if obj.get("warnings") is not None else None
    out["errors"] = json.dumps(obj.get("errors")) if obj.get("errors") is not None else None

    # timestamp handling: accept numeric epoch, ISO strings, or create now
    ts = obj.get("timestamp")
    if ts is None:
        out["timestamp"] = datetime.now(timezone.utc).isoformat()
    else:
        try:
            # epoch in seconds
            if isinstance(ts, (int, float)):
                out["timestamp"] = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            else:
                # try parse as string
                # a simple pass-through (assume valid ISO)
                out["timestamp"] = str(ts)
        except Exception:
            out["timestamp"] = datetime.now(timezone.utc).isoformat()

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
