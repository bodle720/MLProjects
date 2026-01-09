import json
import base64
from datetime import datetime, timezone

ALLOWED_LEVELS = {"info", "warning", "error"}

def _normalize_level(v) -> str:
    s = str(v).strip().lower()
    if s in ALLOWED_LEVELS:
        return s
    # common synonyms
    if s in ("warn", "warning"):
        return "warning"
    if s in ("err", "error", "fatal", "critical"):
        return "error"
    if s in ("info", "information", "debug", "trace"):
        return "info"
    return "warning"

def _normalize_timestamp(ts) -> str:
    # Return ISO8601 UTC "YYYY-mm-ddTHH:MM:SSZ"
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if isinstance(ts, str):
        s = ts.strip()
        if not s:
            return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # If already looks like ISO Z, keep it (optionally you could parse+reformat)
        if "T" in s and (s.endswith("Z") or s.endswith("z")):
            return s[:-1] + "Z"

        # Try a couple formats you might produce elsewhere
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
                return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                pass

        # Fallback: keep as string, but this may break parquet timestamp conversion
        # Better to coerce to now rather than poison the stream:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def normalize_record(record_body: str) -> dict:
    try:
        obj = json.loads(record_body)
        if not isinstance(obj, dict):
            obj = {"message": record_body}
    except Exception:
        preview = record_body[:200].replace("\n", "\\n")
        print(f"[FIREHOSE_LOG_TRANSFORMER] Could not parse JSON. Preview={preview!r}")
        obj = {"message": record_body}

    job_id = obj.get("job_id") or "UNKNOWN"
    user = obj.get("user") or "UNKNOWN"
    event_type = obj.get("event_type") or "UNKNOWN"
    message = obj.get("message") or "UNKNOWN"

    out = {
        "job_id": str(job_id),
        "user": str(user),
        "event_type": str(event_type),
        "level": _normalize_level(obj.get("level")),
        "message": str(message),
        "timestamp": _normalize_timestamp(obj.get("timestamp")),
    }
    return out

def handler(event, context):
    output = {"records": []}

    for rec in event.get("records", []):
        rec_id = rec.get("recordId", "UNKNOWN")

        try:
            raw_bytes = base64.b64decode(rec.get("data") or b"")
            raw = raw_bytes.decode("utf-8", errors="replace")
            raw = raw.lstrip("\ufeff").strip()
        except Exception as e:
            print(f"[FIREHOSE_LOG_TRANSFORMER] decode failed rec_id={rec_id}: {e}")
            raw = ""

        normalized = normalize_record(raw)

        out_data = json.dumps(normalized, ensure_ascii=False) + "\n"
        out_b64 = base64.b64encode(out_data.encode("utf-8")).decode("utf-8")

        output["records"].append({
            "recordId": rec_id,
            "result": "Ok",
            "data": out_b64
        })

    return output