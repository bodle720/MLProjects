import json
import os

from common.general_utils.logging_utils import log
from common.general_utils.s3_utils import (
    delete_s3_prefix,
    parse_s3_uri,
    read_obj_with_retry,
    write_s3_obj,
)

FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]
BATCH_HANDOFF_FILE_NAME = os.environ.get("BATCH_HANDOFF_FILE_NAME", "map-items.jsonl")

TASK_NAME = "[VAL_FILE_BATCHING]"

# We can tune these constants
MAX_MEMORY_MB = 2048  # from the job definition for validation step
IMAGE_SIZE_MB = 3     # worst-case per image
SAFETY_FACTOR = 0.5   # use only ~50% of memory for image data

max_images = int((MAX_MEMORY_MB * SAFETY_FACTOR) / IMAGE_SIZE_MB)
IMAGES_PER_BATCH = max(1, min(max_images, 200))

def _require_event_key(event: dict, key: str):
    if key not in event:
        raise RuntimeError(f"{TASK_NAME} Validation batching Lambda failed: missing required key {key!r}")
    return event[key]

def handler(event, context):
    job_id = _require_event_key(event, "job_id")
    user = _require_event_key(event, "user")
    event_type = _require_event_key(event, "event_type")
    label_type = _require_event_key(event, "label_type")
    data_source = _require_event_key(event, "data_source")
    source_split = _require_event_key(event, "source_split")
    original_manifest_s3_uri = _require_event_key(event, "original_manifest_s3_uri")

    log(
        job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} Starting batching of images for image upload validation job id {job_id}.",
    )

    main_prefix = f"temp/image-upload/{job_id}/batches/validation-step/"
    manifest_prefix = f"{main_prefix}manifests/"
    handoff_prefix = f"{main_prefix}handoff/"
    handoff_key = f"{handoff_prefix}{BATCH_HANDOFF_FILE_NAME}"

    # Clean this stage subtree so reruns are deterministic.
    delete_s3_prefix(FILE_BUCKET_NAME, main_prefix, TASK_NAME)

    try:
        manifest_bucket, manifest_key = parse_s3_uri(original_manifest_s3_uri, TASK_NAME)
    except ValueError as e:
        raise RuntimeError(f"{TASK_NAME} Invalid original_manifest_s3_uri: {e}")

    resp = read_obj_with_retry(manifest_bucket, manifest_key, TASK_NAME)
    if resp is None:
        raise RuntimeError(f"{TASK_NAME} unable to load s3://{manifest_bucket}/{manifest_key} after retries")

    batch_lines: list[str] = []
    handoff_lines: list[str] = []
    total_images = 0
    batch_count = 0

    def flush_batch() -> None:
        nonlocal batch_lines, handoff_lines, batch_count
        if not batch_lines:
            return

        batch_count += 1
        shard = f"{batch_count:03d}"
        manifest_key_out = f"{manifest_prefix}batch-{shard}.jsonl"
        manifest_content = "\n".join(batch_lines) + "\n"

        manifest_uri = write_s3_obj(
            FILE_BUCKET_NAME,
            manifest_key_out,
            manifest_content,
            "application/x-ndjson",
            TASK_NAME,
        )

        # One line per map item for the Distributed Map S3JsonLItemReader.
        # Keep each item small; global fields stay in the Step Functions state.
        handoff_item = {
            "manifest": manifest_uri,
            "shard": f"batch-{shard}",
        }
        handoff_lines.append(json.dumps(handoff_item, separators=(",", ":")))

        batch_lines = []

    # Stream original manifest so memory stays O(batch_size), not O(file_size).
    for raw in resp["Body"].iter_lines():
        if not raw:
            continue

        line = raw.decode("utf-8-sig").strip()
        if not line:
            continue

        total_images += 1
        batch_lines.append(line)

        if len(batch_lines) >= IMAGES_PER_BATCH:
            flush_batch()

    flush_batch()

    handoff_content = ""
    if handoff_lines:
        handoff_content = "\n".join(handoff_lines) + "\n"

    if total_images == 0:
        raise RuntimeError(f"{TASK_NAME} original manifest contained zero images for job_id={job_id}")

    plan_s3_uri = write_s3_obj(
        FILE_BUCKET_NAME,
        handoff_key,
        handoff_content,
        "application/x-ndjson",
        TASK_NAME,
    )

    result = {
        "plan_bucket": FILE_BUCKET_NAME,
        "plan_key": handoff_key,
        "plan_s3_uri": plan_s3_uri,
        "item_count": batch_count,
        "manifest_count": batch_count,
        "expected_count": total_images,
    }

    log(
        job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        (
            f"{TASK_NAME} Done batching {total_images} total images for image upload validation: "
            f"label_type={label_type}, images_per_batch={IMAGES_PER_BATCH}, "
            f"manifest_count={batch_count}, handoff_s3_uri={plan_s3_uri}"
        ),
    )

    return result