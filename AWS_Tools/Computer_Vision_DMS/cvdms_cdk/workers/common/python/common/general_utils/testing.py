import os

def maybe_fail(failpoint: str, shard_name: str) -> None:
    enabled = os.environ.get("TEST_FAIL_ENABLED", "").strip().lower() in {"1", "true", "yes"}
    if not enabled:
        return

    wanted = os.environ.get("TEST_FAILPOINT", "").strip()
    if wanted != failpoint:
        return

    shard_filter = os.environ.get("TEST_FAIL_SHARD", "").strip()
    if shard_filter and shard_filter != shard_name:
        return

    raise RuntimeError(f"[TEST_FAILPOINT] triggered failpoint={failpoint} shard={shard_name}")