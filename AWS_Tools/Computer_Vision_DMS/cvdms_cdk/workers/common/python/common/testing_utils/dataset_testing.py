# Testing failpoints are implemented in:
# 1. the create task,        failpoint = "create_fail"
# 2. the update task,        failpoint = "update_fail"
# 3. the delete task,        failpoint = "delete_fail"
# 4. the visualization task, failpoint = "visualize_fail"
# 5. the cleanup task,       failpoint = "cleanup_fail"
#
# The are implemented in SSM similar to the upload stack, but without the shard param.
# See upload_testing.py for more explanation.

import json
import os
import time
from typing import Any

import boto3

_ssm = boto3.client("ssm")

_CACHE_TTL_SEC = 10
_cached_at: float = 0.0
_cached_config: dict[str, Any] | None = None


def _load_fail_config() -> dict[str, Any]:
    global _cached_at, _cached_config

    now = time.time()
    if _cached_config is not None and (now - _cached_at) < _CACHE_TTL_SEC:
        return _cached_config

    param_name = os.environ.get("DATASET_TESTING_SSM_PARAM_NAME")
    if not param_name:
        cfg = {
            "enabled": False,
            "failpoint_name": None
        }
        _cached_config = cfg
        _cached_at = now
        return cfg

    try:
        resp = _ssm.get_parameter(Name=param_name)
        raw = resp["Parameter"]["Value"]
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("testing config is not a JSON object")
    except Exception:
        parsed = {
            "enabled": False,
            "failpoint_name": None
        }

    _cached_config = parsed
    _cached_at = now
    return parsed


def maybe_fail(failpoint_name: str) -> None:
    cfg = _load_fail_config()

    enabled = bool(cfg.get("enabled", False))
    configured_failpoint = cfg.get("failpoint_name")

    if not enabled:
        return

    if configured_failpoint != failpoint_name:
        return

    raise RuntimeError(
        f"Intentional test failure triggered: "
        f"failpoint_name={failpoint_name}"
    )