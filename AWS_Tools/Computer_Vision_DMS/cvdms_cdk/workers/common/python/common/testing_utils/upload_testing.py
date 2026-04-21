# =============================================================================
# Upload flow failpoint summary (runtime-controlled via SSM Parameter Store)
# =============================================================================
#
# Purpose
# -------
# These failpoints exist to intentionally crash specific parts of the UPLOAD
# workflow so we can test DLQ rollback behavior and cleanup correctness without
# changing code or redeploying.
#
# The upload-specific helper lives in common/testing_utils/upload_testing.py
# and is called like:
#
#     maybe_fail("<failpoint_name>", shard)
#
# The helper reads the current config from the SSM parameter named by:
#
#     UPLOAD_TESTING_SSM_PARAM_NAME
#
# That env var is passed into:
#   - upload Batch workers
#   - upload ingest lambdas
#   - any other upload-side code that should support runtime failpoints
#
#
# -----------------------------------------------------------------------------
# Default SSM parameter value
# -----------------------------------------------------------------------------
# Keep the parameter disabled by default:
#
# {
#   "enabled": false,
#   "failpoint_name": null,
#   "shard": null
# }
#
# With enabled=false, all failpoints are inert and the workflow runs normally.
#
# -----------------------------------------------------------------------------
# How to enable a failpoint in the AWS console
# -----------------------------------------------------------------------------
# 1. Open AWS Systems Manager -> Parameter Store
# 2. Open the parameter referenced by UPLOAD_TESTING_SSM_PARAM_NAME
#    Example path shape:
#        /cvdms/<app_name>/upload/testing/fail_control
# 3. Edit the JSON string value
# 4. Set:
#      - "enabled": true
#      - "failpoint_name": "<one of the supported names below>"
#      - "shard": "<target shard id>"   # or null to match any shard
# 5. Save the parameter
# 6. Start a NEW upload job
#
# Example:
#
# {
#   "enabled": true,
#   "failpoint_name": "after_batch_rollback_seed",
#   "shard": "000001"
# }
#
# After testing, set the parameter back to:
#
# {
#   "enabled": false,
#   "failpoint_name": null,
#   "shard": null
# }
#
# -----------------------------------------------------------------------------
# Supported failpoints currently implemented
# -----------------------------------------------------------------------------
#
# 1) after_batch_rollback_seed
# ----------------------------------
# Location:
#   Registration Batch worker main.py
#
# Call site:
#   maybe_fail("after_batch_rollback_seed", shard_name)
#
# Trigger moment:
#   AFTER the registration Batch worker writes the rollback seed JSON under
#   processed/rollback-batch/, but BEFORE it performs the main side effects
#   such as object copies / DynamoDB sha mapping writes / final processed output.
#
# Why this exists:
#   Tests that the DLQ can safely recover when a registration Batch worker fails
#   very early, but only after enough rollback metadata has been persisted to
#   describe what this worker intended to create.
#
# Expected behavior when triggered:
#   - Registration Batch shard fails
#   - Step Functions / DLQ path activates
#   - DLQ should use persisted rollback seed data and quiescence logic to clean
#     up safely
#
# Good example config:
#
# {
#   "enabled": true,
#   "failpoint_name": "after_batch_rollback_seed",
#   "shard": "000001"
# }
#
# 2) after_target_rollback_plan
# ----------------------------------
# Location:
#   Registration ingest MAP lambda
#
# Call site:
#   maybe_fail("after_target_rollback_plan", shard)
#
# Trigger moment:
#   AFTER the target-shard ingest lambda writes its exact rollback plan under
#   processed/rollback/, but BEFORE the target ingest performs its main inserts.
#
# Why this exists:
#   Tests that the ingest-side rollback plan is sufficient and that DLQ can
#   recover when a registration target ingest unit fails immediately after
#   recording the exact candidate-new keys it may insert.
#
# Expected behavior when triggered:
#   - Registration ingest map item fails
#   - Rollback plan exists for that ingest unit
#   - DLQ should remove exact candidate-new image_labels and
#     image_source_membership rows if needed, plus any other job-created assets
#
# Good example config:
#
# {
#   "enabled": true,
#   "failpoint_name": "after_target_rollback_plan",
#   "shard": "group-target-00000"
# }
#
# Note:
#   After ingest grouping changes, the shard here may be a grouped write-unit
#   name like "group-target-00000" rather than an original worker shard.
#
#
# 3) after_canonical_imagery_insert
# ----------------------------------
# Location:
#   Registration ingest MAP lambda
#
# Call site:
#   maybe_fail("after_canonical_imagery_insert", shard)
#
# Trigger moment:
#   AFTER canonical_imagery rows have been inserted during target ingest, but
#   BEFORE the later insert-only steps finish (for example image_labels and
#   image_source_membership).
#
# Why this exists:
#   Tests a later and more dangerous failure point where part of the ingest work
#   has already happened, including canonical_imagery table mutation, so DLQ
#   must prove it can safely unwind partial registration progress.
#
# Expected behavior when triggered:
#   - Registration ingest map item fails after some writes already succeeded
#   - DLQ should use:
#       * rollback-batch seeds
#       * exact rollback plans
#       * processed registration outputs
#       * canonical/orphan cleanup logic
#     to restore consistency
#
# Good example config:
#
# {
#   "enabled": true,
#   "failpoint_name": "after_canonical_imagery_insert",
#   "shard": "group-target-00000"
# }
#
#
# -----------------------------------------------------------------------------
# Shard matching behavior
# -----------------------------------------------------------------------------
# The helper checks the configured "shard" field against the shard passed into
# maybe_fail(...).
#
# Typical values:
#   - Batch registration worker shard:
#       "000000", "000001", ...
#   - Grouped registration ingest target shard:
#       "group-target-00000", "group-target-00001", ...
#
# If you want the failpoint to match ANY shard, set:
#
#   "shard": null
#
# or possibly:
#
#   "shard": ""
#
# depending on how the helper is implemented.
#
# For precise testing, prefer setting an explicit shard name.
#
#
# -----------------------------------------------------------------------------
# Operational guidance
# -----------------------------------------------------------------------------
# Recommended test pattern:
#
# 1. Set enabled=true and choose exactly one failpoint_name
# 2. Set a single target shard
# 3. Start one upload job
# 4. Observe:
#      - worker logs
#      - Step Functions failure path
#      - DLQ processor logs
#      - rollback correctness in S3 / Iceberg / DynamoDB
# 5. Set enabled=false again after the test
#
# IMPORTANT:
#   Do not leave the testing parameter enabled in normal operation.
#
#
# -----------------------------------------------------------------------------
# Template for future failpoints
# -----------------------------------------------------------------------------
# To add another failpoint later:
#
# 1. Add a call site in code:
#
#       maybe_fail("new_failpoint_name", shard)
#
# 2. Add a new section below in this summary with:
#      - failpoint_name
#      - exact file/location
#      - trigger moment
#      - expected rollback behavior
#      - example JSON config
#
# Suggested entry template:
#
#   N) new_failpoint_name
#   ---------------------
#   Location:
#       <file / function>
#   Call site:
#       maybe_fail("new_failpoint_name", shard)
#   Trigger moment:
#       <what has already happened / what has not happened yet>
#   Why this exists:
#       <what rollback case this validates>
#   Example config:
#       {
#         "enabled": true,
#         "failpoint_name": "new_failpoint_name",
#         "shard": "<target shard>"
#       }
#
# =============================================================================

import json
import os
import time
from typing import Any, Optional

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

    param_name = os.environ.get("UPLOAD_TESTING_SSM_PARAM_NAME")
    if not param_name:
        cfg = {
            "enabled": False,
            "failpoint_name": None,
            "shard": None,
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
            "failpoint_name": None,
            "shard": None,
        }

    _cached_config = parsed
    _cached_at = now
    return parsed


def maybe_fail(failpoint_name: str, shard: Optional[str] = None) -> None:
    cfg = _load_fail_config()

    enabled = bool(cfg.get("enabled", False))
    configured_failpoint = cfg.get("failpoint_name")
    configured_shard = cfg.get("shard")

    if not enabled:
        return

    if configured_failpoint != failpoint_name:
        return

    if configured_shard not in (None, "", shard):
        return

    raise RuntimeError(
        f"Intentional test failure triggered: "
        f"failpoint_name={failpoint_name}, shard={shard}"
    )