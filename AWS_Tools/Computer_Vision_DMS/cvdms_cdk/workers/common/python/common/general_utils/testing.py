# Simulate failing to:
#     one shard fails deterministically
#     the workflow goes to DLQ
#     you do not create chaos across all workers
#
# The current shard filter is a good fit for:
#
# registration file batcher / registration batch worker target shards
# registration ingest target shard items
#
# But it will not match:
#
#     validation steps
#     dedup steps
#     registration ingest label_owner items, unless you set the filter to something like "owner-000000" instead.
#
# Failpoints currently implemented:
#   1. failpoint = "after_batch_rollback_seed"  <-- inside the reg batch job def after making rollback seed
#   2. failpoint = "after_target_rollback_plan" <-- inside the reg ingest map lambda, after writing rollback plan, before table insertions
#   3. failpoint = "after_canonical_imagery_insert" <-- inside the reg ingest map lambda, after writing to canonical table, before other tables, like image_labels etc
#
# to run, set vars:
#     TEST_FAIL_ENABLED = True
#     TEST_FAILPOINT = "after_batch_rollback_seed"
#     TEST_FAIL_SHARD = "000000" or  "000001" etc.. if enough shards are generated to produce these shard numbers.
#
#  or for ingest:
#     TEST_FAIL_ENABLED = True
#     TEST_FAILPOINT = "after_target_rollback_plan" or "after_canonical_imagery_insert"
#     TEST_FAIL_SHARD = "000000" or  "000001" etc.. if enough shards are generated to produce these shard numbers.
#
# Defaults for normal running
# TEST_FAIL_ENABLED = False
# TEST_FAILPOINT = ""
# TEST_FAIL_SHARD = ""

TEST_FAIL_ENABLED = False # Keep False when not testing dlq and triggering failures
TEST_FAILPOINT = "after_canonical_imagery_insert"
TEST_FAIL_SHARD = "000001"

def maybe_fail(failpoint: str, shard_name: str) -> None:
    if not TEST_FAIL_ENABLED:
        return

    if TEST_FAILPOINT != failpoint:
        return

    if TEST_FAIL_SHARD and TEST_FAIL_SHARD != shard_name:
        return

    raise RuntimeError(f"[TEST_FAILPOINT] triggered failpoint={failpoint} shard={shard_name}")