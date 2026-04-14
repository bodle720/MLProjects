# EuroSAT → single-label classification, no --split supported
# BigEarthNet v2.0 → multi-label classification, supports --split for train, val, or test
# Coco (Coco 2017) → object-detection, semantic-segmentation and instance-segmentation, supports --split for train or val
#
# Note: max_items affects selection/upload/manifest size, not upstream download size or amount of data extracted
    # max_items affects selection/upload/manifest size

#Example CLI call:
# --------------------------------------------------------
# cd cvdms_cdk
# python -m dataset_bootstrap.main --dataset eurosat --task single-label --aws-profile developers_admin --bucket cv-imagery-for-ml --max-items 5000
# python -m dataset_bootstrap.main --dataset bigearthnet-v2 --task multi-label --aws-profile developers_admin --bucket cv-imagery-for-ml --max-items 5000 --split train --reuse-from-run-dir "C:\cvdms_files\reuse\bigearthnetv2_multilabel"
# python -m dataset_bootstrap.main --dataset coco --task object-detection --aws-profile developers_admin --bucket cv-imagery-for-ml --max-items 5000 --split train --reuse-from-run-dir "C:\cvdms_files\reuse\coco2017_obj_det_sem_seg_inst_seg"

#Reuse behavior summary
# -----------------------------------------------------------------------------
# - Reuse does NOT skip the bootstrap process itself; selected items are still
#   processed and uploaded to S3 for the new run.
#
# - Reuse saves time and disk churn by reusing prior "_work/downloads" and
#   "_work/extracted" assets when they already exist locally.
#
# - Pass the PRIOR RUN FOLDER that contains the "_work" directory, not the
#   "_work" directory itself.
#
# EuroSAT
# - No --reuse-from-run-dir support. The dataset is small enough that reuse is
#   not worth the added complexity.
#
# BigEarthNet v2
# - Supports --reuse-from-run-dir.
# - Reuse is split-friendly: train / val / test all come from the same shared
#   downloaded assets, and --split only changes which metadata rows are sampled.
# - A single BigEarthNet reuse folder is ideal.
#
# COCO
# - Supports --reuse-from-run-dir.
# - Reuse works across tasks and splits, but COCO images are split-specific:
#   train uses train2017 and val uses val2017.
# - Shared assets like annotations_trainval2017.zip and
#   stuffthingmaps_trainval2017.zip can be reused across splits.
# - If a reuse folder does not yet have the requested split's images
#   (for example val2017), that split may need to be downloaded once.
# - In practice, one shared COCO reuse folder is ideal; over time it can
#   accumulate both train2017 and val2017 assets for future runs.
# -----------------------------------------------------------------------------
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAYER_PYTHON_ROOT = PROJECT_ROOT / "workers" / "common" / "python"

if str(LAYER_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(LAYER_PYTHON_ROOT))

import boto3
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    NoCredentialsError,
    ProfileNotFound,
    UnauthorizedSSOTokenError,
)
from dataset_bootstrap.dataset_helpers import DATASET_HELPERS
from dataset_bootstrap.dataset_helpers.common import (
    TASK_CHOICES,
    BootstrapConfig,
    build_run_dir_name,
    ensure_dir,
    task_slug,
    write_csv_manifest,
    write_failures_json,
    write_json,
    write_jsonl_manifest,
)

REUSE_SUPPORTED_DATASETS = {"bigearthnet-v2", "coco"}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a supported public dataset, copy it into your private S3 bucket, and emit CVDMS-ready manifests."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASET_HELPERS.keys()),
        help="Dataset adapter to use.",
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=TASK_CHOICES,
        help="Target CVDMS task. Must be valid for the selected dataset.",
    )
    parser.add_argument(
        "--aws-profile",
        required=True,
        help="Name of AWS config profile for permissions.",
    )
    parser.add_argument(
        "--bucket",
        required=True,
        help="Target private S3 bucket created by the CVDMS storage stack.",
    )
    parser.add_argument(
        "--s3-prefix",
        default="seed-datasets",
        help="Root key prefix under the target S3 bucket.",
    )
    parser.add_argument(
        "--output-root",
        default=r"C:\cvdms_files\runs",
        help="Local output root. A timestamped run folder will be created inside it.",
    )
    parser.add_argument(
        "--aws-region",
        default="us-east-1",
        help="Optional AWS region for boto3 session.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional maximum number of dataset items to ingest.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default=None,
        help=(
            "Official upstream dataset split to bootstrap when applicable. "
            "Required for split-aware datasets like bigearthnet-v2 and coco; "
            "omit for datasets like eurosat."
        ),
    )
    parser.add_argument(
        "--reuse-from-run-dir",
        type=str,
        default=None,
        help=(
            "For --dataset bigearthnet-v2 or coco only. Path to a previous run directory "
            "(not the _work subdirectory, but the dir containing _work folder). Its "
            "_work/downloads and _work/extracted artifacts may be reused to skip "
            "re-download and re-extraction."
        ),
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Seed for deterministic sampling when --max-items is used.",
    )

    return parser.parse_args()

def validate_split(dataset: str, split: str | None) -> str | None:
    if split is not None:
        split = split.strip().lower()

    if dataset == "eurosat":
        if split is not None:
            raise ValueError("eurosat does not support --split; omit it.")
        return None

    if dataset == "bigearthnet-v2":
        if split not in {"train", "val", "test"}:
            raise ValueError(
                "bigearthnet-v2 requires --split with one of: train, val, test."
            )
        return split

    if dataset == "coco":
        if split not in {"train", "val"}:
            raise ValueError(
                "coco currently requires --split with one of: train, val."
            )
        return split

    if split is not None:
        raise ValueError(f"Unexpected --split for dataset={dataset}.")

    return None

def create_validated_aws_session(*, aws_region: str, aws_profile: str) -> boto3.session.Session:
    """
    Fail fast if the AWS profile is missing, credentials are unavailable,
    or the SSO session is expired / not logged in.
    """
    try:
        session = boto3.session.Session(
            region_name=aws_region,
            profile_name=aws_profile,
        )
    except ProfileNotFound as exc:
        raise RuntimeError(
            f"AWS profile '{aws_profile}' was not found in your AWS config."
        ) from exc

    try:
        sts = session.client("sts")
        identity = sts.get_caller_identity()
        account = identity.get("Account", "unknown")
        arn = identity.get("Arn", "unknown")
        print(f"[INFO] AWS auth verified. account={account} arn={arn}")
        return session

    except UnauthorizedSSOTokenError as exc:
        raise RuntimeError(
            f"AWS SSO session for profile '{aws_profile}' is missing or expired. "
            f"Run: aws sso login --profile {aws_profile}"
        ) from exc

    except NoCredentialsError as exc:
        raise RuntimeError(
            f"No AWS credentials could be resolved for profile '{aws_profile}'. "
            f"If this is an SSO profile, run: aws sso login --profile {aws_profile}"
        ) from exc

    except ClientError as exc:
        error = exc.response.get("Error", {}) if hasattr(exc, "response") else {}
        code = error.get("Code", "Unknown")
        message = error.get("Message", str(exc))

        if code in {"ExpiredToken", "InvalidClientTokenId"}:
            raise RuntimeError(
                f"AWS credentials for profile '{aws_profile}' are expired or invalid. "
                f"If this is an SSO profile, run: aws sso login --profile {aws_profile}"
            ) from exc

        raise RuntimeError(
            f"AWS authentication check failed for profile '{aws_profile}': "
            f"{code}: {message}"
        ) from exc

    except BotoCoreError as exc:
        raise RuntimeError(
            f"AWS SDK initialization/authentication failed for profile '{aws_profile}': {exc}"
        ) from exc

def main() -> int:
    args = parse_args()

    try:
        helper = DATASET_HELPERS[args.dataset]
        helper.validate_task(args.task)
        normalized_split = validate_split(args.dataset, args.split)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    if args.max_items is not None and args.max_items <= 0:
        print("max_items, when specified, must be a positive integer.", file=sys.stderr)
        return 1

    reuse_from_run_dir = None
    if args.dataset == "bigearthnet-v2":
        print(
            "WARNING: bigearthnet-v2 is a very large download and may require tens of gigabytes "
            "of disk space and a long download time.",
            file=sys.stderr,
        )

    if args.reuse_from_run_dir:
        if args.dataset not in REUSE_SUPPORTED_DATASETS:
            print(
                f"[ERROR] reuse_from_run_dir is not supported for dataset={args.dataset}",
                file=sys.stderr,
            )
            return 2

        reuse_from_run_dir = Path(args.reuse_from_run_dir)
        if not reuse_from_run_dir.exists():
            print(f"[ERROR] reuse_from_run_dir does not exist: {reuse_from_run_dir}", file=sys.stderr)
            return 2
        if not reuse_from_run_dir.is_dir():
            print(f"[ERROR] reuse_from_run_dir is not a directory: {reuse_from_run_dir}", file=sys.stderr)
            return 2

        print(f"[INFO] reusing prior source artifacts from: {reuse_from_run_dir}")

    # Fail fast on missing / expired AWS auth before creating run folders
    try:
        session = create_validated_aws_session(
            aws_region=args.aws_region,
            aws_profile=args.aws_profile,
        )
        s3_client = session.client("s3")
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    output_root = Path(args.output_root)
    started_at = datetime.now(timezone.utc)

    run_dir_name = build_run_dir_name(args.dataset, args.task, started_at)
    if normalized_split:
        run_dir_name = f"{run_dir_name}_{normalized_split}"

    output_dir = output_root / run_dir_name
    work_dir = output_dir / "_work"

    ensure_dir(output_root)
    ensure_dir(output_dir)
    ensure_dir(work_dir)

    config = BootstrapConfig(
        dataset=args.dataset,
        task=args.task,
        bucket=args.bucket,
        s3_prefix=args.s3_prefix,
        aws_region=args.aws_region,
        reuse_from_run_dir=reuse_from_run_dir,
        max_items=args.max_items,
        sample_seed=args.sample_seed,
        split=normalized_split,
        output_dir=output_dir,
        work_dir=work_dir,
    )

    try:
        result = helper.bootstrap(config=config, s3_client=s3_client)
    except NotImplementedError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"[ERROR] Bootstrap failure: {exc}", file=sys.stderr)
        return 1

    manifest_jsonl_path = output_dir / "manifest.jsonl"
    manifest_csv_path = output_dir / "manifest.csv"
    summary_path = output_dir / "summary.json"
    failures_path = output_dir / "failures.json"

    write_jsonl_manifest(manifest_jsonl_path, args.task, result.manifest_rows)
    write_csv_manifest(manifest_csv_path, args.task, result.manifest_rows)
    write_failures_json(failures_path, result.failures)

    finished_at = datetime.now(timezone.utc)
    summary = {
        "dataset": args.dataset,
        "task": args.task,
        "split": normalized_split,
        "task_slug": task_slug(args.task),
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "bucket": args.bucket,
        "s3_prefix": args.s3_prefix,
        "requested_max_items": args.max_items,
        "processed_count": len(result.manifest_rows),
        "failure_count": len(result.failures),
        "output_dir": str(output_dir),
        "manifest_jsonl_path": str(manifest_jsonl_path),
        "manifest_csv_path": str(manifest_csv_path),
        "failures_path": str(failures_path),
        "stats": result.stats,
        "reuse_from_run_dir": str(reuse_from_run_dir) if reuse_from_run_dir else None,
    }

    write_json(summary_path, summary)

    print(f"[OK] dataset={args.dataset} task={args.task} split={normalized_split or 'None'}")
    print(f"[OK] rows={len(result.manifest_rows)} failures={len(result.failures)}")
    print(f"[OK] manifest.jsonl -> {manifest_jsonl_path}")
    print(f"[OK] manifest.csv   -> {manifest_csv_path}")
    print(f"[OK] summary.json   -> {summary_path}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())