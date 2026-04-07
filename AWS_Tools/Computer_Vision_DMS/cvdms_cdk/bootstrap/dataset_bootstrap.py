# '''
# EuroSAT → single-label classification
# BigEarthNet v2.0 → multi-label classification
# SpaceNet 2 → object detection, semantic segmentation, instance segmentation via polygon conversion
#
# Testing
# --------------------------------------------------------
#
# import os
# os.chdir("bootstrap")
#
# class TestingConfig:
#     dataset = "eurosat"
#     task ="single-label"
#     output_root = "C:\Users\brian\Desktop\Doc Local\tests\testing_bootstrapper"
#     bucket = "cv-imagery-for-ml"
#     aws_profile = "developers_admin"
#     aws_region = "us-east-1"
#     s3_prefix = "seed-datasets"
#     max_items = None
#     sample_seed = 42
#     keep_work_dir = True
#     overwrite_existing = True
#
# args = TestingConfig()
# --------------------------------------------------------
#Example CLI call:
#
# python dataset_bootstrap.py --dataset eurosat --task single-label --output-root "C:\Users\brian\Desktop\Doc Local\tests\testing_bootstrapper" --bucket cv-imagery-for-ml --aws-profile developers_admin --max-items 5000#
# --------------------------------------------------------
#
# '''

import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone
import shutil

import boto3

from dataset_helpers import DATASET_HELPERS
from dataset_helpers.common import (
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
        "--output-root",
        required=True,
        help="Local output root. A timestamped run folder will be created inside it.",
    )
    parser.add_argument(
        "--bucket",
        required=True,
        help="Target private S3 bucket created by the CVDMS storage stack.",
    )
    parser.add_argument(
        "--aws-profile",
        required=True,
        help="Name of AWS config profile for permissions.",
    )
    parser.add_argument(
        "--aws-region",
        default="us-east-1",
        help="Optional AWS region for boto3 session.",
    )
    parser.add_argument(
        "--s3-prefix",
        default="seed-datasets",
        help="Root key prefix under the target S3 bucket.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional maximum number of dataset items to ingest.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Seed for deterministic sampling when --max-items is used.",
    )
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Keep intermediate downloaded/extracted files for debugging.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Overwrite S3 objects if they already exist.",
    )
    return parser.parse_args()

def main() -> int:
    args = parse_args()

    helper = DATASET_HELPERS[args.dataset]
    helper.validate_task(args.task)

    output_root = Path(args.output_root)
    started_at = datetime.now(timezone.utc)
    output_dir = output_root / build_run_dir_name(args.dataset, args.task, started_at)
    work_dir = output_dir / "_work"

    ensure_dir(output_root)
    ensure_dir(output_dir)
    ensure_dir(work_dir)

    print("output_dir", output_dir)
    print("work_dir", work_dir)
    print("output_root", output_root)

    config = BootstrapConfig(
        dataset=args.dataset,
        task=args.task,
        bucket=args.bucket,
        s3_prefix=args.s3_prefix,
        aws_region=args.aws_region,
        max_items=args.max_items,
        sample_seed=args.sample_seed,
        output_dir=output_dir,
        work_dir=work_dir,
        keep_work_dir=args.keep_work_dir,
        overwrite_existing=args.overwrite_existing,
    )

    session = boto3.session.Session(region_name=args.aws_region, profile_name=args.aws_profile)
    s3_client = session.client("s3")

    try:
        result = helper.bootstrap(config=config, s3_client=s3_client)
    except NotImplementedError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Bootstrap failed: {exc}", file=sys.stderr)
        return 1

    manifest_jsonl_path = output_dir / "manifest.jsonl"
    manifest_csv_path = output_dir / "manifest.csv"
    summary_path = output_dir / "summary.json"
    failures_path = output_dir / "failures.json"

    write_jsonl_manifest(manifest_jsonl_path, result.manifest_rows)
    write_csv_manifest(manifest_csv_path, result.manifest_rows)
    write_failures_json(failures_path, result.failures)

    finished_at = datetime.now(timezone.utc)
    summary = {
        "dataset": args.dataset,
        "task": args.task,
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
    }

    write_json(summary_path, summary)

    if not args.keep_work_dir and work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)

    print(f"[OK] dataset={args.dataset} task={args.task}")
    print(f"[OK] rows={len(result.manifest_rows)} failures={len(result.failures)}")
    print(f"[OK] manifest.jsonl -> {manifest_jsonl_path}")
    print(f"[OK] manifest.csv   -> {manifest_csv_path}")
    print(f"[OK] summary.json   -> {summary_path}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())