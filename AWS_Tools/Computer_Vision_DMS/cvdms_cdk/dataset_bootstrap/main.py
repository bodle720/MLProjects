# '''
# EuroSAT → single-label classification
# BigEarthNet v2.0 → multi-label classification
# Coco → object-detection, semantic-segmentation and instance-segmentation
#
# Note: max_items affects selection/upload/manifest size, not upstream download size
    # max_items affects selection/upload/manifest size
    # max_items does not affect the amount of source data downloaded/extracted
    # reuse dirs are not “shrunk” by a smaller previous max_items value
#
# Testing
# --------------------------------------------------------
# class TestingConfig:
#     dataset = "bigearthnet-v2"
#     task ="multi-label"
#     output_root = r"C:\cvdms_tmp"
#     bucket = "cv-imagery-for-ml"
#     aws_profile = "developers_admin"
#     aws_region = "us-east-1"
#     s3_prefix = "seed-datasets"
#     reuse_from_run_dir = None
#     max_items = 1500
#     sample_seed = 42
#
# args = TestingConfig()
# --------------------------------------------------------
#Example CLI call for each task:
# cd cvdms_cdk
# python -m dataset_bootstrap.main --dataset eurosat --task single-label --output-root "C:\cvdms_tmp" --bucket cv-imagery-for-ml --aws-profile developers_admin --max-items 1500
# python -m dataset_bootstrap.main --dataset bigearthnet-v2 --task multi-label --output-root "C:\cvdms_tmp" --bucket cv-imagery-for-ml --aws-profile developers_admin --max-items 1500 --reuse-from-run-dir "C:\cvdms_tmp\reuse\bigearthnetv2_multilabel"
# python -m dataset_bootstrap.main --dataset coco --task object-detection --output-root "C:\cvdms_tmp" --bucket cv-imagery-for-ml --aws-profile developers_admin --max-items 1500 --reuse-from-run-dir "C:\cvdms_tmp\reuse\coco_obj_det_sem_seg_inst_seg"
# python -m dataset_bootstrap.main --dataset coco --task semantic-segmentation --output-root "C:\cvdms_tmp" --bucket cv-imagery-for-ml --aws-profile developers_admin --max-items 1500 --reuse-from-run-dir "C:\cvdms_tmp\reuse\coco_obj_det_sem_seg_inst_seg"
# python -m dataset_bootstrap.main --dataset coco --task instance-segmentation --output-root "C:\cvdms_tmp" --bucket cv-imagery-for-ml --aws-profile developers_admin --max-items 1500 --reuse-from-run-dir "C:\cvdms_tmp\reuse\coco_obj_det_sem_seg_inst_seg"


# --------------------------------------------------------

# Reuse behavior summary
# -----------------------------------------------------------------------------
# - Reuse does NOT skip the bootstrap process itself; selected images are still
#   processed and uploaded to S3 for the new run.

# - Reuse saves time and disk churn by avoiding unnecessary redownloads and
#   re-extraction of source assets when those files already exist locally.

# EuroSAT (single-label)
# - No --reuse-from-run-dir support.
# - The dataset is small enough that keeping reuse logic is not worth the added
#   complexity or local disk usage.
#
# BigEarthNet v2 (multi-label)
# - Supports --reuse-from-run-dir.
# - Pass the PRIOR RUN FOLDER that contains the "_work" directory, not the
#   "_work" directory itself.
#   Example:
#       --reuse-from-run-dir "C:/cvdms_tmp/bigearthnet-v2_multi_label_20260408_011409"
# - Reuse helps avoid redownloading and/or re-extracting large source assets by
#   reusing prior "_work/downloads" and "_work/extracted" contents when present.
# - This is useful because BigEarthNet v2 is large and can take substantial time
#   and disk space to prepare.
#
# COCO (object-detection)
# - Supports --reuse-from-run-dir.
# - Pass the PRIOR RUN FOLDER that contains the "_work" directory, not the
#   "_work" directory itself.
#   Example:
#       --reuse-from-run-dir "C:/cvdms_tmp/coco_object_detection_20260410_230600"
# - For object-detection, reuse may pull from a previous run's:
#     * extracted train2017 image directory
#     * extracted instances_train2017.json
#     * downloaded train2017.zip
#     * downloaded annotations_trainval2017.zip
#
# COCO (semantic-segmentation)
# - Supports --reuse-from-run-dir.
# - Pass the PRIOR RUN FOLDER that contains the "_work" directory, not the
#   "_work" directory itself.
#   Example:
#       --reuse-from-run-dir "C:/cvdms_tmp/coco_semantic_segmentation_20260410_230600"
# - For semantic-segmentation, reuse may pull from a previous run's:
#     * extracted train2017 image directory
#     * downloaded train2017.zip
#     * extracted COCO-Stuff stuffthingmaps train2017 directory
#     * downloaded stuffthingmaps_trainval2017.zip
#     * downloaded cocostuff_labels.txt
    # NOTE:
    # - COCO semantic-segmentation may also point --reuse-from-run-dir at a prior
    #   COCO object-detection run folder to reuse the shared train2017 images and train2017.zip.
    # - It will still need COCO-Stuff-specific assets (stuffthingmaps and labels)
    #   unless those already exist in the referenced run.
    # - COCO object-detection and semantic-segmentation overlap on the shared
    #   train2017 images, but semantic-segmentation is NOT a full superset of
    #   object-detection reuse/downloads.
    # - Object-detection still needs its own annotation assets
    #   (instances_train2017.json / annotations_trainval2017.zip), while
    #   semantic-segmentation needs its own COCO-Stuff assets
    #   (stuffthingmaps + cocostuff_labels.txt).
    # - So a prior semantic-segmentation run can help object-detection reuse the
    #   shared train images, but it may still need to download object-detection-
    #   specific annotation files, and vice versa.
    # COCO (instance-segmentation)
    # - Supports --reuse-from-run-dir.
    # - Pass the PRIOR RUN FOLDER that contains the "_work" directory, not the
    #   "_work" directory itself.
    #   Example:
    #       --reuse-from-run-dir "C:/cvdms_tmp/coco_instance_segmentation_20260410_230600"
    # - For instance-segmentation, reuse may pull from a previous run's:
    #     * extracted train2017 image directory
    #     * extracted instances_train2017.json
    #     * downloaded train2017.zip
    #     * downloaded annotations_trainval2017.zip
    # - In the current code, COCO instance-segmentation and object-detection use
    #   the same upstream COCO image/annotation assets, so either task's prior run
    #   can serve as a strong reuse source for the other.
# -----------------------------------------------------------------------------
# '''
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAYER_PYTHON_ROOT = PROJECT_ROOT / "workers" / "common" / "python"

if str(LAYER_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(LAYER_PYTHON_ROOT))

import boto3

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
        "--reuse-from-run-dir",
        type=str,
        default=None,
        help=(
            "For --dataset bigearthnet-v2 or coco only. Path to a previous run directory "
            "(not the _work subdirectory, but the dir containing  _work folder). Its _work/downloads and _work/extracted "
            "artifacts may be reused to skip re-download and re-extraction.")
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

    return parser.parse_args()

def main() -> int:
    args = parse_args()

    helper = DATASET_HELPERS[args.dataset]

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

    output_root = Path(args.output_root)
    started_at = datetime.now(timezone.utc)
    output_dir = output_root / build_run_dir_name(args.dataset, args.task, started_at)
    work_dir = output_dir / "_work"

    ensure_dir(output_root)
    ensure_dir(output_dir)
    ensure_dir(work_dir)

    if args.max_items is not None and args.max_items <= 0:
        print("max_items, when specified, must be a positive integer.", file=sys.stderr)
        return 1

    config = BootstrapConfig(
        dataset=args.dataset,
        task=args.task,
        bucket=args.bucket,
        s3_prefix=args.s3_prefix,
        aws_region=args.aws_region,
        reuse_from_run_dir=reuse_from_run_dir,
        max_items=args.max_items,
        sample_seed=args.sample_seed,
        output_dir=output_dir,
        work_dir=work_dir
    )

    try:
        helper.validate_task(args.task)
        session = boto3.session.Session(region_name=args.aws_region, profile_name=args.aws_profile)
        s3_client = session.client("s3")
        result = helper.bootstrap(config=config, s3_client=s3_client)
    except NotImplementedError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"[ERROR] Initialization or bootstrap failure: {exc}", file=sys.stderr)
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
        "reuse_from_run_dir": str(reuse_from_run_dir) if reuse_from_run_dir else None
    }

    write_json(summary_path, summary)

    print(f"[OK] dataset={args.dataset} task={args.task}")
    print(f"[OK] rows={len(result.manifest_rows)} failures={len(result.failures)}")
    print(f"[OK] manifest.jsonl -> {manifest_jsonl_path}")
    print(f"[OK] manifest.csv   -> {manifest_csv_path}")
    print(f"[OK] summary.json   -> {summary_path}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())