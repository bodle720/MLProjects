import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    NoCredentialsError,
    ProfileNotFound,
    UnauthorizedSSOTokenError,
)

from .bootstrap import VALID_SPLITS, bootstrap_wheat_head_2021
from .common import (
    BootstrapConfig,
    build_run_dir_name,
    ensure_dir,
    write_failures_json,
    write_json,
    write_object_detection_csv_manifest,
    write_object_detection_jsonl_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download Global Wheat Head Detection 2021, upload selected images to S3, "
            "and emit CVDMS-ready object-detection manifests."
        )
    )
    parser.add_argument(
        "--aws-profile",
        required=True,
        help="Name of AWS config profile with access to the target CVDMS S3 bucket.",
    )
    parser.add_argument(
        "--bucket",
        required=True,
        help="Target private S3 bucket created by the CVDMS storage stack.",
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=sorted(VALID_SPLITS),
        help="Official GWHD 2021 split to bootstrap.",
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
        help=(
            "Optional maximum number of positive-image records to process. "
            "This affects selected/uploaded manifest rows, not upstream download size."
        ),
    )
    parser.add_argument(
        "--reuse-from-run-dir",
        type=str,
        default=None,
        help=(
            "Path to a previous wheat_head_2021 run directory. "
            "The script may reuse its _work/downloads and _work/extracted assets."
        ),
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Seed for deterministic sampling when --max-items is used.",
    )
    return parser.parse_args()


def create_validated_aws_session(*, aws_region: str, aws_profile: str) -> boto3.session.Session:
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


def validate_args(args: argparse.Namespace) -> Path | None:
    if args.max_items is not None and args.max_items <= 0:
        raise ValueError("max_items, when specified, must be a positive integer.")

    if args.reuse_from_run_dir is None:
        return None

    reuse_from_run_dir = Path(args.reuse_from_run_dir)
    if not reuse_from_run_dir.exists():
        raise ValueError(f"reuse_from_run_dir does not exist: {reuse_from_run_dir}")
    if not reuse_from_run_dir.is_dir():
        raise ValueError(f"reuse_from_run_dir is not a directory: {reuse_from_run_dir}")

    return reuse_from_run_dir


def main() -> int:
    args = parse_args()

    try:
        reuse_from_run_dir = validate_args(args)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    if reuse_from_run_dir is not None:
        print(f"[INFO] reusing prior source artifacts from: {reuse_from_run_dir}")

    print(
        "WARNING: Global Wheat Head Detection 2021 is a large download "
        "(roughly 10 GB). Reuse a prior run directory when possible.",
        file=sys.stderr,
    )

    try:
        session = create_validated_aws_session(
            aws_region=args.aws_region,
            aws_profile=args.aws_profile,
        )
        s3_client = session.client("s3")
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    output_root = Path(args.output_root)
    started_at = datetime.now(timezone.utc)

    run_dir_name = build_run_dir_name(args.split, started_at)
    output_dir = output_root / run_dir_name
    work_dir = output_dir / "_work"

    ensure_dir(output_root)
    ensure_dir(output_dir)
    ensure_dir(work_dir)

    config = BootstrapConfig(
        bucket=args.bucket,
        s3_prefix=args.s3_prefix,
        aws_region=args.aws_region,
        reuse_from_run_dir=reuse_from_run_dir,
        max_items=args.max_items,
        sample_seed=args.sample_seed,
        split=args.split,
        output_dir=output_dir,
        work_dir=work_dir,
    )

    try:
        result = bootstrap_wheat_head_2021(config=config, s3_client=s3_client)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Bootstrap failure: {exc}", file=sys.stderr)
        return 1

    manifest_jsonl_path = output_dir / "manifest.jsonl"
    manifest_csv_path = output_dir / "manifest.csv"
    summary_path = output_dir / "summary.json"
    failures_path = output_dir / "failures.json"

    try:
        write_object_detection_jsonl_manifest(manifest_jsonl_path, result.manifest_rows)
        write_object_detection_csv_manifest(manifest_csv_path, result.manifest_rows)
        write_failures_json(failures_path, result.failures)

        finished_at = datetime.now(timezone.utc)
        summary = {
            "dataset": "wheat_head_2021",
            "upstream_dataset": "Global Wheat Head Detection 2021",
            "task": "object-detection",
            "split": args.split,
            "class_name": "wheat_head",
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

    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed writing output artifacts: {exc}", file=sys.stderr)
        return 1

    print(f"[OK] dataset=wheat_head_2021 task=object-detection split={args.split}")
    print(f"[OK] rows={len(result.manifest_rows)} failures={len(result.failures)}")
    print(f"[OK] manifest.jsonl -> {manifest_jsonl_path}")
    print(f"[OK] manifest.csv   -> {manifest_csv_path}")
    print(f"[OK] summary.json   -> {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())