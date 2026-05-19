import json
import posixpath
from pathlib import Path

from mlflow.tracking import MlflowClient

from helpers import sweep_settings as settings
from helpers.sweep_io import slugify


def _safe_read_json(path: Path) -> dict | None:
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_read_yaml(path: Path) -> dict | None:
    if not path.exists():
        return None

    try:
        import yaml
    except ImportError:
        return None

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if isinstance(data, dict):
        return data

    return None


def _list_artifacts_safe(client: MlflowClient, run_id: str, artifact_path: str | None = None):
    try:
        return client.list_artifacts(run_id, artifact_path)
    except Exception:
        return []


def artifact_exists(client: MlflowClient, run_id: str, artifact_path: str) -> bool:
    parent = posixpath.dirname(artifact_path)
    name = posixpath.basename(artifact_path)

    listed_parent = parent if parent else None
    artifacts = _list_artifacts_safe(client, run_id, listed_parent)

    for artifact in artifacts:
        if posixpath.basename(artifact.path) == name and artifact.path == artifact_path:
            return True

    return False


def find_first_existing_artifact(
    client: MlflowClient,
    run_id: str,
    candidate_paths: list[str],
) -> str | None:
    for artifact_path in candidate_paths:
        if artifact_exists(client, run_id, artifact_path):
            return artifact_path

    return None


def recursively_find_artifacts_by_filename(
    client: MlflowClient,
    run_id: str,
    filename: str,
    start_path: str | None = None,
) -> list[str]:
    matches = []
    stack = [start_path]

    while stack:
        current_path = stack.pop()
        artifacts = _list_artifacts_safe(client, run_id, current_path)

        for artifact in artifacts:
            if artifact.is_dir:
                stack.append(artifact.path)
                continue

            if posixpath.basename(artifact.path) == filename:
                matches.append(artifact.path)

    return sorted(set(matches))


def find_best_pt_artifact(client: MlflowClient, run_id: str) -> str | None:
    explicit_match = find_first_existing_artifact(
        client=client,
        run_id=run_id,
        candidate_paths=settings.BEST_PT_ARTIFACT_CANDIDATES,
    )

    if explicit_match is not None:
        return explicit_match

    matches = recursively_find_artifacts_by_filename(
        client=client,
        run_id=run_id,
        filename="best.pt",
    )

    if not matches:
        return None

    preferred = [
        path for path in matches
        if "weight" in path.lower() or path.endswith("best.pt")
    ]

    if preferred:
        return preferred[0]

    return matches[0]


def find_optional_metadata_artifacts(client: MlflowClient, run_id: str) -> dict:
    artifacts = {}

    args_yaml = find_first_existing_artifact(
        client=client,
        run_id=run_id,
        candidate_paths=settings.ARGS_YAML_ARTIFACT_CANDIDATES,
    )
    if args_yaml is None:
        matches = recursively_find_artifacts_by_filename(client, run_id, "args.yaml")
        args_yaml = matches[0] if matches else None

    config_snapshot = find_first_existing_artifact(
        client=client,
        run_id=run_id,
        candidate_paths=settings.CONFIG_SNAPSHOT_ARTIFACT_CANDIDATES,
    )
    if config_snapshot is None:
        matches = recursively_find_artifacts_by_filename(client, run_id, "config_snapshot.yaml")
        config_snapshot = matches[0] if matches else None

    training_summary = find_first_existing_artifact(
        client=client,
        run_id=run_id,
        candidate_paths=settings.TRAINING_SUMMARY_ARTIFACT_CANDIDATES,
    )
    if training_summary is None:
        matches = recursively_find_artifacts_by_filename(client, run_id, "training_run_summary.json")
        training_summary = matches[0] if matches else None

    artifacts["args_yaml_artifact_path"] = args_yaml
    artifacts["config_snapshot_artifact_path"] = config_snapshot
    artifacts["training_summary_artifact_path"] = training_summary

    return artifacts


def make_run_artifact_download_dir(download_root: Path, run_name: str, run_id: str) -> Path:
    run_slug = slugify(run_name)
    path = download_root / f"{run_slug}_{run_id[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_artifact(
    client: MlflowClient,
    run_id: str,
    artifact_path: str,
    dst_dir: Path,
) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    downloaded_path = client.download_artifacts(
        run_id=run_id,
        path=artifact_path,
        dst_path=str(dst_dir),
    )
    return Path(downloaded_path)


def download_optional_artifact(
    client: MlflowClient,
    run_id: str,
    artifact_path: str | None,
    dst_dir: Path,
) -> Path | None:
    if artifact_path is None:
        return None

    try:
        return download_artifact(client, run_id, artifact_path, dst_dir)
    except Exception:
        return None


def get_file_size_mb(path: Path | None) -> float | None:
    if path is None or not path.exists() or not path.is_file():
        return None

    return path.stat().st_size / (1024 * 1024)


def load_downloaded_metadata(
    args_yaml_path: Path | None,
    config_snapshot_path: Path | None,
    training_summary_path: Path | None,
) -> dict:
    return {
        "args_yaml": _safe_read_yaml(args_yaml_path) if args_yaml_path else None,
        "config_snapshot": _safe_read_yaml(config_snapshot_path) if config_snapshot_path else None,
        "training_summary": _safe_read_json(training_summary_path) if training_summary_path else None,
    }


def prepare_candidate_artifacts(
    client: MlflowClient,
    run_summary: dict,
    download_root: Path,
) -> tuple[dict | None, dict | None]:
    run_id = run_summary["run_id"]
    run_name = run_summary["run_name"]

    best_pt_artifact_path = find_best_pt_artifact(client, run_id)
    if best_pt_artifact_path is None:
        failure = {
            "run_id": run_id,
            "run_name": run_name,
            "reason": "No best.pt artifact found.",
            "artifact_uri": run_summary.get("artifact_uri"),
        }
        return None, failure

    run_download_dir = make_run_artifact_download_dir(download_root, run_name, run_id)

    try:
        best_pt_local_path = download_artifact(
            client=client,
            run_id=run_id,
            artifact_path=best_pt_artifact_path,
            dst_dir=run_download_dir,
        )
    except Exception as exc:
        failure = {
            "run_id": run_id,
            "run_name": run_name,
            "reason": "Failed to download best.pt artifact.",
            "best_pt_artifact_path": best_pt_artifact_path,
            "error": repr(exc),
        }
        return None, failure

    metadata_artifacts = find_optional_metadata_artifacts(client, run_id)

    args_yaml_local_path = download_optional_artifact(
        client=client,
        run_id=run_id,
        artifact_path=metadata_artifacts["args_yaml_artifact_path"],
        dst_dir=run_download_dir,
    )
    config_snapshot_local_path = download_optional_artifact(
        client=client,
        run_id=run_id,
        artifact_path=metadata_artifacts["config_snapshot_artifact_path"],
        dst_dir=run_download_dir,
    )
    training_summary_local_path = download_optional_artifact(
        client=client,
        run_id=run_id,
        artifact_path=metadata_artifacts["training_summary_artifact_path"],
        dst_dir=run_download_dir,
    )

    metadata = load_downloaded_metadata(
        args_yaml_path=args_yaml_local_path,
        config_snapshot_path=config_snapshot_local_path,
        training_summary_path=training_summary_local_path,
    )

    candidate = {
        "run_id": run_id,
        "run_name": run_name,
        "experiment_id": run_summary.get("experiment_id"),
        "status": run_summary.get("status"),
        "artifact_uri": run_summary.get("artifact_uri"),
        "start_time": run_summary.get("start_time"),
        "end_time": run_summary.get("end_time"),
        "mlflow_params": run_summary.get("params", {}),
        "mlflow_metrics": run_summary.get("metrics", {}),
        "mlflow_tags": run_summary.get("tags", {}),
        "best_pt_artifact_path": best_pt_artifact_path,
        "best_pt_local_path": str(best_pt_local_path),
        "model_file_size_mb": get_file_size_mb(best_pt_local_path),
        "args_yaml_artifact_path": metadata_artifacts["args_yaml_artifact_path"],
        "args_yaml_local_path": str(args_yaml_local_path) if args_yaml_local_path else None,
        "config_snapshot_artifact_path": metadata_artifacts["config_snapshot_artifact_path"],
        "config_snapshot_local_path": str(config_snapshot_local_path) if config_snapshot_local_path else None,
        "training_summary_artifact_path": metadata_artifacts["training_summary_artifact_path"],
        "training_summary_local_path": str(training_summary_local_path) if training_summary_local_path else None,
        "metadata": metadata,
    }

    return candidate, None


def prepare_all_candidate_artifacts(
    client: MlflowClient,
    run_summaries: list[dict],
    download_root: Path,
) -> tuple[list[dict], list[dict]]:
    candidates = []
    failures = []

    for run_summary in run_summaries:
        candidate, failure = prepare_candidate_artifacts(
            client=client,
            run_summary=run_summary,
            download_root=download_root,
        )

        if candidate is not None:
            candidates.append(candidate)

        if failure is not None:
            failures.append(failure)

    return candidates, failures