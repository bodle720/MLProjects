from pathlib import Path

import mlflow


def download_mlflow_model_artifact(
    mlflow_tracking_uri: str,
    model_uri: str,
    run_dir: Path,
) -> dict:
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    artifact_dir = run_dir / "mlflow_model_artifact"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    downloaded_path = mlflow.artifacts.download_artifacts(
        artifact_uri=model_uri,
        dst_path=str(artifact_dir),
    )
    downloaded_path = Path(downloaded_path)

    weights_path = find_best_pt_file(downloaded_path)

    return {
        "mlflow_tracking_uri": mlflow_tracking_uri,
        "model_uri": model_uri,
        "downloaded_model_path": str(downloaded_path),
        "weights_path": str(weights_path),
    }


def find_best_pt_file(model_dir: Path) -> Path:
    pt_files = sorted(model_dir.rglob("*.pt"))

    if not pt_files:
        raise FileNotFoundError(
            f"No .pt file found in downloaded MLflow model artifact: {model_dir}"
        )

    best_named = [path for path in pt_files if path.name == "best.pt"]

    if best_named:
        return best_named[0]

    return pt_files[0]