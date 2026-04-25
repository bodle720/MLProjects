"""
Artifact loading layer for the local CVDMS Dataset Viewer.

The visualization Lambda writes versioned JSON artifacts under:

    datasets/<dataset_id>/v<version>/visualization/

This module knows the expected artifact filenames and loads them into a single
bundle that the Streamlit UI can render.

The Lambda currently writes:
- overview.json
- class_distribution_by_split.json
- source_distribution_by_split.json
- source_split_resolution_by_split.json
- quality_distribution_by_split.json
- numeric_feature_summary.json
- numeric_feature_histograms.json
- split_comparison_metrics.json
"""

from dataclasses import dataclass, field
from typing import Any

from botocore.client import BaseClient

from helpers.s3_loader import S3ObjectRef, build_visualization_key, read_json_from_s3

ARTIFACT_FILENAMES: dict[str, str] = {
    "overview": "overview.json",
    "class_distribution": "class_distribution_by_split.json",
    "source_distribution": "source_distribution_by_split.json",
    "source_split_resolution": "source_split_resolution_by_split.json",
    "quality_distribution": "quality_distribution_by_split.json",
    "numeric_summary": "numeric_feature_summary.json",
    "numeric_histograms": "numeric_feature_histograms.json",
    "split_comparison": "split_comparison_metrics.json",
}


@dataclass
class LoadedArtifact:
    name: str
    filename: str
    s3_ref: S3ObjectRef
    payload: dict[str, Any] | None = None
    error: str | None = None

    @property
    def loaded(self) -> bool:
        return self.payload is not None and self.error is None


@dataclass
class VisualizationBundle:
    bucket: str
    dataset_id: str
    version: int
    artifacts: dict[str, LoadedArtifact] = field(default_factory=dict)

    @property
    def visualization_prefix(self) -> str:
        return (
            f"s3://{self.bucket}/datasets/"
            f"{self.dataset_id}/v{self.version}/visualization/"
        )

    @property
    def missing_or_failed(self) -> dict[str, LoadedArtifact]:
        return {
            name: artifact
            for name, artifact in self.artifacts.items()
            if not artifact.loaded
        }

    @property
    def is_complete(self) -> bool:
        return len(self.missing_or_failed) == 0

    def get(self, name: str) -> dict[str, Any]:
        """
        Get a loaded artifact payload by logical name.

        Returns an empty dict when the artifact is missing or failed. This keeps
        rendering code simple and lets the UI show partial results.
        """
        artifact = self.artifacts.get(name)
        if artifact is None or artifact.payload is None:
            return {}
        return artifact.payload

    def require(self, name: str) -> dict[str, Any]:
        """
        Get a loaded artifact payload by logical name, raising if unavailable.
        """
        artifact = self.artifacts.get(name)
        if artifact is None:
            raise KeyError(f"Unknown artifact name: {name!r}")
        if artifact.payload is None:
            raise RuntimeError(
                f"Artifact {name!r} is not loaded: {artifact.error or 'missing payload'}"
            )
        return artifact.payload


def expected_artifact_refs(
    *,
    bucket: str,
    dataset_id: str,
    version: int,
) -> dict[str, S3ObjectRef]:
    """
    Return the expected S3 object refs for all visualization artifacts.
    """
    refs: dict[str, S3ObjectRef] = {}

    for name, filename in ARTIFACT_FILENAMES.items():
        key = build_visualization_key(
            dataset_id=dataset_id,
            version=version,
            filename=filename,
        )
        refs[name] = S3ObjectRef(bucket=bucket, key=key)

    return refs


def load_visualization_bundle(
    s3_client: BaseClient,
    *,
    bucket: str,
    dataset_id: str,
    version: int,
    strict: bool = False,
) -> VisualizationBundle:
    """
    Load all expected visualization artifacts for one dataset version.

    Parameters
    ----------
    s3_client:
        Boto3 S3 client.
    bucket:
        CVDMS datasets bucket.
    dataset_id:
        Dataset ID.
    version:
        Dataset version number.
    strict:
        If True, raise if any artifact fails to load.
        If False, store the error in the bundle and allow partial rendering.

    Returns
    -------
    VisualizationBundle
    """
    bundle = VisualizationBundle(
        bucket=bucket,
        dataset_id=dataset_id,
        version=version,
    )

    refs = expected_artifact_refs(
        bucket=bucket,
        dataset_id=dataset_id,
        version=version,
    )

    for name, s3_ref in refs.items():
        filename = ARTIFACT_FILENAMES[name]

        try:
            payload = read_json_from_s3(
                s3_client,
                bucket=s3_ref.bucket,
                key=s3_ref.key,
            )
            bundle.artifacts[name] = LoadedArtifact(
                name=name,
                filename=filename,
                s3_ref=s3_ref,
                payload=payload,
                error=None,
            )
        except Exception as exc:
            if strict:
                raise

            bundle.artifacts[name] = LoadedArtifact(
                name=name,
                filename=filename,
                s3_ref=s3_ref,
                payload=None,
                error=f"{type(exc).__name__}: {exc}",
            )

    return bundle


def summarize_bundle_status(bundle: VisualizationBundle) -> dict[str, Any]:
    """
    Return a compact status summary useful for debugging/display.
    """
    loaded = []
    failed = {}

    for name, artifact in bundle.artifacts.items():
        if artifact.loaded:
            loaded.append(name)
        else:
            failed[name] = {
                "filename": artifact.filename,
                "uri": artifact.s3_ref.uri,
                "error": artifact.error,
            }

    return {
        "bucket": bundle.bucket,
        "dataset_id": bundle.dataset_id,
        "version": bundle.version,
        "visualization_prefix": bundle.visualization_prefix,
        "is_complete": bundle.is_complete,
        "loaded_artifacts": loaded,
        "missing_or_failed_artifacts": failed,
    }