"""
Local Streamlit viewer for CVDMS dataset visualization artifacts.

Run from the project root:

    streamlit run visualization_tool/cvdms_dataset_viewer.py

Or from inside visualization_tool:

    streamlit run cvdms_dataset_viewer.py

This app reads JSON artifacts produced by the CVDMS Dataset Visualization Lambda
under:

    s3://<datasets-bucket>/datasets/<dataset_id>/v<version>/visualization/

Expected artifacts:
- overview.json
- class_distribution_by_split.json
- source_distribution_by_split.json
- source_split_resolution_by_split.json
- quality_distribution_by_split.json
- numeric_feature_summary.json
- numeric_feature_histograms.json
- split_comparison_metrics.json
"""

import os
from typing import Any

import streamlit as st

from helpers.artifact_loader import (
    load_visualization_bundle,
    summarize_bundle_status,
)
from helpers.charts import (
    render_bundle_status,
    render_class_balance,
    render_numeric_features,
    render_overview,
    render_quality_buckets,
    render_raw_artifacts,
    render_source_balance,
    render_source_split_resolution,
    render_split_comparison,
    render_suggestions,
)
from helpers.s3_loader import (
    S3LoadError,
    create_s3_client,
    list_dataset_ids,
    list_dataset_versions,
)

APP_TITLE = "CVDMS Dataset Viewer"

def _init_session_state() -> None:
    defaults = {
        "aws_profile": os.environ.get("AWS_PROFILE", ""),
        "aws_region": os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "")),
        "datasets_bucket": os.environ.get("CVDMS_DATASETS_BUCKET", ""),
        "manual_dataset_id": "",
        "require_visualization_for_dataset_list": False,
        "show_raw_artifacts": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def _clean_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None

@st.cache_resource(show_spinner=False)
def _cached_s3_client(
    profile_name: str | None,
    region_name: str | None,
):
    return create_s3_client(
        profile_name=profile_name,
        region_name=region_name,
    )

@st.cache_data(show_spinner=False, ttl=60)
def _cached_dataset_ids(
    *,
    profile_name: str | None,
    region_name: str | None,
    bucket: str,
    require_visualization_artifacts: bool,
) -> list[str]:
    s3_client = create_s3_client(
        profile_name=profile_name,
        region_name=region_name,
    )
    return list_dataset_ids(
        s3_client,
        bucket=bucket,
        require_visualization_artifacts=require_visualization_artifacts,
    )

@st.cache_data(show_spinner=False, ttl=60)
def _cached_dataset_versions(
    *,
    profile_name: str | None,
    region_name: str | None,
    bucket: str,
    dataset_id: str,
    require_visualization_artifacts: bool,
) -> list[int]:
    s3_client = create_s3_client(
        profile_name=profile_name,
        region_name=region_name,
    )
    return list_dataset_versions(
        s3_client,
        bucket=bucket,
        dataset_id=dataset_id,
        require_visualization_artifacts=require_visualization_artifacts,
    )

@st.cache_data(show_spinner=False, ttl=30)
def _cached_bundle_status(
    *,
    profile_name: str | None,
    region_name: str | None,
    bucket: str,
    dataset_id: str,
    version: int,
) -> dict[str, Any]:
    """
    Small cached debug/status summary.

    The actual bundle is not cached here because Streamlit can handle loading
    it directly and we want object methods/properties available in rendering.
    """
    s3_client = create_s3_client(
        profile_name=profile_name,
        region_name=region_name,
    )
    bundle = load_visualization_bundle(
        s3_client,
        bucket=bucket,
        dataset_id=dataset_id,
        version=version,
        strict=False,
    )
    return summarize_bundle_status(bundle)

def _clear_caches() -> None:
    _cached_dataset_ids.clear()
    _cached_dataset_versions.clear()
    _cached_bundle_status.clear()

def _render_header() -> None:
    st.title(APP_TITLE)
    st.caption(
        "A local TensorBoard-like viewer for CVDMS dataset version artifacts."
    )

def _render_sidebar() -> dict[str, Any] | None:
    st.sidebar.header("Connection")

    aws_profile = st.sidebar.text_input(
        "AWS profile",
        value=st.session_state.aws_profile,
        help=(
            "Optional. Leave blank to use the default boto3 credential chain. "
            "Example: default, personal, cvdms-dev."
        ),
    )
    st.session_state.aws_profile = aws_profile

    aws_region = st.sidebar.text_input(
        "AWS region",
        value=st.session_state.aws_region,
        help="Optional. Example: us-east-1, us-west-2.",
    )
    st.session_state.aws_region = aws_region

    datasets_bucket = st.sidebar.text_input(
        "Datasets bucket",
        value=st.session_state.datasets_bucket,
        help="The CVDMS datasets bucket containing datasets/<dataset_id>/v<version>/...",
    )
    st.session_state.datasets_bucket = datasets_bucket

    require_visualization_for_dataset_list = st.sidebar.checkbox(
        "Only list datasets with visualization artifacts",
        value=st.session_state.require_visualization_for_dataset_list,
        help=(
            "When enabled, the dataset dropdown only shows datasets where at least "
            "one version has visualization/overview.json."
        ),
    )
    st.session_state.require_visualization_for_dataset_list = (
        require_visualization_for_dataset_list
    )

    col_refresh, col_clear = st.sidebar.columns(2)
    with col_refresh:
        if st.button("Refresh", use_container_width=True):
            _clear_caches()
            st.rerun()

    with col_clear:
        if st.button("Clear cache", use_container_width=True):
            _clear_caches()
            st.success("Cache cleared.")

    profile_name = _clean_optional_text(aws_profile)
    region_name = _clean_optional_text(aws_region)
    bucket = datasets_bucket.strip()

    if not bucket:
        st.info("Enter your CVDMS datasets bucket in the sidebar to begin.")
        return None

    try:
        _cached_s3_client(profile_name, region_name)
    except S3LoadError as exc:
        st.error(str(exc))
        return None
    except Exception as exc:
        st.error(f"Failed to create S3 client: {type(exc).__name__}: {exc}")
        return None

    st.sidebar.header("Dataset")

    dataset_ids: list[str] = []
    try:
        with st.spinner("Discovering datasets..."):
            dataset_ids = _cached_dataset_ids(
                profile_name=profile_name,
                region_name=region_name,
                bucket=bucket,
                require_visualization_artifacts=require_visualization_for_dataset_list,
            )
    except Exception as exc:
        st.sidebar.warning(
            f"Could not discover dataset IDs automatically: {type(exc).__name__}: {exc}"
        )

    selected_from_dropdown: str | None = None
    if dataset_ids:
        selected_from_dropdown = st.sidebar.selectbox(
            "Dataset ID",
            options=dataset_ids,
            index=0,
            help="Discovered from s3://<bucket>/datasets/",
        )
    else:
        st.sidebar.info("No datasets discovered. Use manual dataset ID below.")

    manual_dataset_id = st.sidebar.text_input(
        "Manual dataset ID override",
        value=st.session_state.manual_dataset_id,
        help=(
            "Optional. If provided, this value overrides the dropdown. "
            "Useful when S3 discovery is unavailable or you know the dataset ID."
        ),
    )
    st.session_state.manual_dataset_id = manual_dataset_id

    dataset_id = manual_dataset_id.strip() or selected_from_dropdown

    if not dataset_id:
        st.info("Select or enter a dataset ID to continue.")
        return None

    st.sidebar.header("Version")

    try:
        with st.spinner("Discovering versions..."):
            versions = _cached_dataset_versions(
                profile_name=profile_name,
                region_name=region_name,
                bucket=bucket,
                dataset_id=dataset_id,
                require_visualization_artifacts=True,
            )
    except Exception as exc:
        st.sidebar.warning(
            f"Could not discover versions automatically: {type(exc).__name__}: {exc}"
        )
        versions = []

    version: int | None = None

    if versions:
        version = st.sidebar.selectbox(
            "Version",
            options=versions,
            index=len(versions) - 1,
            format_func=lambda v: f"v{v}",
            help="Discovered versions containing visualization/overview.json.",
        )
    else:
        st.sidebar.info(
            "No visualization-ready versions discovered. Enter a version manually."
        )
        manual_version = st.sidebar.number_input(
            "Manual version",
            min_value=1,
            value=1,
            step=1,
        )
        version = int(manual_version)

    st.sidebar.header("Display")

    show_raw_artifacts = st.sidebar.checkbox(
        "Show raw artifacts tab",
        value=st.session_state.show_raw_artifacts,
        help="Adds a debugging tab for inspecting the raw loaded JSON.",
    )
    st.session_state.show_raw_artifacts = show_raw_artifacts

    return {
        "profile_name": profile_name,
        "region_name": region_name,
        "bucket": bucket,
        "dataset_id": dataset_id,
        "version": version,
        "show_raw_artifacts": show_raw_artifacts,
    }

def _load_bundle(config: dict[str, Any]):
    s3_client = create_s3_client(
        profile_name=config["profile_name"],
        region_name=config["region_name"],
    )

    return load_visualization_bundle(
        s3_client,
        bucket=config["bucket"],
        dataset_id=config["dataset_id"],
        version=config["version"],
        strict=False,
    )

def _render_selected_context(config: dict[str, Any]) -> None:
    st.markdown(
        f"""
        **Selected dataset:** `{config["dataset_id"]}`  
        **Selected version:** `v{config["version"]}`  
        **Datasets bucket:** `{config["bucket"]}`
        """
    )

def _render_main_dashboard(config: dict[str, Any]) -> None:
    _render_selected_context(config)

    try:
        with st.spinner("Loading visualization artifacts..."):
            bundle = _load_bundle(config)
    except Exception as exc:
        st.error(
            f"Failed to load visualization artifacts: {type(exc).__name__}: {exc}"
        )
        return

    render_bundle_status(bundle)

    tab_names = [
        "Overview",
        "Suggestions",
        "Class Balance",
        "Source Balance",
        "Source Split Resolution",
        "Quality Buckets",
        "Numeric Features",
        "Split Comparison",
    ]

    if config["show_raw_artifacts"]:
        tab_names.append("Raw Artifacts")

    tabs = st.tabs(tab_names)

    with tabs[0]:
        render_overview(bundle)

    with tabs[1]:
        render_suggestions(bundle)

    with tabs[2]:
        render_class_balance(bundle)

    with tabs[3]:
        render_source_balance(bundle)

    with tabs[4]:
        render_source_split_resolution(bundle)

    with tabs[5]:
        render_quality_buckets(bundle)

    with tabs[6]:
        render_numeric_features(bundle)

    with tabs[7]:
        render_split_comparison(bundle)

    if config["show_raw_artifacts"]:
        with tabs[8]:
            render_raw_artifacts(bundle)

def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _init_session_state()
    _render_header()

    config = _render_sidebar()
    if config is None:
        return

    _render_main_dashboard(config)

if __name__ == "__main__":
    main()