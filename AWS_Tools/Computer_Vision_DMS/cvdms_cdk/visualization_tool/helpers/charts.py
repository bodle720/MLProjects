"""
Streamlit and Plotly rendering helpers for the local CVDMS Dataset Viewer.

This module renders the visualization artifacts created by the CVDMS
visualization Lambda.

The main Streamlit file should stay thin and call these functions.
"""

from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from helpers.formatting import (
    SPLIT_ORDER,
    bucket_label,
    clean_label,
    counts_by_split_to_dataframe,
    feature_label,
    histogram_to_dataframe,
    integer,
    nested_delta_block_to_dataframe,
    number,
    pct,
    percentages_by_split_to_dataframe,
    sorted_splits,
    summary_stats_to_dataframe,
)
from helpers.suggestions import Suggestion, build_suggestions


def _has_data(df: pd.DataFrame) -> bool:
    return df is not None and not df.empty


def _display_missing_artifact(name: str) -> None:
    st.warning(f"Artifact is missing or unavailable: `{name}`")


def _format_pct_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    out = df.copy()
    if column in out.columns:
        out[column] = out[column].map(lambda x: pct(x))
    return out


def _format_number_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].map(number)
    return out


def _split_metric_columns(split_counts: dict[str, Any], split_percentages: dict[str, Any]) -> None:
    cols = st.columns(3)
    for idx, split in enumerate(SPLIT_ORDER):
        count = split_counts.get(split, 0)
        percentage = split_percentages.get(split, 0.0)
        cols[idx].metric(
            label=split.title(),
            value=integer(count),
            delta=pct(percentage),
            delta_color="off",
        )


def render_bundle_status(bundle: Any) -> None:
    """
    Render a compact status message for loaded/missing artifacts.
    """
    st.caption(f"Visualization prefix: `{bundle.visualization_prefix}`")

    if bundle.is_complete:
        st.success("All visualization artifacts loaded.")
        return

    failed = bundle.missing_or_failed
    st.warning(f"{len(failed)} visualization artifact(s) could not be loaded.")

    with st.expander("Show missing/failed artifacts"):
        for name, artifact in failed.items():
            st.markdown(f"**{name}**")
            st.caption(artifact.s3_ref.uri)
            st.code(artifact.error or "Unknown error")


def render_overview(bundle: Any) -> None:
    overview = bundle.get("overview")
    if not overview:
        _display_missing_artifact("overview")
        return

    st.subheader("Dataset Overview")

    dataset_id = overview.get("dataset_id", bundle.dataset_id)
    version = overview.get("version", bundle.version)
    label_type = overview.get("label_type", "unknown")
    row_count = overview.get("row_count", 0)
    honor_source_splits = overview.get("honor_source_splits")
    effective_split_mode = overview.get("effective_split_mode")

    cols = st.columns(4)
    cols[0].metric("Dataset", str(dataset_id))
    cols[1].metric("Version", f"v{version}")
    cols[2].metric("Label Type", str(label_type))
    cols[3].metric("Rows", integer(row_count))

    cols = st.columns(2)
    cols[0].metric("Honor Source Splits", str(honor_source_splits))
    cols[1].metric("Effective Split Mode", str(effective_split_mode or "—"))

    split_counts = overview.get("split_counts", {})
    split_percentages = overview.get("split_percentages", {})

    if not isinstance(split_counts, dict):
        split_counts = {}
    if not isinstance(split_percentages, dict):
        split_percentages = {}

    st.markdown("#### Split Counts")
    _split_metric_columns(split_counts, split_percentages)

    df = pd.DataFrame(
        [
            {
                "split": split,
                "count": split_counts.get(split, 0),
                "percentage": split_percentages.get(split, 0.0),
            }
            for split in SPLIT_ORDER
        ]
    )

    if _has_data(df):
        fig = px.bar(
            df,
            x="split",
            y="count",
            text="count",
            category_orders={"split": SPLIT_ORDER},
            title="Rows by Split",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis_title="Rows", xaxis_title="Split")
        st.plotly_chart(fig, use_container_width=True)

        display_df = df.copy()
        display_df["percentage"] = display_df["percentage"].map(pct)
        st.dataframe(display_df, use_container_width=True, hide_index=True)


def _render_distribution(
    *,
    artifact: dict[str, Any],
    title: str,
    category_name: str,
    missing_artifact_name: str,
    max_categories_default: int = 40,
) -> None:
    if not artifact:
        _display_missing_artifact(missing_artifact_name)
        return

    counts_by_split = artifact.get("counts_by_split", {})
    percentages_by_split = artifact.get("percentages_by_split", {})

    if not isinstance(counts_by_split, dict):
        counts_by_split = {}
    if not isinstance(percentages_by_split, dict):
        percentages_by_split = {}

    counts_df = counts_by_split_to_dataframe(
        counts_by_split,
        category_name=category_name,
        value_name="count",
    )

    pct_df = percentages_by_split_to_dataframe(
        percentages_by_split,
        category_name=category_name,
        value_name="percentage",
    )

    if counts_df.empty:
        st.info(f"No {category_name} distribution data available.")
        return

    st.subheader(title)

    total_by_category = (
        counts_df.groupby(category_name, observed=False)["count"]
        .sum()
        .sort_values(ascending=False)
    )
    categories = list(total_by_category.index)

    category_count = len(categories)

    if category_count <= 5:
        max_categories = category_count
    else:
        slider_max = min(100, category_count)
        slider_default = min(max_categories_default, slider_max)

        # Keep the default at least 5, but never above slider_max.
        slider_default = max(5, slider_default)

        max_categories = st.slider(
            f"Maximum {category_name}s to display",
            min_value=5,
            max_value=slider_max,
            value=slider_default,
            step=5,
            key=f"{missing_artifact_name}_max_categories",
        )

    selected_categories = categories[:max_categories]

    counts_df = counts_df[counts_df[category_name].isin(selected_categories)]
    pct_df = pct_df[pct_df[category_name].isin(selected_categories)]

    chart_mode = st.radio(
        "Chart mode",
        ["Counts", "Percentages"],
        horizontal=True,
        key=f"{missing_artifact_name}_chart_mode",
    )

    if chart_mode == "Counts":
        fig = px.bar(
            counts_df,
            x=category_name,
            y="count",
            color="split",
            barmode="group",
            category_orders={"split": SPLIT_ORDER},
            title=f"{title}: Counts",
        )
        fig.update_layout(
            xaxis_title=category_name.replace("_", " ").title(),
            yaxis_title="Count",
            xaxis_tickangle=-35,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            counts_df.sort_values(["split", "count"], ascending=[True, False]),
            use_container_width=True,
            hide_index=True,
        )

    else:
        fig = px.bar(
            pct_df,
            x=category_name,
            y="percentage",
            color="split",
            barmode="group",
            category_orders={"split": SPLIT_ORDER},
            title=f"{title}: Percentages",
        )
        fig.update_layout(
            xaxis_title=category_name.replace("_", " ").title(),
            yaxis_title="Percentage",
            xaxis_tickangle=-35,
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        display_df = _format_pct_column(
            pct_df.sort_values(["split", "percentage"], ascending=[True, False]),
            "percentage",
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_class_balance(bundle: Any) -> None:
    artifact = bundle.get("class_distribution")
    _render_distribution(
        artifact=artifact,
        title="Class Balance by Split",
        category_name="class",
        missing_artifact_name="class_distribution",
    )


def render_source_balance(bundle: Any) -> None:
    artifact = bundle.get("source_distribution")
    _render_distribution(
        artifact=artifact,
        title="Source Balance by Split",
        category_name="source",
        missing_artifact_name="source_distribution",
    )


def render_source_split_resolution(bundle: Any) -> None:
    artifact = bundle.get("source_split_resolution")
    if not artifact:
        _display_missing_artifact("source_split_resolution")
        return

    st.subheader("Source Split Resolution")

    status_counts = artifact.get("source_split_status_counts_by_split", {})
    resolved_counts = artifact.get("resolved_source_split_counts_by_split", {})

    if not isinstance(status_counts, dict):
        status_counts = {}
    if not isinstance(resolved_counts, dict):
        resolved_counts = {}

    status_df = counts_by_split_to_dataframe(
        status_counts,
        category_name="status",
        value_name="count",
    )

    resolved_df = counts_by_split_to_dataframe(
        resolved_counts,
        category_name="resolved_source_split",
        value_name="count",
    )

    if status_df.empty and resolved_df.empty:
        st.info("No source split resolution data available.")
        return

    if not status_df.empty:
        st.markdown("#### Source Split Status")
        fig = px.bar(
            status_df,
            x="split",
            y="count",
            color="status",
            barmode="group",
            category_orders={"split": SPLIT_ORDER},
            title="Source Split Status by Final Dataset Split",
        )
        fig.update_layout(xaxis_title="Final Dataset Split", yaxis_title="Rows")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(status_df, use_container_width=True, hide_index=True)

    if not resolved_df.empty:
        st.markdown("#### Resolved Source Split")
        fig = px.bar(
            resolved_df,
            x="split",
            y="count",
            color="resolved_source_split",
            barmode="group",
            category_orders={
                "split": SPLIT_ORDER,
                "resolved_source_split": SPLIT_ORDER,
            },
            title="Original/Resolved Source Split by Final Dataset Split",
        )
        fig.update_layout(
            xaxis_title="Final Dataset Split",
            yaxis_title="Rows",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(resolved_df, use_container_width=True, hide_index=True)


def render_quality_buckets(bundle: Any) -> None:
    artifact = bundle.get("quality_distribution")
    if not artifact:
        _display_missing_artifact("quality_distribution")
        return

    st.subheader("Quality Buckets")

    bucket_fields = list(artifact.keys())
    if not bucket_fields:
        st.info("No quality bucket data available.")
        return

    selected_field = st.selectbox(
        "Quality feature",
        options=bucket_fields,
        format_func=bucket_label,
    )

    block = artifact.get(selected_field, {})
    if not isinstance(block, dict):
        st.info("Selected quality bucket has no data.")
        return

    counts_by_split = block.get("counts_by_split", {})
    if not isinstance(counts_by_split, dict):
        counts_by_split = {}

    df = counts_by_split_to_dataframe(
        counts_by_split,
        category_name="bucket",
        value_name="count",
    )

    if df.empty:
        st.info(f"No data available for {bucket_label(selected_field)}.")
        return

    fig = px.bar(
        df,
        x="bucket",
        y="count",
        color="split",
        barmode="group",
        category_orders={"split": SPLIT_ORDER},
        title=f"{bucket_label(selected_field)} Buckets by Split",
    )
    fig.update_layout(
        xaxis_title="Bucket",
        yaxis_title="Rows",
        xaxis_tickangle=-25,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Local percentage view, since the Lambda currently stores counts only here.
    totals = df.groupby("split", observed=False)["count"].transform("sum")
    pct_df = df.copy()
    pct_df["percentage"] = pct_df["count"] / totals.where(totals != 0, 1)

    fig_pct = px.bar(
        pct_df,
        x="bucket",
        y="percentage",
        color="split",
        barmode="group",
        category_orders={"split": SPLIT_ORDER},
        title=f"{bucket_label(selected_field)} Bucket Percentages by Split",
    )
    fig_pct.update_yaxes(tickformat=".0%")
    fig_pct.update_layout(
        xaxis_title="Bucket",
        yaxis_title="Percentage",
        xaxis_tickangle=-25,
    )
    st.plotly_chart(fig_pct, use_container_width=True)

    display_df = pct_df.copy()
    display_df["percentage"] = display_df["percentage"].map(pct)
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_numeric_features(bundle: Any) -> None:
    summary = bundle.get("numeric_summary")
    histograms = bundle.get("numeric_histograms")

    if not summary and not histograms:
        _display_missing_artifact("numeric_summary / numeric_histograms")
        return

    st.subheader("Numeric Features")

    overall_summary = summary.get("overall", {}) if isinstance(summary, dict) else {}
    by_split_summary = summary.get("by_split", {}) if isinstance(summary, dict) else {}
    overall_histograms = histograms.get("overall", {}) if isinstance(histograms, dict) else {}
    by_split_histograms = histograms.get("by_split", {}) if isinstance(histograms, dict) else {}

    feature_names = sorted(
        set(overall_summary.keys())
        | set(overall_histograms.keys())
    )

    if not feature_names:
        st.info("No numeric feature data available.")
        return

    selected_feature = st.selectbox(
        "Numeric feature",
        options=feature_names,
        format_func=feature_label,
    )

    st.markdown(f"#### {feature_label(selected_feature)}")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("##### Summary")

        rows: list[dict[str, Any]] = []

        overall_stats = overall_summary.get(selected_feature, {})
        if isinstance(overall_stats, dict):
            row = {"split": "overall"}
            row.update(overall_stats)
            rows.append(row)

        if isinstance(by_split_summary, dict):
            for split in SPLIT_ORDER:
                split_stats = by_split_summary.get(split, {}).get(selected_feature, {})
                if isinstance(split_stats, dict):
                    row = {"split": split}
                    row.update(split_stats)
                    rows.append(row)

        summary_df = pd.DataFrame(rows)
        if summary_df.empty:
            st.info("No summary statistics available for this feature.")
        else:
            display_df = _format_number_columns(
                summary_df,
                ["count", "min", "max", "mean", "p10", "p25", "p50", "p75", "p90"],
            )
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("##### Histogram")

        mode = st.radio(
            "Histogram view",
            ["By split", "Overall"],
            horizontal=True,
            key=f"hist_mode_{selected_feature}",
        )

        if mode == "Overall":
            hist = overall_histograms.get(selected_feature, {})
            hist_df = histogram_to_dataframe(hist if isinstance(hist, dict) else {})

            if hist_df.empty:
                st.info("No histogram available for this feature.")
            else:
                fig = px.bar(
                    hist_df,
                    x="bin_mid",
                    y="count",
                    title=f"{feature_label(selected_feature)} Histogram",
                )
                fig.update_layout(
                    xaxis_title=feature_label(selected_feature),
                    yaxis_title="Rows",
                )
                st.plotly_chart(fig, use_container_width=True)

        else:
            frames: list[pd.DataFrame] = []

            if isinstance(by_split_histograms, dict):
                for split in SPLIT_ORDER:
                    split_hist = by_split_histograms.get(split, {}).get(selected_feature, {})
                    if not isinstance(split_hist, dict):
                        continue

                    split_df = histogram_to_dataframe(split_hist)
                    if split_df.empty:
                        continue

                    split_df["split"] = split
                    frames.append(split_df)

            if not frames:
                st.info("No split histograms available for this feature.")
            else:
                hist_df = pd.concat(frames, ignore_index=True)
                fig = px.bar(
                    hist_df,
                    x="bin_mid",
                    y="count",
                    color="split",
                    barmode="overlay",
                    category_orders={"split": SPLIT_ORDER},
                    title=f"{feature_label(selected_feature)} Histogram by Split",
                    opacity=0.65,
                )
                fig.update_layout(
                    xaxis_title=feature_label(selected_feature),
                    yaxis_title="Rows",
                )
                st.plotly_chart(fig, use_container_width=True)


def _render_delta_block(
    *,
    block: dict[str, Any],
    title: str,
    category_name: str,
) -> None:
    if not isinstance(block, dict):
        st.info(f"No data available for {title}.")
        return

    max_delta = block.get("max_absolute_delta", 0.0)
    mean_delta = block.get("mean_absolute_delta", 0.0)

    cols = st.columns(2)
    cols[0].metric("Max Absolute Delta", pct(max_delta))
    cols[1].metric("Mean Absolute Delta", pct(mean_delta))

    df = nested_delta_block_to_dataframe(block)

    if df.empty:
        st.info(f"No delta rows available for {title}.")
        return

    row_count = len(df)

    if row_count <= 5:
        top_n = row_count
    else:
        slider_max = min(100, row_count)
        slider_default = min(25, slider_max)
        slider_default = max(5, slider_default)

        top_n = st.slider(
            f"Rows to show for {title}",
            min_value=5,
            max_value=slider_max,
            value=slider_default,
            step=5,
            key=f"delta_rows_{title}",
        )

    chart_df = df.head(top_n)

    fig = px.bar(
        chart_df,
        x="category",
        y="absolute_delta",
        title=title,
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(
        xaxis_title=category_name.title(),
        yaxis_title="Absolute Percentage Delta",
        xaxis_tickangle=-35,
    )
    st.plotly_chart(fig, use_container_width=True)

    display_df = chart_df.copy()
    for col in ["train_percentage", "compare_percentage", "absolute_delta"]:
        display_df[col] = display_df[col].map(pct)

    display_df = display_df.rename(columns={"category": category_name})
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_split_comparison(bundle: Any) -> None:
    artifact = bundle.get("split_comparison")
    if not artifact:
        _display_missing_artifact("split_comparison")
        return

    st.subheader("Split Comparison Metrics")

    class_comparison = artifact.get("class_comparison", {})
    source_comparison = artifact.get("source_comparison", {})

    comparison_type = st.radio(
        "Comparison type",
        ["Class", "Source"],
        horizontal=True,
    )

    split_pair = st.radio(
        "Compare split against train",
        ["val_vs_train", "test_vs_train"],
        horizontal=True,
    )

    if comparison_type == "Class":
        block = class_comparison.get(split_pair, {}) if isinstance(class_comparison, dict) else {}
        _render_delta_block(
            block=block,
            title=f"Class Distribution: {split_pair.replace('_', ' ').title()}",
            category_name="class",
        )
    else:
        block = source_comparison.get(split_pair, {}) if isinstance(source_comparison, dict) else {}
        _render_delta_block(
            block=block,
            title=f"Source Distribution: {split_pair.replace('_', ' ').title()}",
            category_name="source",
        )


def _render_single_suggestion(suggestion: Suggestion) -> None:
    severity = suggestion.severity.lower().strip()

    label = f"**{suggestion.title}**  \n_{suggestion.category}_  \n{suggestion.detail}"

    if severity == "critical":
        st.error(label)
    elif severity == "warning":
        st.warning(label)
    elif severity == "success":
        st.success(label)
    else:
        st.info(label)


def render_suggestions(bundle: Any) -> None:
    st.subheader("Dataset Suggestions")

    suggestions = build_suggestions(bundle)

    if not suggestions:
        st.info("No suggestions generated.")
        return

    severity_filter = st.multiselect(
        "Severity filter",
        options=["critical", "warning", "info", "success"],
        default=["critical", "warning", "info", "success"],
    )

    category_options = sorted({s.category for s in suggestions})
    category_filter = st.multiselect(
        "Category filter",
        options=category_options,
        default=category_options,
    )

    filtered = [
        s for s in suggestions
        if s.severity in severity_filter and s.category in category_filter
    ]

    if not filtered:
        st.info("No suggestions match the current filters.")
        return

    counts = {
        "critical": sum(1 for s in suggestions if s.severity == "critical"),
        "warning": sum(1 for s in suggestions if s.severity == "warning"),
        "info": sum(1 for s in suggestions if s.severity == "info"),
        "success": sum(1 for s in suggestions if s.severity == "success"),
    }

    cols = st.columns(4)
    cols[0].metric("Critical", counts["critical"])
    cols[1].metric("Warnings", counts["warning"])
    cols[2].metric("Info", counts["info"])
    cols[3].metric("Success", counts["success"])

    st.markdown("---")

    for suggestion in filtered:
        _render_single_suggestion(suggestion)


def render_raw_artifacts(bundle: Any) -> None:
    """
    Optional debugging tab: inspect raw loaded JSON artifacts.
    """
    st.subheader("Raw Artifacts")

    artifact_names = sorted(bundle.artifacts.keys())
    selected = st.selectbox("Artifact", options=artifact_names)

    artifact = bundle.artifacts[selected]

    st.caption(artifact.s3_ref.uri)

    if artifact.loaded:
        st.json(artifact.payload)
    else:
        st.error(artifact.error or "Artifact failed to load.")