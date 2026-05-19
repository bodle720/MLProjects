from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from helpers import sweep_settings as settings


def build_mlflow_client() -> MlflowClient:
    return MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)


def get_experiment_or_raise(client: MlflowClient, experiment_name: str):
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(
            f"Could not find MLflow experiment named '{experiment_name}' "
            f"at tracking URI '{settings.MLFLOW_TRACKING_URI}'."
        )

    return experiment


def list_experiment_runs(
    client: MlflowClient,
    experiment_id: str,
    max_results_per_page: int = 1000,
):
    all_runs = []
    page_token = None

    while True:
        runs_page = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string="",
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=max_results_per_page,
            order_by=["attributes.start_time DESC"],
            page_token=page_token,
        )

        all_runs.extend(runs_page)

        page_token = getattr(runs_page, "token", None)
        if not page_token:
            break

    return all_runs


def get_run_name(run) -> str:
    return (
        run.data.tags.get("mlflow.runName")
        or run.data.tags.get("run_name")
        or run.info.run_name
        or run.info.run_id
    )


def is_finished_run(run) -> bool:
    return run.info.status in settings.VALID_RUN_STATUSES


def summarize_run(run) -> dict:
    return {
        "run_id": run.info.run_id,
        "run_name": get_run_name(run),
        "experiment_id": run.info.experiment_id,
        "status": run.info.status,
        "artifact_uri": run.info.artifact_uri,
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
        "params": dict(run.data.params),
        "metrics": dict(run.data.metrics),
        "tags": dict(run.data.tags),
    }


def list_finished_run_summaries(client: MlflowClient, experiment_name: str) -> list[dict]:
    experiment = get_experiment_or_raise(client, experiment_name)
    runs = list_experiment_runs(client, experiment.experiment_id)

    summaries = []
    for run in runs:
        if not is_finished_run(run):
            continue
        summaries.append(summarize_run(run))

    return summaries


def discover_finished_runs(experiment_name: str) -> list[dict]:
    client = build_mlflow_client()
    return list_finished_run_summaries(client, experiment_name)