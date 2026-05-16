import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunContext:
    run_root_dir: Path
    experiment_name: str
    experiment_dir_name: str
    experiment_dir: Path
    base_run_name: str
    resolved_run_name: str
    run_dir: Path
    reserved: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_root_dir": str(self.run_root_dir),
            "experiment_name": self.experiment_name,
            "experiment_dir_name": self.experiment_dir_name,
            "experiment_dir": str(self.experiment_dir),
            "base_run_name": self.base_run_name,
            "resolved_run_name": self.resolved_run_name,
            "run_dir": str(self.run_dir),
            "reserved": self.reserved,
        }

def make_filesystem_safe_name(value: str, fallback: str) -> str:
    text = str(value).strip()

    if not text:
        text = fallback

    text = re.sub(r"[^\w.\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip("._-")

    if not text:
        text = fallback

    return text[:160]

def get_experiment_name(config: dict[str, Any]) -> str:
    run_cfg = config.get("run", {})
    mlflow_cfg = config.get("mlflow", {})
    project_cfg = config.get("project", {})

    return str(
        run_cfg.get("experiment_name")
        or mlflow_cfg.get("experiment_name")
        or project_cfg.get("name")
        or "default_experiment"
    )

def get_base_run_name(config: dict[str, Any]) -> str:
    run_cfg = config.get("run", {})
    training_cfg = config.get("training", {})

    return str(
        run_cfg.get("run_name")
        or training_cfg.get("run_name")
        or "yolo_run"
    )

def candidate_run_names(base_run_name: str):
    yield base_run_name

    index = 1
    while True:
        yield f"{base_run_name}_{index}"
        index += 1

def prepare_run_context(config: dict[str, Any], reserve: bool) -> RunContext:
    paths_cfg = config["paths"]

    run_root_dir = Path(paths_cfg["run_root_dir_resolved"])
    experiment_name = get_experiment_name(config)
    base_run_name_raw = get_base_run_name(config)

    experiment_dir_name = make_filesystem_safe_name(
        experiment_name,
        fallback="default_experiment",
    )
    base_run_name = make_filesystem_safe_name(
        base_run_name_raw,
        fallback="yolo_run",
    )

    experiment_dir = run_root_dir / experiment_dir_name

    if reserve:
        experiment_dir.mkdir(parents=True, exist_ok=True)

    for resolved_run_name in candidate_run_names(base_run_name):
        run_dir = experiment_dir / resolved_run_name

        if reserve:
            try:
                run_dir.mkdir(parents=True, exist_ok=False)
                return RunContext(
                    run_root_dir=run_root_dir,
                    experiment_name=experiment_name,
                    experiment_dir_name=experiment_dir_name,
                    experiment_dir=experiment_dir,
                    base_run_name=base_run_name,
                    resolved_run_name=resolved_run_name,
                    run_dir=run_dir,
                    reserved=True,
                )
            except FileExistsError:
                continue

        if not run_dir.exists():
            return RunContext(
                run_root_dir=run_root_dir,
                experiment_name=experiment_name,
                experiment_dir_name=experiment_dir_name,
                experiment_dir=experiment_dir,
                base_run_name=base_run_name,
                resolved_run_name=resolved_run_name,
                run_dir=run_dir,
                reserved=False,
            )

    raise RuntimeError("Unable to allocate a unique run directory.")