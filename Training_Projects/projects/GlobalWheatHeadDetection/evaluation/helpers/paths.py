from datetime import datetime
from pathlib import Path

from evaluation.config import PROJECT_ROOT


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)

    if candidate.is_absolute():
        return candidate

    return PROJECT_ROOT / candidate


def create_run_dir(output_root: str | Path, split: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"eval_{split}_split_{timestamp}"
    run_dir = resolve_project_path(output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir