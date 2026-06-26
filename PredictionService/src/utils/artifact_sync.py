from __future__ import annotations

import json
import shutil
from pathlib import Path

from utils.logging_utils import get_logger


logger = get_logger(__name__)


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_directory(directory: Path) -> None:
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
        return

    for item in directory.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def sync_best_artifacts(
    training_artifacts_dir: str | Path,
    prediction_artifacts_dir: str | Path,
) -> None:
    training_dir = Path(training_artifacts_dir)
    prediction_dir = Path(prediction_artifacts_dir)

    if not training_dir.exists():
        raise FileNotFoundError(f"Training artifacts directory not found: {training_dir}")

    registry_path = training_dir / "best_models_registry.json"
    if not registry_path.exists():
        raise FileNotFoundError(f"Training registry not found: {registry_path}")

    registry = _read_json(registry_path)

    clean_directory(prediction_dir)
    shutil.copy2(registry_path, prediction_dir / "best_models_registry.json")

    required_artifacts: set[str] = set()

    for target_name, info in registry.items():
        serving_type = info.get("type", "single_model")

        if serving_type == "single_model":
            required_artifacts.add(info["artifact_model_name"])

        elif serving_type == "weighted_ensemble":
            members = info.get("members", {})
            for _, artifact_name in members.items():
                required_artifacts.add(artifact_name)

        else:
            raise ValueError(
                f"Unsupported serving type '{serving_type}' in registry for target '{target_name}'"
            )

    for artifact_name in required_artifacts:
        src = training_dir / artifact_name
        dst = prediction_dir / artifact_name

        if not src.exists():
            raise FileNotFoundError(
                f"Required artifact folder not found in training artifacts: {src}"
            )

        shutil.copytree(src, dst)

    logger.info(
        "Synced %s artifact folder(s) to %s",
        len(required_artifacts),
        prediction_dir,
    )