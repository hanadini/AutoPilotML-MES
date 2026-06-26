from __future__ import annotations

import shutil
from pathlib import Path

from config.settings import (
    ARTIFACTS_DIR,
    BEST_MODELS_REGISTRY_PATH,
    KEEP_BEST_MODELS_REGISTRY,
)
from utils.decorators import pipeline_step
from utils.logging_utils import get_logger


logger = get_logger(__name__)


@pipeline_step("Clean artifacts directory")
def clean_artifacts_dir(
    artifacts_root: str | Path = ARTIFACTS_DIR,
    keep_registry: bool = KEEP_BEST_MODELS_REGISTRY,
) -> None:
    """
    Clean artifact directory before training.
    """
    artifacts_path = Path(artifacts_root)

    if not artifacts_path.exists():
        logger.info(
            "Artifacts directory does not exist: %s",
            artifacts_path,
        )
        return

    logger.info(
        "Cleaning artifacts directory: %s",
        artifacts_path,
    )

    registry_filename = BEST_MODELS_REGISTRY_PATH.name

    for item in artifacts_path.iterdir():

        if (
            keep_registry
            and item.name == registry_filename
        ):
            logger.info(
                "Keeping registry file: %s",
                item.name,
            )
            continue

        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

            logger.info(
                "Removed artifact item: %s",
                item,
            )

        except Exception as exc:
            logger.warning(
                "Failed to remove artifact item: %s | error=%s",
                item,
                exc,
            )

    logger.info("Artifacts cleanup completed.")