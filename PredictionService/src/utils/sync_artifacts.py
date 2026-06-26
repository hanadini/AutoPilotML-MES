from __future__ import annotations

from config.settings import (
    TRAINING_ARTIFACTS_DIR,
    PREDICTION_ARTIFACTS_DIR,
)
from utils.artifact_sync import sync_best_artifacts
from utils.logging_utils import get_logger


logger = get_logger(__name__)


def main() -> None:
    logger.info("Starting artifact synchronization...")

    sync_best_artifacts(
        training_artifacts_dir=TRAINING_ARTIFACTS_DIR,
        prediction_artifacts_dir=PREDICTION_ARTIFACTS_DIR,
    )

    logger.info("Artifact synchronization completed successfully.")


if __name__ == "__main__":
    main()