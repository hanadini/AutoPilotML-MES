from __future__ import annotations

from fastapi import APIRouter, HTTPException

from config.settings import ARTIFACTS_DIR
from core.shared_model_registry import shared_model_registry
from utils.decorators import service_step
from utils.logging_utils import get_logger


logger = get_logger(__name__)

router = APIRouter()

registry = shared_model_registry


@router.get("/models")
@service_step
def list_models() -> dict:
    try:
        if not ARTIFACTS_DIR.exists():
            raise FileNotFoundError(
                f"Artifacts directory not found: {ARTIFACTS_DIR}"
            )

        registry.load_registry()

        artifact_directories = sorted(
            [
                path.name
                for path in ARTIFACTS_DIR.iterdir()
                if path.is_dir()
            ]
        )

        response = {
            "available_artifact_directories": artifact_directories,
            "artifact_count": len(artifact_directories),
            "registered_targets": registry.target_names(),
            "registry_summary": registry.summary(),
        }

        logger.info(
            "Returned models endpoint response with %s artifact(s).",
            len(artifact_directories),
        )

        return response

    except Exception as exc:
        logger.exception(
            "Failed to load models endpoint: %s",
            exc,
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )