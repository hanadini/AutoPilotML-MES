from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from mes.mes_schema import MESPredictionRequest, MESPredictionResponse
from api import router
from mes.mes_controller import router as mes_router
from config.settings import (
    API_VERSION,
    PREDICTION_ARTIFACTS_DIR,
    SERVICE_NAME,
    TRAINING_ARTIFACTS_DIR,
)
from core.model_registery import ModelRegistry
from db.database import Base, engine
from utils.artifact_sync import sync_best_artifacts
from utils.logging_utils import get_logger


logger = get_logger(__name__)

model_registry = ModelRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("%s startup initiated", SERVICE_NAME)

    logger.info("Creating/checking database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables are ready.")

    logger.info("Syncing artifacts from TrainingService...")

    sync_best_artifacts(
        training_artifacts_dir=TRAINING_ARTIFACTS_DIR,
        prediction_artifacts_dir=PREDICTION_ARTIFACTS_DIR,
    )

    logger.info("Artifact sync completed successfully.")

    logger.info("Preloading registry-driven model artifacts...")

    load_status = model_registry.load_all_from_registry()

    logger.info(
        "Model preload completed | status=%s",
        load_status,
    )

    yield

    model_registry.clear()

    logger.info("%s shutdown completed", SERVICE_NAME)


app = FastAPI(
    title=SERVICE_NAME,
    description=(
        "Registry-driven ML prediction service for MDF1 production data. "
        "Aligned with TrainingService artifact bundles, feature contracts, "
        "weighted ensembles, and explainability outputs."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(
    request: Request,
    call_next,
):
    start_time = time.perf_counter()

    logger.info(
        "HTTP request started | method=%s | path=%s",
        request.method,
        request.url.path,
    )

    try:
        response = await call_next(request)

    except Exception:
        logger.exception(
            "Unhandled error in request | method=%s | path=%s",
            request.method,
            request.url.path,
        )
        raise

    duration_ms = round(
        (time.perf_counter() - start_time) * 1000,
        2,
    )

    logger.info(
        "HTTP request completed | method=%s | path=%s | status_code=%s | duration_ms=%s",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )

    return response


@app.get("/")
def root():
    return {
        "service": SERVICE_NAME,
        "version": API_VERSION,
        "status": "running",
        "architecture": "registry-driven PredictionService",
    }


app.include_router(router)
# app.include_router(mes_router)
