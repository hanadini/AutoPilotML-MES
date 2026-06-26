from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request

from api.monitoring_controller import router as monitoring_router
from config.settings import API_VERSION, SERVICE_NAME
from db.database import Base, engine
from db.models import MESPredictionLog, OperatorFeedback
from utils.logging_utils import get_logger


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("%s startup initiated", SERVICE_NAME)

    logger.info("Creating/checking MonitoringService database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("MonitoringService database tables are ready.")

    yield

    logger.info("%s shutdown completed", SERVICE_NAME)


app = FastAPI(
    title=SERVICE_NAME,
    description=(
        "Standalone Monitoring Service for MDF1 AI platform. "
        "Aggregates MES prediction logs and operator feedback data "
        "to provide production intelligence, risk distribution, "
        "and recommendation acceptance metrics."
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

    response = await call_next(request)

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
        "architecture": "standalone MonitoringService",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": SERVICE_NAME,
    }


app.include_router(monitoring_router)