from __future__ import annotations

import functools
import time
from typing import Any, Callable, TypeVar, cast

from utils.logging_utils import get_logger


F = TypeVar("F", bound=Callable[..., Any])

logger = get_logger(__name__)


def log_execution(func: F) -> F:
    """
    Log when a service/core function starts and finishes.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger.info("Starting: %s", func.__name__)

        result = func(*args, **kwargs)

        logger.info("Finished: %s", func.__name__)
        return result

    return cast(F, wrapper)


def measure_latency(func: F) -> F:
    """
    Measure execution time in milliseconds.
    Useful for prediction, registry loading, DB logging, and explainability.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()

        try:
            return func(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "%s latency: %.2f ms",
                func.__name__,
                elapsed_ms,
            )

    return cast(F, wrapper)


def handle_service_errors(func: F) -> F:
    """
    Log service errors consistently and re-raise them.

    Important:
    This decorator does not hide errors.
    It logs them and allows FastAPI or the caller to handle the response.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)

        except FileNotFoundError:
            logger.exception("Missing required file in: %s", func.__name__)
            raise

        except KeyError:
            logger.exception("Missing required key in: %s", func.__name__)
            raise

        except ValueError:
            logger.exception("Invalid value in: %s", func.__name__)
            raise

        except Exception:
            logger.exception("Unexpected error in: %s", func.__name__)
            raise

    return cast(F, wrapper)


def service_step(func: F) -> F:
    """
    Combined decorator for main PredictionService operations.

    It applies:
    - error logging
    - start/finish logging
    - latency measurement
    """

    return handle_service_errors(
        measure_latency(
            log_execution(func)
        )
    )