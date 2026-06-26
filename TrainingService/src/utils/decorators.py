from __future__ import annotations

import time
from functools import wraps
from typing import Any, Callable

import pandas as pd

from utils.logging_utils import get_logger

logger = get_logger(__name__)


def pipeline_step(step_name: str):
    def decorator(func: Callable[..., Any]):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info("START | %s", step_name)

            start = time.time()

            try:
                result = func(*args, **kwargs)

                elapsed = time.time() - start

                logger.info(
                    "END | %s | elapsed=%.2f sec",
                    step_name,
                    elapsed,
                )

                return result

            except Exception:
                logger.exception(
                    "FAILED | %s",
                    step_name,
                )
                raise

        return wrapper

    return decorator


def log_dataframe_shape(step_name: str):
    def decorator(func: Callable[..., Any]):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            if isinstance(result, pd.DataFrame):
                logger.info(
                    "%s shape | rows=%s cols=%s",
                    step_name,
                    result.shape[0],
                    result.shape[1],
                )

            elif (
                isinstance(result, tuple)
                and result
                and isinstance(result[0], pd.DataFrame)
            ):
                df = result[0]

                logger.info(
                    "%s shape | rows=%s cols=%s",
                    step_name,
                    df.shape[0],
                    df.shape[1],
                )

            return result

        return wrapper

    return decorator