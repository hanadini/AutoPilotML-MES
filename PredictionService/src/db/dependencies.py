from __future__ import annotations

from collections.abc import Generator

from sqlalchemy.orm import Session

from db.database import SessionLocal
from utils.decorators import service_step
from utils.logging_utils import get_logger


logger = get_logger(__name__)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()

    try:
        yield db

    except Exception:
        logger.exception(
            "Database session error."
        )
        raise

    finally:
        db.close()

        logger.debug(
            "Database session closed."
        )