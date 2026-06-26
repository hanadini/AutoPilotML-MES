from __future__ import annotations

import time
from typing import Optional

from utils.logging_utils import get_logger


logger = get_logger(__name__)


class TimeBudget:
    _quiet_mode = False

    def __init__(self, name: str) -> None:
        self.name = name
        self._elapsed_ms: Optional[float] = None
        self._start_time: Optional[float] = None

    def __enter__(self) -> "TimeBudget":
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._start_time is None:
            return

        self._elapsed_ms = (
            time.perf_counter() - self._start_time
        ) * 1000

        if not self._quiet_mode:
            logger.info(
                "%s took %s",
                self.name,
                self.elapsed_time,
            )

    @property
    def elapsed_time(self) -> str:
        if self._elapsed_ms is None:
            return "0.000ms"

        return self.ms_format(self._elapsed_ms)

    @staticmethod
    def ms_format(milliseconds: float) -> str:
        if milliseconds < 1:
            return f"{milliseconds:.3f}ms"

        if milliseconds < 1000:
            return f"{milliseconds:.2f}ms"

        return f"{milliseconds / 1000:.3f}sec"

    @classmethod
    def set_quiet(cls, quiet: bool = True) -> None:
        cls._quiet_mode = quiet