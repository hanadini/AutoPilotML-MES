from __future__ import annotations

from typing import Any, Dict

from config.settings import RISK_RULES
from utils.decorators import service_step
from utils.logging_utils import get_logger


logger = get_logger(__name__)

@service_step
def calculate_risk(
    target: str,
    value: float,
) -> str:
    rule = RISK_RULES.get(target)

    if not rule:
        logger.warning(
            "No risk rule defined for target '%s'. Returning LOW.",
            target,
        )

        return "LOW"

    direction = rule["direction"]
    medium = float(rule["medium"])
    high = float(rule["high"])

    if direction == "high_is_risky":
        if value >= high:
            risk = "HIGH"

        elif value >= medium:
            risk = "MEDIUM"

        else:
            risk = "LOW"

    elif direction == "low_is_risky":
        if value <= high:
            risk = "HIGH"

        elif value <= medium:
            risk = "MEDIUM"

        else:
            risk = "LOW"

    else:
        logger.warning(
            "Unsupported risk direction '%s' for target '%s'.",
            direction,
            target,
        )

        risk = "LOW"

    logger.info(
        "Calculated risk | target=%s | value=%s | risk=%s",
        target,
        value,
        risk,
    )

    return risk