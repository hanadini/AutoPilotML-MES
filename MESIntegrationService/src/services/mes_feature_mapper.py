from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from config.settings import MES_FEATURE_MAPPING_FILE_PATH
from utils.logging_utils import get_logger


logger = get_logger(__name__)


@lru_cache(maxsize=1)
def load_mes_feature_mapping() -> dict[str, str]:
    if not MES_FEATURE_MAPPING_FILE_PATH.exists():
        raise FileNotFoundError(
            f"MES feature mapping file not found: "
            f"{MES_FEATURE_MAPPING_FILE_PATH}"
        )

    with MES_FEATURE_MAPPING_FILE_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        mapping = json.load(file)

    logger.info(
        "MES feature mapping loaded | path=%s | feature_count=%s",
        MES_FEATURE_MAPPING_FILE_PATH,
        len(mapping),
    )

    return mapping


def map_mes_features_to_ml_features(
    mes_features: dict[str, Any],
    *,
    strict: bool = False,
) -> dict[str, Any]:
    feature_mapping = load_mes_feature_mapping()

    mapped_features: dict[str, Any] = {}
    unmapped_features: list[str] = []

    for mes_name, value in mes_features.items():
        ml_name = feature_mapping.get(mes_name)

        if ml_name:
            mapped_features[ml_name] = value
        else:
            unmapped_features.append(mes_name)

    if unmapped_features:
        logger.warning(
            "Unmapped MES features ignored | count=%s | features=%s",
            len(unmapped_features),
            unmapped_features,
        )

        if strict:
            raise ValueError(
                f"Unmapped MES features found: {unmapped_features}"
            )

    return mapped_features