from __future__ import annotations

import pandas as pd

from config.columns import pressure_cols, temp_cols, thickness_cols
from utils.decorators import log_dataframe_shape, pipeline_step


def _existing(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [col for col in cols if col in df.columns]


def _safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()

    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def _filter_by_suffix(cols: list[str], suffix: str) -> list[str]:
    return [col for col in cols if col.endswith(suffix)]


def _extract_zone_number(col_name: str) -> int | None:
    digits = "".join(ch for ch in col_name if ch.isdigit())
    return int(digits) if digits else None


def _filter_zone_range(
    cols: list[str],
    start_zone: int,
    end_zone: int,
) -> list[str]:
    selected = []

    for col in cols:
        zone = _extract_zone_number(col)

        if zone is not None and start_zone <= zone <= end_zone:
            selected.append(col)

    return selected


@pipeline_step("Add engineered features")
@log_dataframe_shape("Engineered dataframe")
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    existing_pressure = _existing(out, pressure_cols)
    existing_temp = _existing(out, temp_cols)
    existing_thickness = _existing(out, thickness_cols)

    out = _safe_numeric(
        out,
        existing_pressure + existing_temp + existing_thickness,
    )

    new_cols: dict[str, pd.Series] = {}

    pressure_l = _filter_by_suffix(existing_pressure, "L")
    pressure_c = _filter_by_suffix(existing_pressure, "C")
    pressure_r = _filter_by_suffix(existing_pressure, "R")

    if pressure_l:
        new_cols["pressPressureL_mean"] = out[pressure_l].mean(axis=1)
        new_cols["pressPressureL_std"] = out[pressure_l].std(axis=1)

    if pressure_c:
        new_cols["pressPressureC_mean"] = out[pressure_c].mean(axis=1)
        new_cols["pressPressureC_std"] = out[pressure_c].std(axis=1)

    if pressure_r:
        new_cols["pressPressureR_mean"] = out[pressure_r].mean(axis=1)
        new_cols["pressPressureR_std"] = out[pressure_r].std(axis=1)

    if pressure_l and pressure_r:
        new_cols["pressPressureLR_mean_diff"] = (
            out[pressure_l].mean(axis=1)
            - out[pressure_r].mean(axis=1)
        )

    if existing_pressure:
        pressure_min = out[existing_pressure].min(axis=1)
        pressure_max = out[existing_pressure].max(axis=1)

        new_cols["pressPressureGlobal_mean"] = out[existing_pressure].mean(axis=1)
        new_cols["pressPressureGlobal_std"] = out[existing_pressure].std(axis=1)
        new_cols["pressPressureGlobal_min"] = pressure_min
        new_cols["pressPressureGlobal_max"] = pressure_max
        new_cols["pressPressureGlobal_range"] = pressure_max - pressure_min

    pressure_front = _filter_zone_range(existing_pressure, 1, 7)
    pressure_mid = _filter_zone_range(existing_pressure, 8, 14)
    pressure_end = _filter_zone_range(existing_pressure, 15, 21)

    if pressure_front:
        new_cols["pressPressureFront_mean"] = out[pressure_front].mean(axis=1)

    if pressure_mid:
        new_cols["pressPressureMid_mean"] = out[pressure_mid].mean(axis=1)

    if pressure_end:
        new_cols["pressPressureEnd_mean"] = out[pressure_end].mean(axis=1)

    if pressure_front and pressure_end:
        new_cols["pressPressureFrontEnd_diff"] = (
            out[pressure_front].mean(axis=1)
            - out[pressure_end].mean(axis=1)
        )

    if existing_temp:
        temp_min = out[existing_temp].min(axis=1)
        temp_max = out[existing_temp].max(axis=1)

        new_cols["tempGlobal_mean"] = out[existing_temp].mean(axis=1)
        new_cols["tempGlobal_std"] = out[existing_temp].std(axis=1)
        new_cols["tempGlobal_min"] = temp_min
        new_cols["tempGlobal_max"] = temp_max
        new_cols["tempGlobal_range"] = temp_max - temp_min

    if existing_thickness:
        thickness_min = out[existing_thickness].min(axis=1)
        thickness_max = out[existing_thickness].max(axis=1)

        new_cols["thicknessClosed_mean"] = out[existing_thickness].mean(axis=1)
        new_cols["thicknessClosed_std"] = out[existing_thickness].std(axis=1)
        new_cols["thicknessClosed_min"] = thickness_min
        new_cols["thicknessClosed_max"] = thickness_max
        new_cols["thicknessClosed_range"] = thickness_max - thickness_min

    engineered_df = pd.DataFrame(new_cols, index=out.index)

    return pd.concat([out, engineered_df], axis=1)


def get_engineered_feature_names(df: pd.DataFrame) -> list[str]:
    base_cols = set(df.columns)
    enriched = add_engineered_features(df)

    return [
        col
        for col in enriched.columns
        if col not in base_cols
    ]