from typing import Dict

MES_TO_ML_FEATURE_MAP = {
    "press_fibre_density": "operPressFibreDensity",
    "cooking_time": "operCookingTime",
    "fibres_density": "operFibresDensity",
    "press_pressure_mid": "pressPressureMid_mean",
    "press_pressure_global": "pressPressureGlobal_mean",
    "belt_speed": "beltSpeed1",
    "raw_thickness": "rawThickness",
}

def map_mes_features_to_ml_features(mes_features: dict) -> dict:
    """
    Maps raw MES features to the corresponding ML features based on the defined mapping.

    Args:
        mes_features (dict): A dictionary of raw MES features.

    Returns:
        dict: A dictionary of mapped ML features.
    """
    mapped_features: Dict[str, any] = {}

    for mes_name, value in mes_features.items():
        ml_name = MES_TO_ML_FEATURE_MAP.get(mes_name)

        if ml_name:
            mapped_features[ml_name] = value

    return mapped_features