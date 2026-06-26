all_features = [
    'operPressFactor', 'operPressSpeed', 'operPressTemp1', 'operPressTemp2', 'operPressTemp3',
    'operPressTemp4', 'operPressTemp5', 'sprayUp', 'pressLeadLeftAct', 'pressLeadRightAct',
    'operBoardDensity', 'operWeightOfBoard', 'catcherRate', 'operPressFibreDensity',
    'cookingPressure', 'operFibrePreHeatTemp', 'operCookingTime', 'distance1', 'distance10',
    'distance11', 'distance12', 'distance13', 'distance14', 'distance15', 'distance16',
    'distance17', 'distance18', 'distance19', 'distance2', 'distance20', 'distance21',
    'distance3', 'distance4', 'distance5', 'distance6', 'distance7', 'distance8', 'distance9',
    'couplingFactor1L', 'couplingFactor10L', 'couplingFactor11L', 'couplingFactor12L',
    'couplingFactor13L', 'couplingFactor14L', 'couplingFactor15L', 'couplingFactor16L',
    'couplingFactor17L', 'couplingFactor18L', 'couplingFactor19L', 'couplingFactor2L',
    'couplingFactor20L', 'couplingFactor21L', 'couplingFactor3L', 'couplingFactor4L',
    'couplingFactor5L', 'couplingFactor6L', 'couplingFactor7L', 'couplingFactor8L',
    'couplingFactor9L', 'couplingFactor1R', 'couplingFactor10R', 'couplingFactor11R',
    'couplingFactor12R', 'couplingFactor13R', 'couplingFactor14R', 'couplingFactor15R',
    'couplingFactor16R', 'couplingFactor17R', 'couplingFactor18R', 'couplingFactor19R',
    'couplingFactor2R', 'couplingFactor20R', 'couplingFactor21R', 'couplingFactor3R',
    'couplingFactor4R', 'couplingFactor5R', 'couplingFactor6R', 'couplingFactor7R',
    'couplingFactor8R', 'couplingFactor9R', 'dischargeSpeed', 'drier1TempOut',
    'fibersHumidityAfterDryer', 'operFibres', 'operFibresDensity', 'operFibreTemp',
    'beltSpeed1', 'operGlue1', 'grindingSpeed', 'pressHotPlateTemp1', 'pressHotPlateTemp2',
    'pressHotPlateTemp3', 'heatInletTemp', 'pressHotPlateTemp4', 'pressHotPlateTemp5',
    'operGlue2', 'operMatTemp', 'operSteamBottom', 'pressLeadLeftSet', 'pressPressure1L',
    'pressPressure10L', 'pressPressure11L', 'pressPressure12L', 'pressPressure13L',
    'pressPressure14L', 'pressPressure15L', 'pressPressure16L', 'pressPressure17L',
    'pressPressure18L', 'pressPressure19L', 'pressPressure2L', 'pressPressure20L',
    'pressPressure21L', 'pressPressure3L', 'pressPressure4L', 'pressPressure5L',
    'pressPressure6L', 'pressPressure7L', 'pressPressure8L', 'pressPressure9L',
    'pressPressure1C', 'pressPressure10C', 'pressPressure11C', 'pressPressure12C',
    'pressPressure13C', 'pressPressure14C', 'pressPressure15C', 'pressPressure16C',
    'pressPressure17C', 'pressPressure18C', 'pressPressure19C', 'pressPressure2C',
    'pressPressure20C', 'pressPressure21C', 'pressPressure3C', 'pressPressure4C',
    'pressPressure5C', 'pressPressure6C', 'pressPressure7C', 'pressPressure8C',
    'pressPressure9C', 'pressPressure1R', 'pressPressure10R', 'pressPressure11R',
    'pressPressure12R', 'pressPressure13R', 'pressPressure14R', 'pressPressure15R',
    'pressPressure16R', 'pressPressure17R', 'pressPressure18R', 'pressPressure19R',
    'pressPressure2R', 'pressPressure20R', 'pressPressure21R', 'pressPressure3R',
    'pressPressure4R', 'pressPressure5R', 'pressPressure6R', 'pressPressure7R',
    'pressPressure8R', 'pressPressure9R', 'operSteamPreHeat', 'operRefTempInDrying',
    'operRefTempOutDrying', 'thicknessClosed1', 'thicknessClosed2', 'thicknessClosed3',
    'thicknessClosed4', 'thicknessClosed5', 'thicknessClosed6', 'thicknessClosed7',
    'operPressWeightOfMat', 'actDistanceInfeedShaft', 'operEmulsion', 'operBlowHardener',
    'operRefLeafTimber', 'operRefConiferTimber', 'operPressFibresMoisture'
]


# Define percentage attributes
percentage_attributes = ['fibersHumidityAfterDryer', 'operPressFibresMoisture', 'catcherRate']

# Define targets
targets = ['labBendingAvg', 'labEModulAvg', 'labTensileAvg', 'labSurfaceSoundnessAvg', 'labDensityAverage']

# Define column categories
distance_cols = ['distance1', 'distance2', 'distance3', 'distance4', 'distance5', 'distance6',
                 'distance7', 'distance8', 'distance9', 'distance10', 'distance11', 'distance12',
                 'distance13', 'distance14', 'distance15', 'distance16', 'distance17', 'distance18',
                 'distance19', 'distance20', 'distance21', 'actDistanceInfeedShaft']

temp_cols = ['pressHotPlateTemp1', 'pressHotPlateTemp2', 'pressHotPlateTemp3',
             'pressHotPlateTemp4', 'pressHotPlateTemp5', 'operPressTemp1', 'operPressTemp2',
             'operPressTemp3', 'operPressTemp4', 'operPressTemp5', 'heatInletTemp', 'operFibreTemp']

pressure_cols = [col for col in all_features if col.startswith('pressPressure')]

thickness_cols = ['thicknessClosed1', 'thicknessClosed2', 'thicknessClosed3', 'thicknessClosed4',
                  'thicknessClosed5', 'thicknessClosed6', 'thicknessClosed7']

speed_cols = ['operPressSpeed', 'dischargeSpeed', 'beltSpeed1', 'grindingSpeed']
time_cols = ['operCookingTime']

other_cols = ['operPressFactor', 'sprayUp', 'pressLeadLeftAct', 'pressLeadRightAct',
              'pressLeadLeftSet', 'operEmulsion', 'operBlowHardener', 'operRefLeafTimber',
              'operRefConiferTimber', 'operGlue1', 'operGlue2']

target_cols = ['labDensityAverage','labBendingAvg', 'labEModulAvg', 'labTensileAvg', 'labSurfaceSoundnessAvg']

positive_cols = [
    'operBoardDensity', 'labDensityAverage', 'operFibresDensity', 'operPressFibreDensity',
    'operFibrePreHeatTemp', 'drier1TempOut', 'operMatTemp', 'operRefTempInDrying',
    'operRefTempOutDrying', 'cookingPressure', 'operSteamBottom', 'operSteamPreHeat',
    'fibersHumidityAfterDryer', 'operPressFibresMoisture',
    'operWeightOfBoard', 'operPressWeightOfMat', 'operFibres',
    'couplingFactor1L', 'couplingFactor2L', 'couplingFactor3L', 'couplingFactor4L',
    'couplingFactor5L', 'couplingFactor6L', 'couplingFactor7L', 'couplingFactor8L',
    'couplingFactor9L', 'couplingFactor10L', 'couplingFactor11L', 'couplingFactor12L',
    'couplingFactor13L', 'couplingFactor14L', 'couplingFactor15L', 'couplingFactor16L',
    'couplingFactor17L', 'couplingFactor18L', 'couplingFactor19L', 'couplingFactor20L',
    'couplingFactor21L', 'couplingFactor1R', 'couplingFactor2R', 'couplingFactor3R',
    'couplingFactor4R', 'couplingFactor5R', 'couplingFactor6R', 'couplingFactor7R',
    'couplingFactor8R', 'couplingFactor9R', 'couplingFactor10R', 'couplingFactor11R',
    'couplingFactor12R', 'couplingFactor13R', 'couplingFactor14R', 'couplingFactor15R',
    'couplingFactor16R', 'couplingFactor17R', 'couplingFactor18R', 'couplingFactor19R',
    'couplingFactor20R', 'couplingFactor21R'
]

# PRIMARY_TARGET= 'labDensityAverage'
# PRIMARY_TARGETs= ['labDensityAverage', 'labBendingAvg']
PRIMARY_TARGETs= ['labDensityAverage','labBendingAvg', 'labEModulAvg', 'labTensileAvg', 'labSurfaceSoundnessAvg']

TARGET_LABELS = {
    "labDensityAverage": "Density",
    "labBendingAvg": "Bending strength",
    "labEModulAvg": "E-modulus",
    "labTensileAvg": "Tensile strength",
    "labSurfaceSoundnessAvg": "Surface soundness",
}

#metadata_columns = ['labcut.id', 'baseProductionDate']
ID_COLUMNS = ['labcut.id']
TIME_COLUMN = 'baseProductionDate'

STRICT_REQUIRED_COLUMNS = ID_COLUMNS + [TIME_COLUMN] + PRIMARY_TARGETs
OPTIONAL_FEATURE_COLUMNS = all_features.copy()

# REQUIRED_COLUMNS = all_features + targets + ID_COLUMNS + [TIME_COLUMN]
MODEL_FEATURES = all_features.copy()
ALL_TARGETS = targets.copy()


engineered_only_features = [
    "rawThickness",
    "beltSpeed1",
    "pressPressureL_mean",
    "pressPressureL_std",
    "pressPressureC_mean",
    "pressPressureC_std",
    "pressPressureR_mean",
    "pressPressureR_std",
    "pressPressureLR_mean_diff",
    "pressPressureGlobal_mean",
    "pressPressureGlobal_std",
    "pressPressureGlobal_min",
    "pressPressureGlobal_max",
    "pressPressureGlobal_range",
    "pressPressureFront_mean",
    "pressPressureMid_mean",
    "pressPressureEnd_mean",
    "pressPressureFrontEnd_diff",
    "tempGlobal_mean",
    "tempGlobal_std",
    "tempGlobal_min",
    "tempGlobal_max",
    "tempGlobal_range",
    "thicknessClosed_mean",
    "thicknessClosed_std",
    "thicknessClosed_min",
    "thicknessClosed_max",
    "thicknessClosed_range",
]

PROCESS_FEATURE_GROUPS = {
    "fiber_preparation": [
        "operCookingTime",
        "cookingPressure",
        "operFibrePreHeatTemp",
        "operFibreTemp",
        "operFibresDensity",
        "operPressFibreDensity",
        "operPressFibresMoisture",
        "operSteamPreHeat",
        "operSteamBottom",
    ],

    "wood_mix": [
        "operRefLeafTimber",
        "operRefConiferTimber",
    ],

    "mat_forming": [
        "operMatTemp",
        "operPressWeightOfMat",
        "operBoardDensity",
        "operWeightOfBoard",
        "catcherRate",
    ],

    "press_process": [
        "operPressFactor",
        "operPressSpeed",
        "pressLeadLeftAct",
        "pressLeadRightAct",
        "pressLeadLeftSet",
    ],

    "glue_chemical": [
        "operGlue1",
        "operGlue2",
        "operEmulsion",
        "operBlowHardener",
    ],

    "temperature_process": [
        "operPressTemp1",
        "operPressTemp2",
        "operPressTemp3",
        "operPressTemp4",
        "operPressTemp5",
        "pressHotPlateTemp1",
        "pressHotPlateTemp2",
        "pressHotPlateTemp3",
        "pressHotPlateTemp4",
        "pressHotPlateTemp5",
        "heatInletTemp",
        "drier1TempOut",
        "operRefTempInDrying",
        "operRefTempOutDrying",
    ],

    "speed_process": [
        "dischargeSpeed",
        "beltSpeed1",
        "grindingSpeed",
    ],
}

FORCED_PRODUCTION_FEATURES = [
    "operCookingTime",
    "operFibresDensity",
    "operPressFibreDensity",
    "pressPressureMid_mean",
]


REQUIRED_FINAL_EXPLAINABILITY_COLUMNS = {
    "target",
    "algorithm",
    "process_group",
    "feature",
    "impact_direction",
    "correlation_with_target",
    "absolute_correlation",
    "ml_importance",
    "mean_abs_shap",
    "final_explainability_score",
    "missing_ratio",
    "valid_pair_count",
}

NUMERIC_FINAL_EXPLAINABILITY_COLUMNS = [
    "correlation_with_target",
    "absolute_correlation",
    "ml_importance",
    "mean_abs_shap",
    "final_explainability_score",
    "missing_ratio",
    "valid_pair_count",
]

RECOMMENDED_ACTION_TEXT = (
    "Review this signal together with production/lab context before making "
    "process changes. Use it as decision support, not as an automatic "
    "control instruction."
)




REQUIRED_ML_COLUMNS = {
    "target",
    "algorithm",
    "artifact_model_name",
    "process_group",
    "feature",
    "impact_direction",
    "correlation_with_target",
    "absolute_correlation",
    "ml_importance",
    "missing_ratio",
    "valid_pair_count",
}

REQUIRED_SHAP_COLUMNS = {
    "target",
    "algorithm",
    "artifact_model_name",
    "feature",
    "shap_rank",
    "mean_abs_shap",
}


FINAL_REPORT_SORT_COLUMNS = [
    "target",
    "algorithm",
    "final_explainability_score",
    "mean_abs_shap",
    "ml_importance",
    "absolute_correlation",
]

FINAL_REPORT_SORT_ASCENDING = [
    True,
    True,
    False,
    False,
    False,
    False,
]