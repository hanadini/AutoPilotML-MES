from __future__ import annotations

from typing import Any

import pandas as pd

from config.settings import RANDOM_STATE
from pipeline.training_pipeline import run_single_target_regression_pipeline
from utils.decorators import log_dataframe_shape, pipeline_step


@pipeline_step("Run single target model comparison")
@log_dataframe_shape("Model comparison dataframe")
def run_single_target_model_comparison(
    *,
    df: pd.DataFrame,
    candidate_feature_columns: list[str],
    target_column: str,
    artifacts_root: str,
    base_artifact_name: str,
    algorithms: list[dict[str, Any]],
    preprocessor: Any | None = None,
    notes: str | None = None,
    random_state: int = RANDOM_STATE,
    use_time_based_split: bool = True,
    time_column: str | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for algo_cfg in algorithms:
        algorithm_name = algo_cfg["algorithm_name"]
        model_params = algo_cfg.get("model_params", {})

        artifact_model_name = (
            f"{base_artifact_name}_{algorithm_name}_v1"
        )

        result = run_single_target_regression_pipeline(
            df=df,
            candidate_feature_columns=candidate_feature_columns,
            target_column=target_column,
            algorithm_name=algorithm_name,
            artifact_model_name=artifact_model_name,
            model_params=model_params,
            artifacts_root=artifacts_root,
            preprocessor=preprocessor,
            notes=notes,
            random_state=random_state,
            use_time_based_split=use_time_based_split,
            time_column=time_column,
        )

        rows.append(
            {
                "algorithm_name": result.algorithm_name,
                "artifact_model_name": result.artifact_model_name,
                "target_name": result.target_name,
                "val_mae": result.metrics["validation"]["mae"],
                "val_rmse": result.metrics["validation"]["rmse"],
                "val_r2": result.metrics["validation"]["r2"],
                "test_mae": result.metrics["test"]["mae"],
                "test_rmse": result.metrics["test"]["rmse"],
                "test_r2": result.metrics["test"]["r2"],
                "artifact_dir": result.artifact_dir,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            by=["test_r2", "val_r2"],
            ascending=False,
        )
        .reset_index(drop=True)
    )