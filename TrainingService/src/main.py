from __future__ import annotations

import time
from typing import Any

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from config.columns import PRIMARY_TARGETs, TIME_COLUMN, engineered_only_features
from config.settings import (
    ARTIFACTS_DIR,
    BEST_MODELS_REGISTRY_PATH,
    ENABLE_ARTIFACT_CLEAN_BEFORE_RUN,
    PROCESS_RECOMMENDATION_TOP_N,
    RANDOM_STATE,
    SHAP_MAX_ROWS,
    TIMING_REPORT_PATH,
)
from data.cleaner import clean_dataframe
from data.ingest_raw_ksoft import ingest_raw_ksoft_file
from data.validator import validate_dataframe
from eda.audit import run_initial_audit
from eda.data_retention_audit import save_pre_ml_data_retention_reports
from features.engineering import add_engineered_features
from models.ensemble_selection import try_build_xgb_rf_ensemble
from models.model_catalog import get_algorithm_configs
from models.registry import (
    log_best_models_summary,
    save_best_models_registry,
)
from pipeline.evaluation_comparison_pipeline import run_evaluation_comparison
from pipeline.training_pipeline import run_single_target_regression_pipeline
from reports.final_explainability_report import save_final_explainability_report
from reports.ml_process_importance_report import save_ml_process_importance_report
from reports.process_group_summary import save_process_group_summary
from reports.process_impact_report import save_process_impact_reports
from reports.process_recommendation_report import save_process_recommendation_report
from reports.shap_explainability_report import save_shap_explainability_reports
from utils.artifact_cleaner import clean_artifacts_dir
from utils.comparison_utils import save_comparison_summary
from utils.decorators import pipeline_step
from utils.logging_utils import get_logger
from utils.seed import set_seed


logger = get_logger(__name__)


@pipeline_step("Save timing report")
def save_timing_report(rows: list[dict[str, Any]]) -> None:
    if not rows:
        logger.warning("Timing report was not saved because timing rows are empty.")
        return

    TIMING_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    timing_df = pd.DataFrame(rows)

    timing_df.to_csv(
        TIMING_REPORT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info("Timing report saved: %s", TIMING_REPORT_PATH)


def _build_timing_row(
    *,
    target: str,
    algorithm: str,
    artifact_model_name: str,
    duration_sec: float,
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "target": target,
        "algorithm": algorithm,
        "artifact_model_name": artifact_model_name,
        "duration_sec": round(duration_sec, 2),
        "test_r2": metrics["r2"],
        "test_rmse": metrics["rmse"],
        "test_mae": metrics["mae"],
    }


@pipeline_step("Run TrainingService pipeline")
def main() -> None:
    pipeline_start = time.time()
    timing_rows: list[dict[str, Any]] = []
    all_results: list[Any] = []
    results_by_target: dict[str, dict[str, Any]] = {}

    logger.info("TrainingService pipeline started.")
    logger.info("Random state: %s", RANDOM_STATE)

    set_seed(RANDOM_STATE)

    if ENABLE_ARTIFACT_CLEAN_BEFORE_RUN:
        clean_artifacts_dir(
            artifacts_root=ARTIFACTS_DIR,
            keep_registry=False,
        )
    else:
        logger.info("Artifact cleanup skipped by settings.")

    raw_df = ingest_raw_ksoft_file()
    logger.info("Raw data shape: %s", raw_df.shape)

    validation_report = validate_dataframe(raw_df)
    missing_required = validation_report["schema_checks"]["missing_required_columns"]

    if missing_required:
        logger.warning("Missing required columns: %s", missing_required)

    clean_df, dropped_features = clean_dataframe(raw_df)

    logger.info("Cleaned data shape: %s", clean_df.shape)
    logger.info("Dropped feature count: %s", len(dropped_features))

    retention_outputs = save_pre_ml_data_retention_reports(
        raw_df=raw_df,
        final_df=clean_df,
    )

    logger.info("Data retention reports saved:")
    for report_name, report_path in retention_outputs.items():
        logger.info(" - %s: %s", report_name, report_path)

    engineered_df = add_engineered_features(clean_df)
    logger.info("Engineered data shape: %s", engineered_df.shape)

    run_initial_audit(engineered_df)
    save_process_impact_reports(engineered_df)

    candidate_features = engineered_only_features.copy()
    algorithm_configs = get_algorithm_configs()

    logger.info("Candidate features count: %s", len(candidate_features))
    logger.info(
        "Configured algorithms: %s",
        [cfg.algorithm_name for cfg in algorithm_configs],
    )

    run_evaluation_comparison(
        df=engineered_df,
        candidate_features=candidate_features,
        targets=PRIMARY_TARGETs,
        algorithm_configs=algorithm_configs,
        time_column=TIME_COLUMN,
    )

    target_iterator = tqdm(PRIMARY_TARGETs, desc="Targets") if tqdm else PRIMARY_TARGETs

    for target in target_iterator:
        target_start = time.time()

        logger.info("Production training target: %s", target)

        results_by_target[target] = {}

        algo_iterator = (
            tqdm(
                algorithm_configs,
                desc=f"Algorithms for {target}",
                leave=False,
            )
            if tqdm
            else algorithm_configs
        )

        for algo_cfg in algo_iterator:
            algo_start = time.time()
            artifact_model_name = f"{target}_{algo_cfg.artifact_suffix}"

            result = run_single_target_regression_pipeline(
                df=engineered_df,
                candidate_feature_columns=candidate_features,
                target_column=target,
                algorithm_name=algo_cfg.algorithm_name,
                artifact_model_name=artifact_model_name,
                model_params=algo_cfg.model_params,
                artifacts_root=ARTIFACTS_DIR,
                time_column=TIME_COLUMN,
                use_time_based_split=True,
                notes=algo_cfg.notes,
                save_artifacts=True,
            )

            duration_sec = time.time() - algo_start

            timing_rows.append(
                _build_timing_row(
                    target=target,
                    algorithm=algo_cfg.algorithm_name,
                    artifact_model_name=artifact_model_name,
                    duration_sec=duration_sec,
                    metrics=result.metrics["test"],
                )
            )

            all_results.append(result)
            results_by_target[target][algo_cfg.algorithm_name] = result

            logger.info(
                "Model completed | target=%s | algorithm=%s | duration=%.2f sec",
                target,
                algo_cfg.algorithm_name,
                duration_sec,
            )

        if "rf" in results_by_target[target] and "xgb" in results_by_target[target]:
            ensemble_start = time.time()

            ensemble_result, val_metrics, test_metrics, best_weight = (
                try_build_xgb_rf_ensemble(
                    target,
                    results_by_target[target]["rf"],
                    results_by_target[target]["xgb"],
                )
            )

            ensemble_duration = time.time() - ensemble_start

            logger.info(
                "Ensemble checked | target=%s | xgb_weight=%.2f",
                target,
                best_weight,
            )

            if ensemble_result is not None:
                all_results.append(ensemble_result)
                results_by_target[target]["ensemble_xgb_rf"] = ensemble_result

                timing_rows.append(
                    _build_timing_row(
                        target=target,
                        algorithm="ensemble_xgb_rf",
                        artifact_model_name=ensemble_result.artifact_model_name,
                        duration_sec=ensemble_duration,
                        metrics=test_metrics,
                    )
                )
        else:
            logger.warning(
                "Ensemble skipped | target=%s | RF or XGB result missing",
                target,
            )

        logger.info(
            "Target completed | target=%s | duration=%.2f sec",
            target,
            time.time() - target_start,
        )

    save_comparison_summary(all_results)

    best_models = save_best_models_registry(
        all_results=all_results,
        output_path=BEST_MODELS_REGISTRY_PATH,
    )

    log_best_models_summary(best_models)

    save_ml_process_importance_report()
    save_process_group_summary()

    save_shap_explainability_reports(
        df=engineered_df,
        max_rows=SHAP_MAX_ROWS,
    )

    save_final_explainability_report()

    save_process_recommendation_report(
        top_n_per_target=PROCESS_RECOMMENDATION_TOP_N,
    )

    save_timing_report(timing_rows)

    logger.info(
        "TrainingService pipeline completed | duration=%.2f sec",
        time.time() - pipeline_start,
    )


if __name__ == "__main__":
    main()