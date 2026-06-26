from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.settings import ML_PROCESS_IMPORTANCE_DIR, PROCESS_GROUP_SUMMARY_DIR


def load_ml_process_importance_report(
    input_path: str | Path = ML_PROCESS_IMPORTANCE_DIR / "all_targets_ml_process_importance.csv",
) -> pd.DataFrame:
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(
            f"ML process importance report not found: {path}. "
            "Run save_ml_process_importance_report() first."
        )

    return pd.read_csv(path)


def build_process_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "target",
        "process_group",
        "feature",
        "algorithm",
        "correlation_with_target",
        "absolute_correlation",
        "ml_importance",
        "missing_ratio",
        "valid_pair_count",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    work_df = df.copy()

    work_df["correlation_with_target"] = pd.to_numeric(
        work_df["correlation_with_target"], errors="coerce"
    )
    work_df["absolute_correlation"] = pd.to_numeric(
        work_df["absolute_correlation"], errors="coerce"
    )
    work_df["ml_importance"] = pd.to_numeric(
        work_df["ml_importance"], errors="coerce"
    )
    work_df["missing_ratio"] = pd.to_numeric(
        work_df["missing_ratio"], errors="coerce"
    )
    work_df["valid_pair_count"] = pd.to_numeric(
        work_df["valid_pair_count"], errors="coerce"
    )

    summary_rows = []

    group_cols = ["target", "algorithm", "process_group"]

    for (target, algorithm, process_group), group_df in work_df.groupby(group_cols, dropna=False):
        available_importance_df = group_df[group_df["ml_importance"].notna()].copy()

        if not available_importance_df.empty:
            top_importance_row = available_importance_df.sort_values(
                by="ml_importance", ascending=False
            ).iloc[0]
            top_ml_feature = top_importance_row["feature"]
            top_ml_importance = top_importance_row["ml_importance"]
        else:
            top_ml_feature = None
            top_ml_importance = None

        available_corr_df = group_df[group_df["absolute_correlation"].notna()].copy()

        if not available_corr_df.empty:
            top_corr_row = available_corr_df.sort_values(
                by="absolute_correlation", ascending=False
            ).iloc[0]
            top_corr_feature = top_corr_row["feature"]
            top_abs_corr = top_corr_row["absolute_correlation"]
            top_corr_direction = top_corr_row.get("impact_direction", None)
        else:
            top_corr_feature = None
            top_abs_corr = None
            top_corr_direction = None

        summary_rows.append(
            {
                "target": target,
                "algorithm": algorithm,
                "process_group": process_group,
                "feature_count": int(group_df["feature"].nunique()),
                "features_with_ml_importance": int(group_df["ml_importance"].notna().sum()),
                "avg_ml_importance": group_df["ml_importance"].mean(),
                "max_ml_importance": group_df["ml_importance"].max(),
                "top_ml_feature": top_ml_feature,
                "top_ml_importance": top_ml_importance,
                "avg_absolute_correlation": group_df["absolute_correlation"].mean(),
                "max_absolute_correlation": group_df["absolute_correlation"].max(),
                "top_correlation_feature": top_corr_feature,
                "top_absolute_correlation": top_abs_corr,
                "top_correlation_direction": top_corr_direction,
                "avg_missing_ratio": group_df["missing_ratio"].mean(),
                "min_valid_pair_count": group_df["valid_pair_count"].min(),
                "max_valid_pair_count": group_df["valid_pair_count"].max(),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    if summary_df.empty:
        return summary_df

    summary_df["group_score"] = (
        summary_df["avg_ml_importance"].fillna(0)
        + summary_df["avg_absolute_correlation"].fillna(0)
    )

    summary_df = summary_df.sort_values(
        by=["target", "algorithm", "group_score"],
        ascending=[True, True, False],
        na_position="last",
    ).reset_index(drop=True)

    return summary_df


def save_process_group_summary(
    input_path: str | Path = ML_PROCESS_IMPORTANCE_DIR / "all_targets_ml_process_importance.csv",
    output_dir: str | Path = PROCESS_GROUP_SUMMARY_DIR,
) -> pd.DataFrame:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = load_ml_process_importance_report(input_path)
    summary_df = build_process_group_summary(df)

    csv_path = output_path / "all_targets_process_group_summary.csv"
    xlsx_path = output_path / "all_targets_process_group_summary.xlsx"

    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary_df.to_excel(xlsx_path, index=False)

    for target in summary_df["target"].dropna().unique():
        target_df = summary_df[summary_df["target"] == target].copy()

        target_df.to_csv(
            output_path / f"{target}_process_group_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        target_df.to_excel(
            output_path / f"{target}_process_group_summary.xlsx",
            index=False,
        )

    return summary_df