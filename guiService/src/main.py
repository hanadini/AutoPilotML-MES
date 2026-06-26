from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, text

from config.config import (
    ALL_TARGETS,
    CONTROL_FEATURE_LABELS,
    CONTROL_FEATURE_UNITS,
    CORE_PROCESS_FEATURES,
    DEFAULT_MANUAL_FEATURE_VALUES,
    FORCED_PRODUCTION_FEATURES,
    GUI_REFRESH_SECONDS,
    HISTORY_ALLOWED_LIMITS,
    HISTORY_DEFAULT_LIMIT,
    TARGET_LABELS, BASE_DIR,
)
from service.historyLoader import (
    get_history_target_options,
    load_history,
    load_history_entry_by_timestamp,
)
from service.predictionClient import (
    PredictionServiceError,
    get_registered_models,
    health_check,
    predict_all,
)
from service.predictionLogger import log_prediction
from service.prediction_formatter import (
    build_prediction_display_value,
    build_prediction_status,
    format_prediction_result_item,
)

from service.monitoring_client import (
    MonitoringServiceError,
    create_monitoring_snapshot,
    get_monitoring_dashboard,
    get_monitoring_snapshots,
    monitoring_health_check,
)

from fastapi.responses import HTMLResponse, RedirectResponse

DATABASE_URL = "postgresql+psycopg2://postgres:1@localhost:5432/mdf1_ml"

app = FastAPI(
    title="MDF1 GUI Service",
    description="Web GUI for MDF1 prediction system",
    version="1.0.0",
)

templates = Jinja2Templates(directory=str(BASE_DIR /"templates"))

engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
)

LAST_MANUAL_VALUES: Dict[str, Any] = DEFAULT_MANUAL_FEATURE_VALUES.copy()
LAST_BASELINE_VALUES: Dict[str, Any] = DEFAULT_MANUAL_FEATURE_VALUES.copy()
LAST_SCENARIO_VALUES: Dict[str, Any] = DEFAULT_MANUAL_FEATURE_VALUES.copy()


def get_default_features() -> Dict[str, Any]:
    return DEFAULT_MANUAL_FEATURE_VALUES.copy()


def get_target_options() -> List[Dict[str, str]]:
    return [
        {
            "name": target,
            "label": TARGET_LABELS.get(target, target),
        }
        for target in ALL_TARGETS
    ]


def build_input_features() -> Dict[str, Dict[str, Any]]:
    feature_names = list(
        dict.fromkeys(
            CORE_PROCESS_FEATURES + FORCED_PRODUCTION_FEATURES
        )
    )

    return {
        feature_name: {
            "label": CONTROL_FEATURE_LABELS.get(feature_name, feature_name),
            "unit": CONTROL_FEATURE_UNITS.get(feature_name, ""),
            "default_value": DEFAULT_MANUAL_FEATURE_VALUES.get(feature_name, 0),
            "is_forced_production_feature": feature_name in FORCED_PRODUCTION_FEATURES,
        }
        for feature_name in feature_names
    }


def build_quick_actions() -> Dict[str, Dict[str, Any]]:
    return {
        "beltSpeed1": {"label": "Belt Speed", "step": 1},
        "pressPressureGlobal_mean": {"label": "Global Pressure", "step": 1},
        "pressPressureMid_mean": {"label": "Mid Pressure", "step": 1},
        "operCookingTime": {"label": "Cooking Time", "step": 10},
        "operFibresDensity": {"label": "Fibre Density", "step": 10},
        "operPressFibreDensity": {"label": "Press Fibre", "step": 20},
    }


def parse_feature_values(
    form: Any,
    *,
    prefix: str = "",
) -> Dict[str, Any]:
    values: Dict[str, Any] = {}

    for feature_name in build_input_features().keys():
        field_name = "{}{}".format(prefix, feature_name)

        if field_name not in form:
            values[feature_name] = DEFAULT_MANUAL_FEATURE_VALUES.get(
                feature_name,
                0,
            )
            continue

        values[feature_name] = float(form[field_name])

    return values


def enrich_prediction_response(
    response: Dict[str, Any],
) -> Dict[str, Any]:
    enriched_response = dict(response)
    predictions = enriched_response.get("predictions", {})

    enriched_predictions: Dict[str, Any] = {}

    for target_name, item in predictions.items():
        prediction_value = item.get("prediction")

        if prediction_value is None:
            enriched_predictions[target_name] = item
            continue

        enriched_predictions[target_name] = format_prediction_result_item(
            target_name=target_name,
            prediction_value=float(prediction_value),
            item=item,
        )

    enriched_response["predictions"] = enriched_predictions

    return enriched_response


def build_comparison(
    baseline_result: Dict[str, Any],
    scenario_result: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    comparison: Dict[str, Dict[str, Any]] = {}

    baseline_predictions = baseline_result.get("predictions", {})
    scenario_predictions = scenario_result.get("predictions", {})

    for target_name, baseline_data in baseline_predictions.items():
        scenario_data = scenario_predictions.get(target_name)

        if not scenario_data:
            continue

        baseline_prediction = baseline_data.get("prediction")
        scenario_prediction = scenario_data.get("prediction")

        if baseline_prediction is None or scenario_prediction is None:
            continue

        delta = float(scenario_prediction) - float(baseline_prediction)

        _, _, risk_level = build_prediction_status(
            target_name=target_name,
            prediction_value=float(scenario_prediction),
        )

        comparison[target_name] = {
            "target_label": TARGET_LABELS.get(target_name, target_name),
            "baseline_prediction": float(baseline_prediction),
            "scenario_prediction": float(scenario_prediction),
            "baseline_display_value": build_prediction_display_value(
                float(baseline_prediction),
                target_name,
            ),
            "scenario_display_value": build_prediction_display_value(
                float(scenario_prediction),
                target_name,
            ),
            "delta": delta,
            "risk_level": risk_level,
        }

    return comparison


def build_sensitivity_scenarios(
    baseline_values: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    scenarios: Dict[str, Dict[str, Any]] = {}

    for feature_name, action in build_quick_actions().items():
        step = float(action["step"])

        base_value = float(
            baseline_values.get(
                feature_name,
                DEFAULT_MANUAL_FEATURE_VALUES.get(feature_name, 0),
            )
        )

        scenarios["{} + {:g}".format(feature_name, step)] = {
            **baseline_values,
            feature_name: base_value + step,
        }

        scenarios["{} - {:g}".format(feature_name, step)] = {
            **baseline_values,
            feature_name: base_value - step,
        }

    return scenarios


def build_sensitivity_results(
    baseline_values: Dict[str, Any],
    baseline_result: Dict[str, Any],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    sensitivity_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    baseline_predictions = baseline_result.get("predictions", {})
    sensitivity_scenarios = build_sensitivity_scenarios(baseline_values)

    for scenario_name, scenario_features in sensitivity_scenarios.items():
        try:
            sensitivity_result = enrich_prediction_response(
                predict_all(
                    features=scenario_features,
                    request_source="GUIService-Sensitivity",
                )
            )
        except PredictionServiceError:
            continue

        sensitivity_predictions = sensitivity_result.get("predictions", {})
        target_comparison: Dict[str, Dict[str, Any]] = {}

        for target_name, baseline_data in baseline_predictions.items():
            sensitivity_data = sensitivity_predictions.get(target_name)

            if not sensitivity_data:
                continue

            baseline_prediction = baseline_data.get("prediction")
            changed_prediction = sensitivity_data.get("prediction")

            if baseline_prediction is None or changed_prediction is None:
                continue

            target_comparison[target_name] = {
                "target_label": TARGET_LABELS.get(target_name, target_name),
                "baseline_prediction": float(baseline_prediction),
                "changed_prediction": float(changed_prediction),
                "delta": float(changed_prediction) - float(baseline_prediction),
            }

        sensitivity_results[scenario_name] = target_comparison

    return sensitivity_results


def build_sensitivity_summary(
    sensitivity_results: Dict[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}

    for target_name in ALL_TARGETS:
        best_feature = None
        max_impact = 0.0

        for scenario_name, targets in sensitivity_results.items():
            item = targets.get(target_name)

            if not item:
                continue

            impact = abs(float(item["delta"]))

            if impact > max_impact:
                max_impact = impact
                best_feature = scenario_name

        summary[target_name] = {
            "target_label": TARGET_LABELS.get(target_name, target_name),
            "most_impactful_change": best_feature,
            "impact_value": max_impact,
        }

    return summary


def build_recommendations_from_summary(
    sensitivity_summary: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    recommendations: Dict[str, Dict[str, Any]] = {}

    for target_name, item in sensitivity_summary.items():
        impact_value = float(item.get("impact_value", 0))

        if impact_value >= 0.75:
            risk_level = "HIGH"
        elif impact_value >= 0.10:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        most_impactful_change = item.get("most_impactful_change")

        recommendations[target_name] = {
            "target_label": TARGET_LABELS.get(target_name, target_name),
            "risk_level": risk_level,
            "impact_value": impact_value,
            "recommended_action": (
                "Review process change"
                if most_impactful_change
                else "No strong action required"
            ),
            "message": (
                "Most influential simulated change: {}".format(most_impactful_change)
                if most_impactful_change
                else "No significant sensitivity detected."
            ),
        }

    return recommendations


def build_mini_trend_data(
    sensitivity_results: Dict[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    trend_data: Dict[str, List[Dict[str, Any]]] = {}

    for scenario_name, targets in sensitivity_results.items():
        for target_name, item in targets.items():
            trend_data.setdefault(target_name, []).append(
                {
                    "label": scenario_name,
                    "delta": item.get("delta", 0),
                }
            )

    return trend_data


def get_shift_window(
    time_range: str,
) -> Tuple[Optional[datetime], Optional[datetime]]:
    now = datetime.now()

    if time_range == "shift1":
        start = now.replace(hour=6, minute=0, second=0, microsecond=0)
        end = now.replace(hour=14, minute=0, second=0, microsecond=0)
        return start, end

    if time_range == "shift2":
        start = now.replace(hour=14, minute=0, second=0, microsecond=0)
        end = now.replace(hour=22, minute=0, second=0, microsecond=0)
        return start, end

    if time_range == "shift3":
        if now.time() >= time(22, 0):
            start = now.replace(hour=22, minute=0, second=0, microsecond=0)
        else:
            yesterday = now - timedelta(days=1)
            start = yesterday.replace(
                hour=22,
                minute=0,
                second=0,
                microsecond=0,
            )

        return start, start + timedelta(hours=8)

    return None, None


def build_latest_kpi(
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    high_risk_count = 0
    density_values: List[float] = []
    bending_values: List[float] = []

    for row in rows:
        prediction_value = row.get("prediction_value")
        target_name = row.get("target_name")

        if prediction_value is None:
            continue

        _, _, risk_level = build_prediction_status(
            target_name=target_name,
            prediction_value=float(prediction_value),
        )

        row["risk_level"] = risk_level

        if risk_level == "HIGH":
            high_risk_count += 1

        if target_name == "labDensityAverage":
            density_values.append(float(prediction_value))

        if target_name == "labBendingAvg":
            bending_values.append(float(prediction_value))

    return {
        "total_predictions": len(rows),
        "density_count": len(density_values),
        "bending_count": len(bending_values),
        "avg_density": (
            round(sum(density_values) / len(density_values), 3)
            if density_values
            else None
        ),
        "avg_bending": (
            round(sum(bending_values) / len(bending_values), 3)
            if bending_values
            else None
        ),
        "high_risk_count": high_risk_count,
    }


def enrich_latest_rows(
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    enriched_rows: List[Dict[str, Any]] = []

    for row in rows:
        target_name = row.get("target_name")
        prediction_value = row.get("prediction_value")

        if target_name and prediction_value is not None:
            row_class, prediction_class, risk_level = build_prediction_status(
                target_name=target_name,
                prediction_value=float(prediction_value),
            )

            row["row_status_class"] = row_class
            row["prediction_status_class"] = prediction_class
            row["risk_level"] = risk_level
            row["target_label"] = TARGET_LABELS.get(target_name, target_name)
            row["prediction_display_value"] = build_prediction_display_value(
                prediction_value=float(prediction_value),
                target_name=target_name,
            )

        features = row.get("features") or {}

        if not isinstance(features, dict):
            features = {}

        row["control_features"] = {}

        for feature_name in CORE_PROCESS_FEATURES + FORCED_PRODUCTION_FEATURES:
            if feature_name not in features:
                continue

            row["control_features"][feature_name] = {
                "label": CONTROL_FEATURE_LABELS.get(feature_name, feature_name),
                "value": features.get(feature_name),
                "unit": CONTROL_FEATURE_UNITS.get(feature_name, ""),
                "is_forced_production_feature": (
                    feature_name in FORCED_PRODUCTION_FEATURES
                ),
            }

        enriched_rows.append(row)

    return enriched_rows


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    prediction_service_healthy = False
    registered_model_count: Union[int, str] = "N/A"

    try:
        health_check()
        prediction_service_healthy = True
    except PredictionServiceError:
        prediction_service_healthy = False

    try:
        model_response = get_registered_models()
        registered_model_count = len(model_response.get("models", {}))
    except PredictionServiceError:
        registered_model_count = "N/A"

    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "title": "MDF1 AI Production Cockpit",
            "target_count": len(ALL_TARGETS),
            "target_labels": list(TARGET_LABELS.values()),
            "registered_model_count": registered_model_count,
            "prediction_service_healthy": prediction_service_healthy,
        },
    )


@app.get("/manual-prediction", response_class=HTMLResponse)
def manual_prediction_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "manual_prediction.html",
        {
            "request": request,
            "title": "Manual Prediction",
            "result": None,
            "error": None,
            "submitted_values": LAST_MANUAL_VALUES.copy(),
            "input_features": build_input_features(),
        },
    )


@app.post("/manual-prediction", response_class=HTMLResponse)
async def manual_prediction_submit(request: Request) -> HTMLResponse:
    global LAST_MANUAL_VALUES

    form = await request.form()
    submitted_values = parse_feature_values(form)

    LAST_MANUAL_VALUES = submitted_values.copy()

    try:
        result = enrich_prediction_response(
            predict_all(
                features=submitted_values,
                request_source="GUIService",
            )
        )

        log_prediction(
            {
                "source": "GUIService",
                "mode": "manual_prediction",
                "features": submitted_values,
                "predictions": result.get("predictions", {}),
                "model_version": result.get("model_version"),
                "actual_values": None,
            }
        )

        return templates.TemplateResponse(
            "manual_prediction.html",
            {
                "request": request,
                "title": "Manual Prediction",
                "result": result,
                "error": None,
                "submitted_values": submitted_values,
                "input_features": build_input_features(),
            },
        )

    except PredictionServiceError as exc:
        return templates.TemplateResponse(
            "manual_prediction.html",
            {
                "request": request,
                "title": "Manual Prediction",
                "result": None,
                "error": str(exc),
                "submitted_values": submitted_values,
                "input_features": build_input_features(),
            },
        )


@app.get("/simulation", response_class=HTMLResponse)
def simulation_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "simulation.html",
        {
            "request": request,
            "title": "What-if Simulation",
            "baseline_result": None,
            "scenario_result": None,
            "comparison": None,
            "sensitivity_results": None,
            "sensitivity_summary": None,
            "recommendations": None,
            "mini_trend_data": None,
            "error": None,
            "baseline_values": LAST_BASELINE_VALUES.copy(),
            "scenario_values": LAST_SCENARIO_VALUES.copy(),
            "input_features": build_input_features(),
            "quick_actions": build_quick_actions(),
            "forced_feature_names": FORCED_PRODUCTION_FEATURES,
            "replay_timestamp": None,
        },
    )


@app.post("/simulation", response_class=HTMLResponse)
async def simulation_submit(request: Request) -> HTMLResponse:
    global LAST_BASELINE_VALUES
    global LAST_SCENARIO_VALUES

    form = await request.form()

    baseline_values = parse_feature_values(form, prefix="baseline_")
    scenario_values = parse_feature_values(form, prefix="scenario_")

    LAST_BASELINE_VALUES = baseline_values.copy()
    LAST_SCENARIO_VALUES = scenario_values.copy()

    try:
        baseline_result = enrich_prediction_response(
            predict_all(
                features=baseline_values,
                request_source="GUIService-Simulation-Baseline",
            )
        )

        scenario_result = enrich_prediction_response(
            predict_all(
                features=scenario_values,
                request_source="GUIService-Simulation-Scenario",
            )
        )

        comparison = build_comparison(
            baseline_result=baseline_result,
            scenario_result=scenario_result,
        )

        sensitivity_results = build_sensitivity_results(
            baseline_values=baseline_values,
            baseline_result=baseline_result,
        )

        sensitivity_summary = build_sensitivity_summary(
            sensitivity_results
        )

        recommendations = build_recommendations_from_summary(
            sensitivity_summary
        )

        mini_trend_data = build_mini_trend_data(
            sensitivity_results
        )

        log_prediction(
            {
                "source": "GUIService",
                "mode": "what_if_simulation",
                "baseline": baseline_values,
                "scenario": scenario_values,
                "baseline_predictions": baseline_result.get("predictions", {}),
                "scenario_predictions": scenario_result.get("predictions", {}),
                "comparison": comparison,
                "recommendations": recommendations,
                "sensitivity_summary": sensitivity_summary,
                "mini_trend_data": mini_trend_data,
                "actual_values": None,
            }
        )

        return templates.TemplateResponse(
            "simulation.html",
            {
                "request": request,
                "title": "What-if Simulation",
                "baseline_result": baseline_result,
                "scenario_result": scenario_result,
                "comparison": comparison,
                "sensitivity_results": sensitivity_results,
                "sensitivity_summary": sensitivity_summary,
                "recommendations": recommendations,
                "mini_trend_data": mini_trend_data,
                "error": None,
                "baseline_values": baseline_values,
                "scenario_values": scenario_values,
                "input_features": build_input_features(),
                "quick_actions": build_quick_actions(),
                "forced_feature_names": FORCED_PRODUCTION_FEATURES,
                "replay_timestamp": None,
            },
        )

    except PredictionServiceError as exc:
        return templates.TemplateResponse(
            "simulation.html",
            {
                "request": request,
                "title": "What-if Simulation",
                "baseline_result": None,
                "scenario_result": None,
                "comparison": None,
                "sensitivity_results": None,
                "sensitivity_summary": None,
                "recommendations": None,
                "mini_trend_data": None,
                "error": str(exc),
                "baseline_values": baseline_values,
                "scenario_values": scenario_values,
                "input_features": build_input_features(),
                "quick_actions": build_quick_actions(),
                "forced_feature_names": FORCED_PRODUCTION_FEATURES,
                "replay_timestamp": None,
            },
        )


@app.get("/history", response_class=HTMLResponse)
def history_page(
    request: Request,
    risk: Optional[str] = None,
    target: Optional[str] = None,
    limit: int = HISTORY_DEFAULT_LIMIT,
) -> HTMLResponse:
    if limit not in HISTORY_ALLOWED_LIMITS:
        limit = HISTORY_DEFAULT_LIMIT

    history = load_history(
        limit=limit,
        risk=risk,
        target=target,
    )

    return templates.TemplateResponse(
        "history.html",
        {
            "request": request,
            "title": "Simulation / Prediction History",
            "history": history,
            "selected_risk": risk,
            "selected_target": target,
            "selected_target_label": (
                TARGET_LABELS.get(target, target)
                if target
                else None
            ),
            "selected_limit": limit,
            "allowed_limits": HISTORY_ALLOWED_LIMITS,
            "target_options": get_history_target_options(),
        },
    )


@app.get("/history/replay", response_class=HTMLResponse)
def replay_history_entry(
    request: Request,
    timestamp: str,
) -> HTMLResponse:
    item = load_history_entry_by_timestamp(timestamp)

    if item is None:
        raise HTTPException(
            status_code=404,
            detail="History entry not found",
        )

    baseline_values = {
        **get_default_features(),
        **(item.get("baseline") or item.get("features") or {}),
    }

    scenario_values = {
        **get_default_features(),
        **(item.get("scenario") or {}),
    }

    return templates.TemplateResponse(
        "simulation.html",
        {
            "request": request,
            "title": "Replay Simulation",
            "baseline_values": baseline_values,
            "scenario_values": scenario_values,
            "baseline_result": None,
            "scenario_result": None,
            "comparison": None,
            "sensitivity_results": None,
            "sensitivity_summary": None,
            "recommendations": None,
            "mini_trend_data": None,
            "error": None,
            "replay_timestamp": timestamp,
            "input_features": build_input_features(),
            "quick_actions": build_quick_actions(),
            "forced_feature_names": FORCED_PRODUCTION_FEATURES,
        },
    )


@app.get("/latest-predictions", response_class=HTMLResponse)
def latest_predictions(
    request: Request,
    target: Optional[str] = None,
    source: Optional[str] = None,
    time_range: str = "1h",
) -> HTMLResponse:
    time_filter_map = {
        "10min": "10 minutes",
        "1h": "1 hour",
        "all": None,
    }

    selected_interval = time_filter_map.get(time_range)

    query = """
    SELECT
        id,
        created_at,
        production_order,
        board_id,
        target_name,
        prediction_value,
        model_name,
        algorithm_name,
        request_source,
        status,
        input_features AS features,
        missing_features
    FROM prediction_log
    WHERE 1=1
    """

    params: Dict[str, Any] = {}

    if target:
        query += " AND target_name = :target"
        params["target"] = target

    if source:
        query += " AND request_source = :source"
        params["source"] = source

    if time_range in {"shift1", "shift2", "shift3"}:
        shift_start, shift_end = get_shift_window(time_range)

        query += """
        AND created_at >= :shift_start
        AND created_at < :shift_end
        """

        params["shift_start"] = shift_start
        params["shift_end"] = shift_end

    elif selected_interval:
        query += """
        AND created_at >= NOW() - CAST(:interval_value AS INTERVAL)
        """

        params["interval_value"] = selected_interval

    query += """
    ORDER BY created_at DESC
    LIMIT 100;
    """

    with engine.connect() as conn:
        result = conn.execute(text(query), params)
        rows = [dict(row._mapping) for row in result]

    rows = enrich_latest_rows(rows)
    kpi = build_latest_kpi(rows)

    return templates.TemplateResponse(
        "latest_predictions.html",
        {
            "request": request,
            "title": "Latest Predictions",
            "rows": rows,
            "kpi": kpi,
            "selected_target": target or "",
            "selected_source": source or "",
            "selected_time_range": time_range,
            "target_options": get_target_options(),
            "refresh_seconds": GUI_REFRESH_SECONDS,
        },
    )


@app.get("/monitoring-dashboard", response_class=HTMLResponse)
def monitoring_dashboard(request: Request) -> HTMLResponse:
    monitoring_service_healthy = False
    dashboard: Dict[str, Any] = {}
    error: Optional[str] = None

    try:
        monitoring_health_check()
        monitoring_service_healthy = True
        dashboard = get_monitoring_dashboard()

    except Exception as exc:
        error = str(exc)
        print("GUI monitoring dashboard error:", exc)

    return templates.TemplateResponse(
        "monitoring_dashboard.html",
        {
            "request": request,
            "title": "MDF1 MOM Monitoring Dashboard",
            "monitoring_service_healthy": monitoring_service_healthy,
            "system_overview": dashboard.get("system_overview", {}),
            "prediction_stats": dashboard.get("prediction_stats", {}),
            "feedback_stats": dashboard.get("feedback_stats", {}),
            "model_performance": dashboard.get("model_performance", {}),
            "model_performance_by_target": dashboard.get(
                "model_performance_by_target",
                {},
            ),
            "retraining_advisor": dashboard.get("retraining_advisor", {}),
            "drift_status": dashboard.get("drift_status", {}),
            "error": error,
        },
    )

@app.post("/monitoring-snapshots/create")
def create_snapshot_from_gui():
    try:
        create_monitoring_snapshot()

    except Exception as exc:
        print("Create monitoring snapshot failed:", exc)

    return RedirectResponse(
        url="/monitoring-snapshots",
        status_code=303,
    )

@app.get(
    "/monitoring-snapshots",
    response_class=HTMLResponse,
)
def monitoring_snapshots(
    request: Request,
) -> HTMLResponse:

    snapshots: Dict[str, Any] = {}
    error: Optional[str] = None

    try:
        snapshots = get_monitoring_snapshots()

    except Exception as exc:
        error = str(exc)

    return templates.TemplateResponse(
        "monitoring_snapshots.html",
        {
            "request": request,
            "title": "Monitoring Snapshot History",
            "snapshots": snapshots,
            "error": error,
        },
    )