# MDF1 Industrial AI / MOM Monitoring Platform

## Project Overview

This project is an end-to-end industrial AI platform designed for MDF manufacturing quality prediction, MES integration, operator feedback, quality result validation, and model monitoring.

The platform simulates a production-ready Manufacturing Operations Management (MOM) / MES-connected AI workflow where machine learning predictions are generated, explained, logged, validated against laboratory results, and monitored through a GUI dashboard.

## Architecture

```text
TrainingService
    ↓
PredictionService
    ↓
MESIntegrationService
    ↓
mes_prediction_logs
    ↓
QualityResultService
    ↓
quality_results
    ↓
MonitoringService
    ↓
GUIService Monitoring Dashboard
```

## Main Services

### 1. TrainingService

Responsible for preparing production data, training machine learning models, evaluating model performance, and generating versioned model artifacts.

Key responsibilities:

- Data preprocessing and validation
- ML model training
- Random split and time-based validation support
- Feature governance
- Model metrics and metadata generation
- Artifact preparation for PredictionService

### 2. PredictionService

Responsible for serving trained models through FastAPI.

Key responsibilities:

- Registry-driven model loading
- Feature order validation
- Single-model and ensemble prediction support
- SHAP explainability
- Risk level calculation
- Recommendation generation
- Prediction response formatting

### 3. MESIntegrationService

Acts as the bridge between MES-style production requests and the PredictionService.

Key responsibilities:

- Accept MES prediction requests
- Map MES feature names to ML feature names
- Call PredictionService
- Return prediction, risk level, SHAP details, and recommendations
- Save MES prediction logs
- Provide MES prediction history

### 4. OperatorFeedbackService

Captures operator decisions after AI recommendations.

Key responsibilities:

- Store operator feedback
- Track accepted/rejected recommendations
- Support feedback analytics
- Provide recommendation acceptance rate for monitoring

### 5. QualityResultService

Connects actual laboratory quality results to previous AI predictions.

Key responsibilities:

- Receive actual lab results
- Automatically match quality results with MES prediction logs
- Reject unmatched quality results
- Prevent duplicate quality results using application-level checks and PostgreSQL constraints
- Calculate absolute error and percentage error
- Maintain traceability using foreign keys

### 6. MonitoringService

Provides operational monitoring for the AI platform.

Key responsibilities:

- System overview metrics
- Prediction statistics
- Risk distribution
- Operator feedback distribution
- Recommendation acceptance rate
- Model performance monitoring
- MAE and MAPE calculation
- Per-target performance monitoring
- Drift detection
- Retraining advisor
- Monitoring snapshot history
- Consolidated dashboard API endpoint

### 7. GUIService

Provides a web-based operator and engineering interface.

Key pages:

- Home dashboard
- Manual prediction page
- What-if simulation page
- Latest predictions page
- Prediction history page
- MOM monitoring dashboard

The GUI monitoring dashboard connects to MonitoringService through a single consolidated endpoint:

```text
GET /api/v1/monitoring/dashboard
```

## Database Tables

Main PostgreSQL tables used in the platform:

```text
prediction_log
mes_prediction_logs
operator_feedback
quality_results
monitoring_snapshots
```

## Monitoring KPIs

The Monitoring Dashboard displays:

- Total MES predictions
- Total quality result records
- Prediction coverage percentage
- Operator feedback count
- Recommendation acceptance rate
- Risk distribution
- Feedback distribution
- MAE
- MAPE
- Model performance by target
- Drift status
- Retraining advisor status

## Current Dashboard Example

Example dashboard values from the current implementation:

```text
MES Predictions: 14
Quality Results: 4
Operator Feedback: 7
Acceptance Rate: 100%
MAE: 31.9899
MAPE: 4.51%
Risk Distribution: MEDIUM = 14
Retraining Status: NOT_ENOUGH_DATA
Drift Status: NOT_ENOUGH_DATA
```

`NOT_ENOUGH_DATA` is expected when the number of quality result records is still below the configured threshold for drift and retraining decisions.

## Technical Stack

- Python
- FastAPI
- SQLAlchemy
- PostgreSQL
- Pydantic
- Requests
- Jinja2 Templates
- SHAP explainability
- Scikit-learn model artifacts
- Modular microservice-style architecture

## Key Engineering Features

- Microservice-based industrial AI architecture
- MES-to-ML feature mapping
- Prediction traceability
- Laboratory result feedback loop
- Operator feedback loop
- PostgreSQL constraints for data quality
- Foreign key relationship between predictions and quality results
- Monitoring snapshot persistence
- Consolidated dashboard API
- GUI dashboard integration
- Production-oriented logging and error handling

## Production Flow

```text
1. MES sends production data
2. MESIntegrationService maps MES features to ML features
3. PredictionService loads trained model and predicts target quality
4. Prediction result, risk level, SHAP explanation, and recommendations are returned
5. Prediction is logged in mes_prediction_logs
6. Operator feedback is stored in operator_feedback
7. Actual lab result is submitted to QualityResultService
8. QualityResultService links the lab result to the original prediction
9. Error metrics are calculated
10. MonitoringService evaluates model performance, drift, and retraining status
11. GUIService displays the full MOM monitoring dashboard
```

## Why This Project Matters

This project demonstrates how industrial AI can be integrated into a manufacturing environment beyond simple model prediction. It includes the full lifecycle needed for production-grade AI governance:

- Prediction
- Explainability
- MES integration
- Operator feedback
- Actual quality validation
- Performance monitoring
- Drift detection
- Retraining governance
- Dashboard visualization

## Future Improvements

Planned improvements include:

- Automatic monitoring snapshot scheduler
- Trend charts for MAE, MAPE, coverage, and drift
- GUI visualization of monitoring snapshot history
- More advanced drift detection
- Multi-line deployment support
- Role-based access control
- Docker Compose deployment
- CI/CD pipeline
- Integration with live MES / SCADA data sources

## Portfolio Summary

This project represents a practical industrial AI platform for MDF manufacturing. It combines machine learning, MES integration, model explainability, quality result validation, operator feedback, monitoring, and GUI visualization into a single end-to-end architecture.

It is suitable for demonstrating skills in:

- Industrial AI
- Manufacturing Execution Systems integration
- MOM architecture
- Machine learning operations
- FastAPI backend development
- PostgreSQL data modeling
- Monitoring and retraining governance
- Production-oriented software engineering
