# 🧠 Production-Safe Predictive Scrap Risk Platform (MDF1)

An enterprise-ready Machine Learning serving platform designed for **manufacturing environments (MES)** to predict production quality and scrap risk in real-time.

---

## 🎯 Project Overview

This project implements a **production-grade ML inference service** for MDF manufacturing lines, enabling:

* Real-time prediction of key quality metrics (e.g., density)
* Integration with MES systems
* Scalable, containerized deployment
* Reliable and explainable AI-driven decision support

---

## 🏗️ Architecture

```
Client (Swagger / Java MOM / MES)
            ↓
     FastAPI Service (PredictionService)
            ↓
        Model Loader
            ↓
   Trained ML Model (Random Forest)
            ↓
        Prediction Output
```

---

## ⚙️ Core Features

### ✅ ML Model Serving

* Loads trained models from `/artifacts`
* Supports multiple model versions
* Feature validation before prediction

### ✅ REST API (FastAPI)

* Fully documented with Swagger UI
* Clean request/response structure
* Designed for integration with Java MOM systems

### ✅ Dockerized Deployment

* Portable and production-ready
* Consistent environment across machines
* Easy scaling and integration

### ✅ Robust Error Handling

* Missing model detection
* Missing feature reporting
* Structured API responses

---

## 📦 Project Structure

```
PredictionService/
├── src/
│   ├── main.py                # FastAPI entry point
│   ├── api/
│   │   └── routes.py          # API endpoints
│   ├── service/
│   │   ├── model_loader.py    # Model loading logic
│   │   └── prediction_service.py
│   ├── schemas/
│   │   ├── request_schema.py
│   │   └── response_schema.py
│   └── config/
│       └── settings.py
│
├── artifacts/
│   └── labDensityAverage_rf_v1/
│       ├── model.joblib
│       └── model_features.json
│
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

---

## 🔌 API Endpoints

### 🔹 Health Check

```
GET /health
```

Response:

```json
{
  "status": "ok"
}
```

---

### 🔹 List Available Models

```
GET /models
```

Response:

```json
{
  "models": ["labDensityAverage_rf_v1"]
}
```

---

### 🔹 Predict

```
POST /predict
```

Request:

```json
{
  "model_name": "labDensityAverage_rf_v1",
  "features": {
    "rawThickness": 18.2,
    "pressPressureMid_mean": 142.5,
    "beltSpeed1": 31.2
  }
}
```

Response:

```json
{
  "model_name": "labDensityAverage_rf_v1",
  "prediction": 743.8,
  "used_feature_count": 28,
  "missing_features": []
}
```

---

## 🐳 Running with Docker

### 1. Build image

```bash
docker build -t mdf1-prediction-service .
```

### 2. Run container

```bash
docker run -p 8000:8000 mdf1-prediction-service
```

### 3. Open Swagger UI

```
http://127.0.0.1:8000/docs
```

---

## 💻 Running Locally (Development)

```bash
uvicorn main:app --reload --app-dir src
```

---

## 🧠 Machine Learning Details

* Model: Random Forest Regressor
* Target: `labDensityAverage`

### Feature Engineering:

* Pressure statistics (mean, std, range)
* Thickness metrics
* Temperature aggregates

### Feature Selection:

* Based on correlation analysis

### Model Performance:

* R² ≈ 0.74
* MAE ≈ 21

---

## 🔄 Integration with MES / MOM

This service is designed to be integrated with:

* Java-based MOM systems
* MES platforms (LiraMES, KSoft, ProdIQ)
* PLC / historian data pipelines

### Typical Flow:

```
MES / Java Service
      ↓
Fetch production data
      ↓
Call /predict API
      ↓
Receive prediction
      ↓
Trigger alerts / decisions
```

---

## ⚠️ Known Limitations (Current Version)

* Manual feature input required (Swagger/testing)
* No direct DB/MES integration yet
* No real-time streaming pipeline

---

## 🚀 Future Improvements

* 🔹 Auto-fetch features from MES / database
* 🔹 Docker Compose (ML + Java + DB)
* 🔹 Model versioning & registry
* 🔹 Prediction logging & monitoring
* 🔹 Drift detection
* 🔹 Scrap risk classification layer
* 🔹 High availability deployment

---

## 🧩 Use Case

This platform enables:

* Early detection of production issues
* Reduction of scrap rate
* Data-driven decision making
* AI integration into industrial systems

---

## 👤 Author

**edvazat**

---

## 📌 Summary

This project demonstrates how to:

* Serve ML models in production
* Integrate AI into manufacturing systems
* Build scalable, maintainable ML microservices
