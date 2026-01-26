# ML Pipelines - Master Documentation v1.1

## 📦 Project Overview

**Unified ML Pipelines** is a mathematics-driven parallel machine learning system for regression tasks. It trains multiple model families with optimized preprocessing pipelines and provides a complete web interface.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 STREAMLIT FRONTEND                      │
│                 (streamlit_app.py)                      │
│  📊 Train │ 📈 Results │ 📋 History │ 🏠 Home           |
└─────────────────────┬───────────────────────────────────┘
                      │ HTTP Requests
                      ▼
┌─────────────────────────────────────────────────────────┐
│                   FASTAPI BACKEND                       │
│                     (app.py)                            │
│  POST /api/train │ GET /api/results │ GET /api/jobs     │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              MODEL TRAINING ENGINE                      │
│                                                         │
│  ┌─────────────┬─────────────┬─────────────┬─────────┐  │
│  │ Weight-Based│ Tree-Based  │ Neural Net  │Instance │  │
│  │ Linear,Ridge│ RF,XGBoost  │ MLP         │ KNN     │  │
│  │ Lasso       │ GBM,DT      │ Regressor   │ Radius  │  │
│  └─────────────┴─────────────┴─────────────┴─────────┘  │
│                Sequential Execution                     │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│               MLFLOW TRACKING                           │
│  Experiments │ Metrics │ Models │ Parameters            │
└─────────────────────────────────────────────────────────┘
```

**Alternative Deployment (Hugging Face Spaces):**
The `hf_app.py` merges Frontend and Backend into a single file for monolithic deployment.


---

## 📁 File Structure

```
ml-pipelines/
ml-pipelines/
├── app.py                    # FastAPI backend (REST API)
├── streamlit_app.py          # Streamlit frontend (Web UI)
├── hf_app.py                 # Unified app for Hugging Face Spaces (Backend + Frontend)
├── requirements.txt          # Python dependencies
├── README.md                 # Hugging Face Spaces configuration
│
├── dataset/
│   └── Housing.csv          # Sample dataset
│
├── Superwised_Regression/
│   ├── preprocessing.py     # Data cleaning & validation
│   └── tabular_data/
│       ├── weight_reg.py        # Linear, Ridge, Lasso
│       ├── tree_reg.py          # DT, RF, XGBoost, GBM
│       ├── nn_reg.py            # MLP Neural Network
│       ├── instance_reg.py      # KNN, Radius Neighbors
│       ├── parallel_executor.py # Sequential execution
│       └── mlflow_tracker.py    # MLFlow integration
│
└── mlruns/                  # MLFlow experiment data
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd /home/yash-test/Desktop/ml-pipelines
pip install -r requirements.txt
```

### 2. Start Backend
```bash
uvicorn app:app --reload
# Running on http://localhost:8000
```

### 3. Start Frontend
```bash
streamlit run streamlit_app.py
# Running on http://localhost:8501
```

### 4. (Optional) Run Unified App (Hugging Face Mode)
```bash
streamlit run hf_app.py
# Running on http://localhost:8501
```

### 5. (Optional) MLFlow UI
```bash
mlflow ui
# Running on http://localhost:5000
```

---

## 🎯 Model Families

| Family | Models | Preprocessing | Use Case |
|--------|--------|---------------|----------|
| **Weight-Based** | Linear, Ridge, Lasso | StandardScaler + OneHotEncoder | Linear relationships |
| **Tree-Based** | DT, RF, XGBoost, GBM | No scaling + OrdinalEncoder | Non-linear patterns |
| **Neural Network** | MLP Regressor | StandardScaler + OneHotEncoder | Deep patterns |
| **Instance-Based** | KNN, Radius Neighbors | StandardScaler (numeric only) | Local patterns |

---

## 🔌 API Reference

### Health Check
```
GET /api/health
Response: { "status": "healthy", "version": "1.1.0" }
```

### Train Models
```
POST /api/train
Form Data:
  - file: CSV file (required)
  - target_column: string (required)
  - use_parallel: "true"/"false"
  - enable_mlflow: "true"/"false"

Response: { "job_id": "uuid", "status": "queued" }
```

### Get Results
```
GET /api/results/{job_id}
Response: {
  "job_id": "uuid",
  "status": "completed",
  "results": [...]
}
```

### List Jobs
```
GET /api/jobs
Response: { "total": 5, "jobs": [...] }
```

---

## 🖥️ Streamlit UI Guide

### 📊 Train Models Page
1. Upload CSV file
2. Preview data (rows, columns)
3. Select target column
4. Click "Start Training"
5. Copy Job ID

### 📈 View Results Page
1. Enter Job ID
2. Click "Auto-Refresh" to monitor
3. View: Top 3 Models, Charts, Detailed Results, Error Analysis

### 📋 Job History
- List all training jobs
- View job details
- Quick access to results

---

## 📊 Metrics Explained

| Metric | Interpretation |
|--------|----------------|
| **MAE** | Average error magnitude (lower = better) |
| **RMSE** | Penalizes large errors (lower = better) |
| **R²** | % variance explained (higher = better) |
| **MAPE** | Percentage error |

---

## 🔍 Error & Outlier Analysis

### RMSE vs MAE Ratio
- **Ratio ≈ 1.0**: No outliers present
- **Ratio > 1.2**: Moderate outlier sensitivity
- **Ratio > 1.5**: High outlier sensitivity

### Issue Detection
- High Outlier Sensitivity: >20% above ideal
- Negative R²: Model worse than baseline
- High MAPE: >50% percentage error

---

## ⚡ Performance Optimizations

| Optimization | Impact |
|-------------|--------|
| Reduced hyperparameter grids | ~80x faster |
| 3-fold CV (from 5-fold) | ~40% faster |
| Sequential execution | Stable & reliable |

**Expected Training Time:** ~5-8 minutes for 10K rows

---

## ✅ v1.1 Features Summary

| Feature | Status |
|---------|--------|
| Error Handling | ✅ |
| Optimized Training | ✅ |
| MLFlow Integration | ✅ |
| FastAPI Backend | ✅ |
| Streamlit Frontend | ✅ |
| Outlier Analysis | ✅ |
| Hyperparameters in Results | ✅ |

---

*Last Updated: 2026-01-25*
