---
title: Unified ML Pipelines
emoji: 🤖
colorFrom: yellow
colorTo: orange
sdk: streamlit
sdk_version: 1.31.0
app_file: hf_app.py
pinned: false
license: mit
---

> **💡 Personal Suggestions:**
> 1. If you have a better CPU or a GPU in your device, you must fork this repository and use it in your personal computer to get faster results.
> 2. Make sure you preprocess your data (handle missing values, datatypes, outliers etc.) to get more accuracy on our models.

# Unified ML Pipelines

**Mathematics-Driven Parallel Machine Learning Pipelines for Regression & Classification**

This application allows you to train multiple families of Machine Learning models on your tabular data simultaneously, with **mathematically-correct preprocessing** tailored for each model family.

---

## ✨ Features

### Core Capabilities
- ✅ **Dual Learning Types**: Support for both **Regression** and **Classification** tasks
- ✅ **Upload CSV**: Bring your own dataset
- ✅ **Automated Preprocessing**: Mathematical-aware preprocessing for different model types
- ✅ **5 Model Families**: Weight-based, Tree-based, Neural Network, Instance-based, and Kernel-based (classification)
- ✅ **14+ ML Models**: Comprehensive model coverage across all families
- ✅ **Hyperparameter Tuning**: GridSearchCV and RandomizedSearchCV with optimized grids
- ✅ **MLFlow Integration**: Full experiment tracking, metrics logging, and model versioning

### Results & Analysis
- ✅ **Interactive Visualizations**: Plotly charts for model comparison
- ✅ **Top 3 Models Display**: Quick identification of best performers
- ✅ **Detailed Metrics Tables**: Complete breakdown of all model results
- ✅ **Error & Outlier Detection**: Automated issue detection with actionable suggestions
- ✅ **Hyperparameters Display**: View tuned hyperparameters for each model

### Deployment Options
- ✅ **Streamlit Cloud**: One-click deployment
- ✅ **Hugging Face Spaces**: Unified app deployment
- ✅ **Local Development**: FastAPI + Streamlit separation
- ✅ **MLFlow UI**: Experiment tracking dashboard

---

## 🚀 Quick Start

### Option 1: Use Streamlit Deployed App (Recommended)
Simply visit the deployed Space and upload your CSV file!
```
https://unified-ml-pipelines.streamlit.app/
```

### Option 2: Local Installation
```bash 
# Clone the repository
git clone https://github.com/okyashgajjar/Unified-ML-Pipelines.git
cd Unified-ML-Pipelines

# Install dependencies
pip install -r requirements.txt

# Run the unified app
streamlit run hf_app.py
```

### Option 3: API + Frontend (Development)
```bash
# Terminal 1: Start FastAPI backend
uvicorn app:app --reload

# Terminal 2: Start Streamlit frontend
streamlit run streamlit_app.py
```

### Option 4: MLFlow Tracking UI
```bash
mlflow ui
# Running on http://localhost:5000
```

---

## 📚 Project Structure

```
Unified-ML-Pipelines/
├── hf_app.py                       # Unified app for HF Spaces (Frontend + Backend)
├── app.py                          # FastAPI backend (REST API)
├── streamlit_app.py                # Streamlit frontend (Web UI)
├── requirements.txt                # Dependencies
│
├── Superwised_Regression/
│   ├── preprocessing.py            # Data cleaning & validation
│   └── tabular_data/
│       ├── weight_reg.py           # Linear, Ridge, Lasso
│       ├── tree_reg.py             # DT, RF, XGBoost, GBM
│       ├── nn_reg.py               # MLP Regressor
│       ├── instance_reg.py         # KNN, Radius Neighbors
│       ├── parallel_executor.py    # Sequential execution
│       └── mlflow_tracker.py       # MLFlow integration
│
├── Superwised_Classification/
│   └── tabular_data/
│       ├── weight_class.py         # Logistic Regression, Ridge Classifier
│       ├── tree_class.py           # DT, RF, GBM, AdaBoost, LightGBM, XGBoost
│       ├── nn_class.py             # MLP Classifier
│       ├── kernel_class.py         # SVC (RBF, Linear, Poly)
│       └── instance_class.py       # KNN Classifier
│
├── dataset/                        # Sample datasets
├── mlruns/                         # MLFlow experiment data
├── MASTER_DOCUMENTATION.md         # Full technical documentation
└── PROJECT_SUMMARY.md              # Project philosophy & approach
```

---

## 🎯 Regression Model Families

| Family | Models |
|--------|--------|
| **Weight-Based** | Linear, Ridge, Lasso |
| **Tree-Based** | DT, RF, XGBoost, GBM |
| **Neural Network** | MLP Regressor |
| **Instance-Based** | KNN, Radius Neighbors |

---

## 🎯 Classification Model Families

| Family | Models |
|--------|--------|
| **Weight-Based** | Logistic Regression, Ridge Classifier |
| **Tree-Based** | DT, RF, GBM, AdaBoost, LightGBM, XGBoost |
| **Neural Network** | MLP Classifier |
| **Kernel-Based** | SVC (RBF, Linear, Poly) |
| **Instance-Based** | KNN Classifier |

---

## 📊 Metrics & Analysis

### Regression Metrics
| Metric | Interpretation |
|--------|----------------|
| **MAE** | Average error magnitude (lower = better) |
| **RMSE** | Penalizes large errors (lower = better) |
| **R²** | % variance explained (higher = better) |
| **MAPE** | Percentage error (lower = better) |
| **MSE** | Squared error (lower = better) |

### Classification Metrics
| Metric | Interpretation |
|--------|----------------|
| **Accuracy** | Overall correct predictions (higher = better) |
| **Precision** | True positives / Predicted positives (higher = better) |
| **Recall** | True positives / Actual positives (higher = better) |
| **F1 Score** | Harmonic mean of precision & recall (higher = better) |

### Error & Outlier Analysis (Regression)
- **RMSE vs MAE Ratio**: Detects outlier sensitivity (>1.2 indicates outliers)
- **Negative R²**: Identifies models worse than baseline mean
- **High MAPE**: Flags issues with small target values (>50%)

---

## 🔌 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check with version info |
| `/api/train` | POST | Submit training job with CSV file |
| `/api/train-classification` | POST | Submit classification training job |
| `/api/results/{job_id}` | GET | Get training results |
| `/api/jobs` | GET | List all training jobs |

---

## ⚡ Performance

- **Optimized Hyperparameter Grids**: ~80% reduction in search space
- **3-Fold Cross-Validation**: Faster than 5-fold with minimal accuracy loss
- **Sequential Execution**: Stable and reliable on all hardware
- **Expected Training Time**: ~5-8 minutes for 10K rows

---

## 🎨 UI Features

### 🏠 Home Page
- Project overview and quick start guide

### 📊 Train Models Page
1. Upload CSV file
2. Preview data with row/column counts
3. **Select Learning Type**: Regression or Classification
4. Choose target column
5. Enable/disable MLFlow tracking
6. Start training with real-time progress

### 📈 Results Page
- **Top 3 Models**: Side-by-side comparison cards
- **Interactive Charts**: Bar charts, heatmaps, scatter plots
- **Detailed Results Table**: Sortable with all metrics
- **Error Analysis**: Automated issue detection (Regression)

### 📋 Job History
- View all past training jobs
- Quick access to results
- Job status tracking

---

## 📄 License

MIT License - feel free to use and modify.

---

## 🔗 Links

- **Live Demo**: [unified-ml-pipelines.streamlit.app](https://unified-ml-pipelines.streamlit.app/)
- **GitHub**: [github.com/okyashgajjar/Unified-ML-Pipelines](https://github.com/okyashgajjar/Unified-ML-Pipelines)
- **Documentation**: See `MASTER_DOCUMENTATION.md` for full technical details

---

Built with ❤️ focusing on **mathematical correctness** and **educational value**.

*Version 1.2 | Last Updated: January 2026*
