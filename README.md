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

# Unified ML Pipelines

**Mathematics-Driven Parallel Machine Learning Pipelines**

This application allows you to train multiple families of Machine Learning models (Weight-based, Tree-based, Neural Networks, and Instance-based) on your tabular data simultaneously.

## Features

- ✅ **Upload CSV**: Bring your own dataset
- ✅ **Automated Preprocessing**: Mathematical-aware preprocessing for different model types
- ✅ **Model Training**: Trains Ridge, Lasso, Random Forest, XGBoost, MLP, KNN, and more
- ✅ **Results Analysis**: Interactive charts, metric comparison, and error analysis
- ✅ **Error & Outlier Detection**: Automated issue detection with suggestions

## 🚀 Quick Start

### Option 1: Hugging Face Spaces (Recommended)
Simply visit the deployed Space and upload your CSV file!

### Option 2: Local Installation
```bash
# Clone the repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/ml-pipelines
cd ml-pipelines

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

## 📚 Project Structure

```
ml-pipelines/
├── hf_app.py                       # Unified app for HF Spaces
├── app.py                          # FastAPI backend
├── streamlit_app.py                # Streamlit frontend
├── requirements.txt                # Dependencies
├── Superwised_Regression/
│   ├── preprocessing.py            # Data cleaning
│   └── tabular_data/
│       ├── weight_reg.py           # Weight-based models
│       ├── tree_reg.py             # Tree-based models
│       ├── nn_reg.py               # Neural network models
│       ├── instance_reg.py         # Instance-based models
│       ├── parallel_executor.py    # Parallel execution
│       └── mlflow_tracker.py       # MLFlow integration
└── MASTER_DOCUMENTATION.md         # Full documentation
```

## 🎯 Model Families

| Family | Models | Preprocessing |
|--------|--------|---------------|
| **Weight-Based** | Linear, Ridge, Lasso | StandardScaler + OneHotEncoder |
| **Tree-Based** | DT, RF, XGBoost, GBM | No scaling + OrdinalEncoder |
| **Neural Network** | MLP Regressor | StandardScaler + OneHotEncoder |
| **Instance-Based** | KNN, Radius Neighbors | StandardScaler (categorical dropped) |

## 📊 Metrics & Analysis

| Metric | Interpretation |
|--------|----------------|
| **MAE** | Average error magnitude (lower = better) |
| **RMSE** | Penalizes large errors (lower = better) |
| **R²** | % variance explained (higher = better) |
| **MAPE** | Percentage error |

### Error & Outlier Analysis
- **RMSE vs MAE Ratio**: Detects outlier sensitivity
- **Negative R²**: Identifies models worse than baseline
- **High MAPE**: Flags issues with small target values

## � License

MIT License - feel free to use and modify.

---

Built with ❤️ focusing on **mathematical correctness** and **educational value**.
