# Unified ML Pipelines - Mathematics-Driven Regression System

> **Version 1.1** - Production-ready ML pipeline system with parallel execution, MLFlow tracking, FastAPI backend, and Streamlit frontend.

## 🚀 Quick Start

### Installation
```bash
# Clone or navigate to project
cd /home/yash-test/Desktop/ml-pipelines

# Install dependencies
pip install -r requirements.txt
```

### Running the System

**Option 1: Full Stack (Recommended)**
```bash
# Terminal 1: FastAPI Backend
uvicorn app:app --reload
# Access API docs: http://localhost:8000/docs

# Terminal 2: Streamlit Frontend
streamlit run streamlit_app.py
# Access UI: http://localhost:8501

# Terminal 3 (Optional): MLFlow UI
mlflow ui
# Access MLFlow: http://localhost:5000
```

**Option 2: API Only**
```bash
# Start FastAPI backend
uvicorn app:app --reload

# Use curl or Postman to interact with API
curl -X POST "http://localhost:8000/api/train" \
  -F "file=@dataset/Housing.csv" \
  -F "target_column=price"
```

## 📚 Project Structure

```
ml-pipelines/
├── app.py                          # FastAPI backend
├── streamlit_app.py                # Streamlit frontend
├── requirements.txt                # Dependencies
├── dataset/
│   └── Housing.csv                 # Sample dataset
├── Superwised_Regression/
│   ├── preprocessing.py            # Data cleaning
│   └── tabular_data/
│       ├── weight_reg.py           # Weight-based models
│       ├── tree_reg.py             # Tree-based models
│       ├── nn_reg.py               # Neural network models
│       ├── instance_reg.py         # Instance-based models
│       ├── parallel_executor.py    # Parallel execution
│       └── mlflow_tracker.py       # MLFlow integration
└── PROJECT_SUMMARY.md              # Project philosophy
```

## 🎯 Features

### Core Capabilities
- ✅ **4 Model Families**: Weight-based, Tree-based, Neural Networks, Instance-based
- ✅ **Parallel Execution**: 2x faster training with ProcessPoolExecutor
- ✅ **Error Handling**: Comprehensive validation at all levels
- ✅ **MLFlow Tracking**: Experiment tracking and model registry
- ✅ **REST API**: 7 endpoints for complete control
- ✅ **Interactive UI**: Streamlit frontend with real-time monitoring

### Model Families

| Family | Models | Preprocessing | Hyperparameter Search |
|--------|--------|---------------|----------------------|
| **Weight-Based** | Linear, Ridge, Lasso | StandardScaler + OneHotEncoder | GridSearchCV |
| **Tree-Based** | DT, RF, XGBoost, GBM | No scaling + OrdinalEncoder | RandomizedSearchCV |
| **Neural Network** | MLP Regressor | StandardScaler + OneHotEncoder | GridSearchCV |
| **Instance-Based** | KNN, Radius Neighbors | StandardScaler (categorical dropped) | GridSearchCV |

## 📊 Usage Examples

### Using Streamlit UI

1. **Upload Dataset**
   - Navigate to "📊 Train Models"
   - Upload CSV file
   - Preview data

2. **Configure & Train**
   - Select target column
   - Enable parallel execution
   - Click "🚀 Start Training"

3. **View Results**
   - Navigate to "📈 View Results"
   - Enter job ID
   - Click "🔄 Auto-Refresh"
   - View interactive charts

### Using API

```python
import requests

# Upload and train
with open("dataset/Housing.csv", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/train",
        files={"file": f},
        data={"target_column": "price", "use_parallel": True}
    )

job_id = response.json()["job_id"]

# Get results
result = requests.get(f"http://localhost:8000/api/results/{job_id}")
print(result.json())
```

## 📈 Performance

- **Sequential Execution**: ~11 minutes
- **Parallel Execution**: ~5 minutes
- **Speedup**: ~2x faster

## 🔧 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/api/health` | API status |
| `POST` | `/api/train` | Upload CSV and train models |
| `GET` | `/api/results/{job_id}` | Get training results |
| `GET` | `/api/jobs` | List all jobs |
| `GET` | `/api/experiments` | List MLFlow experiments |
| `DELETE` | `/api/jobs/{job_id}` | Delete a job |

## 📖 Documentation

- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Project philosophy and approach
- **[understanding_v1.md](understanding_v1.md)** - Model implementation analysis
- **[summary_v1.1.md](summary_v1.1.md)** - v1.1 improvements summary
- **[walkthrough.md](walkthrough.md)** - Quick start walkthrough

## 🧪 Testing

```bash
# Test API health
curl http://localhost:8000/api/health

# Test training (with sample dataset)
curl -X POST "http://localhost:8000/api/train" \
  -F "file=@dataset/Housing.csv" \
  -F "target_column=price"
```

## 🛠️ Technology Stack

- **Backend**: FastAPI, Uvicorn
- **Frontend**: Streamlit, Plotly
- **ML Libraries**: scikit-learn, XGBoost, LightGBM
- **Tracking**: MLFlow
- **Data**: Pandas, NumPy
- **Validation**: Pydantic

## 📝 Version History

### v1.1 (Current)
- ✅ Error handling and validation
- ✅ Parallel execution (O(n) time)
- ✅ MLFlow integration
- ✅ FastAPI backend
- ✅ Streamlit frontend

### v1.0
- ✅ 4 model families
- ✅ Family-specific preprocessing
- ✅ Multi-metric evaluation
- ✅ CLI interface

## 🤝 Contributing

This is an educational project demonstrating mathematics-driven ML pipelines.

## 📄 License

Educational project - feel free to use and modify.

## 🙏 Acknowledgments

Built with a focus on **mathematical correctness** and **educational value**.

---