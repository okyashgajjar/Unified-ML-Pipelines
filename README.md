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

-   **Upload CSV**: Bring your own dataset.
-   **Automated Preprocessing**: Mathematical-aware preprocessing for different model types.
-   **Model Training**: Trains Ridge, Lasso, Random Forest, XGBoost, MLP, KNN, and more.
-   **Results Analysis**: Interactive charts, metric comparison, and error analysis.

## Local Installation

1.  Clone the repository:
    ```bash
    git clone https://huggingface.co/spaces/YOUR_USERNAME/ml-pipelines
    cd ml-pipelines
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

<<<<<<< HEAD
3.  Run the app:
    ```bash
    streamlit run hf_app.py
    ```
=======
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
- **[summary_v1.1.md](summary_v1.1.md)** - v1.1 improvements summary
  
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
>>>>>>> df827a5857afc019caed92f10e33c97783ab30d8
