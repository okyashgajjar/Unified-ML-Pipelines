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

3.  Run the app:
    ```bash
    streamlit run hf_app.py
    ```