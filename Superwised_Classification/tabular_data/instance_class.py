import numpy as np
import pandas as pd
import sklearn
# Instance based
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# Evaluation Metrics
from sklearn.metrics import accuracy_score, precision_score, classification_report, roc_auc_score
# PCA
from sklearn.decomposition import PCA
# Scaling
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
# Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# Hyperparameter tuning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import warnings
warnings.filterwarnings("ignore")

# df = '/home/yash-test/Desktop/ml-pipelines/dataset/mushrooms.csv'

def instance_based(df, target) : 
  try:
    # Input validation
    if not isinstance(df, pd.DataFrame):
      raise TypeError("Input must be a pandas DataFrame")
    if target not in df.columns:
      raise ValueError(f"Target column '{target}' not found in dataframe")
    if df.empty:
      raise ValueError("DataFrame is empty")

    # df = pd.read_csv(df)

    x = df.drop(columns=[target])
    y = df[target]

    # now remove categorical features from X
    X = x.select_dtypes(exclude=['object', 'category'])

    if X.shape[1] == 0:
      raise ValueError("Instance Learns from only numerical columns & No numerical features found in the input")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    num_features = X.columns

    # pca-pipeline
    pca_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95))
    ])

    # make a preprocessor
    if len(num_features) >= 10 :
      preprocessor = ColumnTransformer([
          ('num', StandardScaler(), num_features)
      ])
      pca_preprocessor = ColumnTransformer([
          ('num', pca_pipeline, num_features)
      ])
    else:
      preprocessor = ColumnTransformer([
          ('num', StandardScaler(), num_features)
      ])
      pca_preprocessor = preprocessor


    # models 
    models = {
        'KNN' : KNeighborsClassifier(n_neighbors=5)
    }

    all_results = []

    for model_name, model in models.items():
      try:
        if model_name == 'KNN':
          param_grid = {
            'model__n_neighbors': [3, 5, 7, 9],
            'model__weights': ['uniform', 'distance'],
            'model__metric': ['euclidean', 'manhattan']
          }

        if len(num_features) >= 10 :
          pipeline = Pipeline([
              ('preprocessor', pca_preprocessor),
              ('model', model)
          ])
        else :
          pipeline = Pipeline([
              ('preprocessor', preprocessor),
              ('model', model)
          ])

        grid_search = GridSearchCV(
          estimator=pipeline,
          param_grid=param_grid,
          cv=3,
          scoring='accuracy',
          n_jobs=-1
        )

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        y_pred = best_model.predict(X_test)
        acc_score = accuracy_score(y_test, y_pred)
        prec_score = precision_score(y_test, y_pred, average='weighted')
        class_report = classification_report(y_test, y_pred)
        
        all_results.append({
            'model_name': model_name,
            'best_params': best_params,
            'pipeline': pipeline,
            'accuracy': acc_score,
            'precision': prec_score,
            'status': 'success',
            'error': None
        })

      except Exception as model_error:
        all_results.append({
            'model_name': model_name,
            'best_params': None,
            'pipeline': None,
            'accuracy': None,
            'precision': None,
            'status': 'failed',
            'error': str(model_error)
        })

    return pd.DataFrame(all_results)

  except Exception as e:
    return pd.DataFrame([{
        'model_name': 'Instance-Based Classifiers',
        'best_params': None,
        'pipeline': None,
        'accuracy': None,
        'precision': None,
        'status': 'failed',
        'error': f"Instance-based classifier family failed: {str(e)}"
    }])

# instance_based(df, 'class')
