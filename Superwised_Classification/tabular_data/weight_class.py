import numpy as np
import pandas as pd
import sklearn
# Weight Based
from sklearn.linear_model import LogisticRegression, RidgeClassifier
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
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# df = '/home/yash-test/Desktop/ml-pipelines/dataset/mushrooms.csv'

def weight_based(df, target) :
  try:
    # Input validation
    if not isinstance(df, pd.DataFrame):
      raise TypeError("Input must be a pandas DataFrame")
    if target not in df.columns:
      raise ValueError(f"Target column '{target}' not found in dataframe")
    if df.empty:
      raise ValueError("DataFrame is empty")

    # df = pd.read_csv(df)

    # Split Data into x & y
    x = df.drop(target, axis=1)
    y = df[target]

    num_features = x.select_dtypes(exclude=['object']).columns
    cat_features = x.select_dtypes(include=['object']).columns

    pca_num_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95))
    ])

    # Preprocessor
    if len(num_features) > 0 and len(cat_features) > 0:
      preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ]
      )
      pca_preprocessor = ColumnTransformer(
          transformers=[
          ('num', pca_num_pipeline, num_features),
          ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
      ])
    elif len(num_features) == 0 :
      preprocessor = ColumnTransformer(
          transformers=[
              ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
          ]
      )
      pca_preprocessor = ColumnTransformer(
          transformers=[
          ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
      ])
    elif len(cat_features) == 0:
      preprocessor = ColumnTransformer(
          transformers=[
              ('num', StandardScaler(), num_features)
          ]
      )
      pca_preprocessor = ColumnTransformer(
          transformers=[
          ('num', pca_num_pipeline, num_features)
      ])
    else :
      raise ValueError("No numerical or categorical features found in the input data.")

    # Training and testing
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2 ,random_state=42)

    # models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Ridge Classifier' : RidgeClassifier()
    }

    all_results = []

    # print(y.shape)
    # print(y_train.shape)
    # print(y_test.shape)

    for model_name, model in models.items() :
      try:
        if model_name == 'Logistic Regression' :
          param_grid = {
              'model__C': [0.1, 1.0, 10.0],
              'model__solver': ['lbfgs', 'saga']
          }
        elif model_name == 'Ridge Classifier' :
          param_grid = {
              'model__alpha': [0.1, 1.0, 10.0]
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

        grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        
        # FIRST fit, THEN access best_estimator_
        grid_search.fit(X_train, y_train)
        
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        
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
        'model_name': 'Weight-Based Classifiers',
        'best_params': None,
        'pipeline': None,
        'accuracy': None,
        'precision': None,
        'status': 'failed',
        'error': f"Weight-based classifier family failed: {str(e)}"
    }])

# weight_based(df, 'class')