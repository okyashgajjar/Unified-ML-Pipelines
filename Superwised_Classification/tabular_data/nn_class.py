import numpy as np
import pandas as pd
import sklearn
# neural network - multilayer perception
from sklearn.neural_network import MLPClassifier
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
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


# df = '/home/yash-test/Desktop/ml-pipelines/dataset/mushrooms.csv'


def neural_network(df, target) :
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

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    num_features = x.select_dtypes(exclude=['object']).columns
    cat_features = x.select_dtypes(include=['object']).columns

    # pipeline for pca
    pca_pipeline = Pipeline([
        ('scaling', StandardScaler()),
        ('pca', PCA(n_components=0.95))
    ])

    # preprocessors
    if (len(num_features) > 0 and len(cat_features) > 0) :
      preprocessor = ColumnTransformer([
          ('num', StandardScaler(), num_features),
          ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
      ])
      pca_preprocessor = ColumnTransformer([
          ('num', pca_pipeline, num_features),
          ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
      ])
    elif len(num_features) == 0 :
      preprocessor = ColumnTransformer([
          ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
      ])
      pca_preprocessor = ColumnTransformer([
          ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
      ])
    elif len(cat_features) == 0 :
      preprocessor = ColumnTransformer([
          ('num', StandardScaler(), num_features)
      ])
      pca_preprocessor = ColumnTransformer([
          ('num', pca_pipeline, num_features)
      ])
    
    # models
    models = {
        'MLP Classifier' : MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            max_iter=500
        )
    }

    all_results = []

    for model_name, model in models.items() : 
      try:
        if model_name == 'MLP Classifier' :
          param_grid = {
            'model__hidden_layer_sizes': [(100,), (100, 50), (50, 25)],
            'model__activation': ['relu', 'tanh'],
            'model__alpha': [0.0001, 0.001],
            'model__learning_rate': ['adaptive'],
            'model__max_iter': [500]
          }

        if len(num_features) >= 30 :
          pipeline = Pipeline([
              ('preprocessor', pca_preprocessor),
              ('model', model)
          ])
        else : 
          pipeline = Pipeline([
              ('preprocessor', preprocessor),
              ('model', model)
          ])
      
        random_search = RandomizedSearchCV(
          estimator=pipeline,
          param_distributions=param_grid,
          n_iter=10,
          cv=3,
          scoring='accuracy',
          n_jobs=-1,
          random_state=42
        )

        random_search.fit(X_train, y_train)

        best_model = random_search.best_estimator_
        best_params = random_search.best_params_

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
        'model_name': 'Neural Network Classifiers',
        'best_params': None,
        'pipeline': None,
        'accuracy': None,
        'precision': None,
        'status': 'failed',
        'error': f"Neural network classifier family failed: {str(e)}"
    }])
      
# neural_network(df, 'class')