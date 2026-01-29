import numpy as np
import pandas as pd
import sklearn
# kernel based
from sklearn.svm import SVC
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

def kernel_based(df, target) :
  try:
    # Input validation
    if not isinstance(df, pd.DataFrame):
      raise TypeError("Input must be a pandas DataFrame")
    if target not in df.columns:
      raise ValueError(f"Target column '{target}' not found in dataframe")
    if df.empty:
      raise ValueError("DataFrame is empty")

    #   df = pd.read_csv(df)

    # Split Data x, y
    x = df.drop(columns=[target])
    y = df[target]

    num_features = x.select_dtypes(exclude=['object']).columns
    cat_features = x.select_dtypes(include=['object']).columns

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # pca num_pipeline
    pca_num_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('PCA', PCA(n_components=0.95))
    ])

    # pipeline
    if( len(num_features) > 0 and len(cat_features) > 0) :
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
            ]
        )
        pca_preprocessor = ColumnTransformer(
            transformers=[
                ('num', pca_num_pipeline, num_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
            ]
        )
    elif len(num_features) == 0 :
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
            ]
        )
        pca_preprocessor = ColumnTransformer(
            transformers= [
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
            ]
        )
    elif len(cat_features) == 0:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_features)
            ]
        )
        pca_preprocessor = ColumnTransformer(
            transformers=[
                ('num', pca_num_pipeline, num_features)
            ]
        )


    # models
    models = {
          'SVC' : SVC()
    }

    all_results = []

    for model_name, model in models.items() :
      try:
        if model_name == 'SVC' :
            param_grid = {
              'model__C': [0.1, 1, 10],
              'model__kernel': ['rbf', 'linear', 'poly'],
              'model__gamma': ['scale', 'auto'],
              'model__class_weight': [None, 'balanced']
            }

        # separate pca & normal pipeline logic
        if len(num_features) >= 10:
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
        'model_name': 'Kernel-Based Classifiers',
        'best_params': None,
        'pipeline': None,
        'accuracy': None,
        'precision': None,
        'status': 'failed',
        'error': f"Kernel-based classifier family failed: {str(e)}"
    }])

# kernel_based(df, 'class')
