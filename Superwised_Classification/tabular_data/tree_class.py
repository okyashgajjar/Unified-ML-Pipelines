import numpy as np
import pandas as pd
import sklearn
# tree & ensemble based
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
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

def tree_based(df, target) :
  try:
    # Input validation
    if not isinstance(df, pd.DataFrame):
      raise TypeError("Input must be a pandas DataFrame")
    if target not in df.columns:
      raise ValueError(f"Target column '{target}' not found in dataframe")
    if df.empty:
      raise ValueError("DataFrame is empty")

    # df = pd.read_csv(df)

    # separate data for x & y
    x = df.drop(columns=[target])
    y = df[target]


    # train test split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    num_features = x.select_dtypes(include='number').columns
    cat_features = x.select_dtypes(exclude='number').columns


    if len(num_features) > 0 and  len(cat_features) > 0 :
              preprocessor = ColumnTransformer(
                  transformers=[
                      ('num', 'passthrough', num_features),
                      ('cat', OrdinalEncoder(
                          handle_unknown='use_encoded_value',
                          unknown_value=-1
                      ),cat_features)
                  ]
              )
    elif len(num_features) == 0 :
              preprocessor = ColumnTransformer(
                  transformers=[
                      ('cat', OrdinalEncoder(
                          handle_unknown='use_encoded_value',
                          unknown_value=-1
                      ),cat_features)
                  ]
              )
    elif len(cat_features) == 0 :
      preprocessor = ColumnTransformer(
          transformers=[
              ('num', 'passthrough', num_features)
          ]
      )
    else :
      raise ValueError("Dataframe Columns is not proper numeric and catagorical")

    # we don't use PCA in tree based model, Scaling & Encoding too.
    models = {
        'Decision Tree Classifier' : DecisionTreeClassifier(),
        'Random Forest Classifier' : RandomForestClassifier(),
        'Gradient Boosting Classifier' : GradientBoostingClassifier(),
        'AdaBoost Classifier' : AdaBoostClassifier(),
        # 'XGBoost Classifier' : XGBClassifier(),
        'LightGBM Classifier' : LGBMClassifier(force_col_wise=True, verbose=-1)
    }

    all_results = []

    for model_name, model in models.items() :
      try:
        if model_name == "Decision Tree Classifier" : 
          param_grid = {
            'model__max_depth': [5, 10, 20, None],
            'model__min_samples_split': [2, 5],
            'model__criterion': ['gini', 'entropy']
          }
        elif model_name == "Random Forest Classifier" :
          param_grid = {
            'model__n_estimators': [100, 200],
            'model__max_depth': [10, 20, None],
            'model__min_samples_split': [2, 5]
          }
        elif model_name == "Gradient Boosting Classifier" :
          param_grid = {
            'model__learning_rate': [0.05, 0.1],
            'model__n_estimators': [100, 200],
            'model__max_depth': [3, 5, 7]
          }
        elif model_name == "AdaBoost Classifier" :
          param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.5, 1.0]
          }
        elif model_name == "XGBoost Classifier" :
          param_grid = {
            'model__n_estimators': [100, 200],
            'model__learning_rate': [0.05, 0.1],
            'model__max_depth': [3, 5, 7]
          }
        elif model_name == "LightGBM Classifier" :
          param_grid = {
            # 'model__n_estimators': [100, 200],
            'model__learning_rate': [0.05, 0.1],
            'model__num_leaves': [31, 50]
          }

        pipeline = Pipeline(
            steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ]
        )

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
        best_params = random_search.best_params_
        best_model = random_search.best_estimator_
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
        'model_name': 'Tree-Based Classifiers',
        'best_params': None,
        'pipeline': None,
        'accuracy': None,
        'precision': None,
        'status': 'failed',
        'error': f"Tree-based classifier family failed: {str(e)}"
    }])

# tree_based(df, 'class')