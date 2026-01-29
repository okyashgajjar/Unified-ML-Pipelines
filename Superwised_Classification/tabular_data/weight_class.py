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

def weight_based(df, target) :

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
      'Logistic Regression': LogisticRegression(),
      'Ridge Classifier' : RidgeClassifier()
  }

  # print(y.shape)
  # print(y_train.shape)
  # print(y_test.shape)

  for model_name, model in models.items() :

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

      pipeline.fit(X_train, y_train)
      y_pred = pipeline.predict(X_test)

      acc_score = accuracy_score(y_test, y_pred)
      prec_score = precision_score(y_test, y_pred, average='weighted')
      class_report = classification_report(y_test, y_pred)


      print(f"Model : {model_name}")
      print(f"Accuracy Score : {acc_score}")
      print(f"Precision Score : {prec_score}")
      print(f"Classification Report : \n{class_report}")
      # print(f"ROC AUC Score : {roc_auc}")
      print("\n")