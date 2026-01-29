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


def neural_network(df, target) :

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
      'MLP' : MLPClassifier(
          hidden_layer_sizes=(100, 50),
          activation='relu',
          solver='adam',
          alpha=0.0001,
          batch_size='auto',
      )
  }

  for model_name, model in models.items() : 
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
    