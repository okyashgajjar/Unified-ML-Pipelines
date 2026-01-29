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

def instance_based(df, target) : 

  x = df.drop(columns=[target])
  y = df[target]

  # now remove categorical features from X
  X = x.select_dtypes(exclude=['object', 'category'])

  if X.shape[1] == 0:
    raise ValueError("Instance Learns from only numerical columns & No numerical features found in the input")

  X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

  num_featues = x.select_dtypes(exlude=['object']).columns

  # pca-pipeline
  pca_pipeline = Pipeline([
      ('scaler', StandardScaler()),
      ('pca', PCA(n_components=0.95))
  ])

  # make a preprocessor
  if len(num_featues) >= 10 :
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_featues)
    ])
    pca_preprocessor = ColumnTransformer([
        ('num', pca_pipeline, num_featues)
    ])
  elif len(num_featues) == 0 :
    raise ValueError("Instance Learns from only numerical columns & No numerical features found in the input")


  # models 
  models = {
      'KNN' : KNeighborsClassifier(n_neighbors=5)
  }

  for model_name, model in models.items():
      if len(num_featues) >= 10 :
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
      prec_score = precision_score(y_test, y_pred)
      class_report = classification_report(y_test, y_pred)
      
      print(f"Model : {model_name}")
      print(f"Accuracy Score : {acc_score}")
      print(f"Precision Score : {prec_score}")
      print(f"Classification Report : \n{class_report}")
      # print(f"ROC AUC Score : {roc_auc}")
      print("\n")
