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

def kernel_based(df, target) :

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
              ('num', pca_preprocessor, num_features)
          ]
      )


  # models
  models = {
        'SVC' : SVC()
  }

  for model_name, model in models.items() :

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