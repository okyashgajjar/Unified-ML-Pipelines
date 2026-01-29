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

def tree_based(df, target) :

  # separate data for x & y
  x = df.drop(columns=[target])
  y = df[target]


  # train test split
  X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


  # we don't use PCA in tree based model, Scaling & Encoding too.
  models = {
      'Decision Tree' : DecisionTreeClassifier(),
      'Random Forest' : RandomForestClassifier(),
      'Gradient Boosting' : GradientBoostingClassifier(),
      'AdaBoost' : AdaBoostClassifier(),
      'XGBoost' : XGBClassifier(),
      'LightGBM' : LGBMClassifier()
  }


  for model_name, model in models.items() :

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc_score = accuracy_score(y_test, y_pred)
    prec_score = precision_score(y_test, y_pred, average='weighted')
    class_report = classification_report(y_test, y_pred)

    print(f"Model : {model_name}")
    print(f"Accuracy Score : {acc_score}")
    print(f"Precision Score : {prec_score}")
    print(f"Classification Report : \n{class_report}")
    # print(f"ROC AUC Score : {roc_auc}")
    print("\n")


