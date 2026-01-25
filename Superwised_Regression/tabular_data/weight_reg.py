from jinja2.nodes import Include
# import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV

# Used GridSearchCV to find the best parameters for each model because hyperparameter is less in weight based models.

# df = '/home/yashgajjar/Desktop/ml-pipelines/dataset/Housing.csv'

def weight_based_model(df, target) :
    
    try :
        # Input validation
        if not isinstance(df, pd.DataFrame) :
            raise TypeError("Input must be a pandas DataFrame")
        
        if target not in df.columns :
            raise ValueError(f"Target column '{target}' not found in dataframe")
        
        if df.empty :
            raise ValueError("DataFrame is empty")
        
        # df = pd.read_csv(df)
        X = df.drop(columns=target)
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        num_features = X.select_dtypes(include='number').columns
        cat_features = X.select_dtypes(exclude='number').columns

        if len(num_features) > 0 and len(cat_features) > 0 :
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), num_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
                ]
            )
        elif len(num_features) == 0 :
            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
                ]
            )
        elif len(cat_features) == 0 :
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), num_features)
                ]
            )
        else :
            raise ValueError("Dataframe Columns is not proper numeric and catagorical")

        models = {
            'Linear Regression' : LinearRegression(),
            'Ridge Regression' : Ridge(),
            'Lasso Regression' : Lasso()
        }

        all_results = []

        for model_name, model in models.items() :
            try :
                if model_name == 'Linear Regression' :
                    param_grid = {
                        'model__fit_intercept' : [True, False]
                    }
                elif model_name == 'Ridge Regression' :
                    param_grid = {
                        'model__alpha' : [0.1, 1.0, 10.0]
                    }
                elif model_name == 'Lasso Regression' :
                    param_grid = {
                        'model__alpha' : [0.01, 0.3, 0.1, 0.5, 0.7, 1.0]
                    }

                pipeline = Pipeline(
                    steps=[
                        ('preprocessor', preprocessor),
                        ('model', model)
                    ]
                )

                grid_search = GridSearchCV(
                    estimator=pipeline,
                    param_grid=param_grid,
                    cv=3,  # Reduced for faster training
                    scoring='neg_root_mean_squared_error',
                    n_jobs=-1
                )

                grid_search.fit(X_train, y_train)

                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_

                y_pred = best_model.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                mape = mean_absolute_percentage_error(y_test, y_pred)

                all_results.append({
                    'model_name' : model_name,
                    # 'best_model' : best_model,
                    'best_params' : best_params,
                    'pipeline': pipeline,
                    'mse': mse,
                    'rmse': np.sqrt(mse),
                    'r2': r2,
                    'mae': mae,
                    'mape': f'{mape*100:.2f}%',
                    'status': 'success',
                    'error': None
                })
            
            except Exception as model_error :
                all_results.append({
                    'model_name' : model_name,
                    'best_params' : None,
                    'pipeline': None,
                    'mse': None,
                    'rmse': None,
                    'r2': None,
                    'mae': None,
                    'mape': None,
                    'status': 'failed',
                    'error': str(model_error)
                })

        return pd.DataFrame(all_results)
    
    except Exception as e :
        # Return error as dataframe for consistency
        return pd.DataFrame([{
            'model_name': 'Weight-Based Models',
            'best_params': None,
            'pipeline': None,
            'mse': None,
            'rmse': None,
            'r2': None,
            'mae': None,
            'mape': None,
            'status': 'failed',
            'error': f"Weight-based model family failed: {str(e)}"
        }])


# results_df = weight_based_model(df, 'price')

# print(
#     results_df
#     .round(3)
#     .sort_values(by="r2", ascending=False)
# )

