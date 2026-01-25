from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
import pandas as pd
import numpy as np

# df = '/home/yash-test/Desktop/ml-pipelines/dataset/Housing.csv'

def instance_based_model(df, target):
    
    try :
        # Input validation
        if not isinstance(df, pd.DataFrame) :
            raise TypeError("Input must be a pandas DataFrame")
        
        if target not in df.columns :
            raise ValueError(f"Target column '{target}' not found in dataframe")
        
        if df.empty :
            raise ValueError("DataFrame is empty")
        
        # df = pd.read_csv(df)
        df = df.select_dtypes(exclude=['object', 'category'])
        
        if df.empty :
            raise ValueError("DataFrame is empty after dropping categorical columns")

        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        num_features = X.columns

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_features)
            ]
        )

        models = {
            'KNN': KNeighborsRegressor(),
            'RadiusNeighbourRegressor': RadiusNeighborsRegressor()
        }
        
        all_results = []

        for model_name, model in models.items():
            
            try :
                if model_name == "KNN":
                    param_grid = {
                        "model__n_neighbors": [3, 5, 7],
                        "model__weights": ['distance']
                    }

                elif model_name == "RadiusNeighbourRegressor":
                    param_grid = {
                        "model__radius": [1.0, 2.0],
                        "model__weights": ['distance']
                    }

                pipeline = Pipeline(
                    steps=[
                        ('preprocessor', preprocessor),
                        ('model', model)
                    ]
                )

                grid = GridSearchCV(
                    estimator=pipeline,
                    param_grid=param_grid,
                    cv=3,  # Reduced for faster training
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )

                grid.fit(X_train, y_train)

                y_pred = grid.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                mape = mean_absolute_percentage_error(y_test, y_pred)

                all_results.append({
                    'model_name': model_name,
                    'best_params': grid.best_params_,
                    'pipeline': pipeline,
                    'mse': mse,
                    'rmse': np.sqrt(mse),
                    'r2': r2,
                    'mae': mae,
                    'mape': f'{mape * 100:.2f}%',
                    'status': 'success',
                    'error': None
                })
            
            except Exception as model_error :
                all_results.append({
                    'model_name': model_name,
                    'best_params': None,
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
            'model_name': 'Instance-Based Models',
            'best_params': None,
            'pipeline': None,
            'mse': None,
            'rmse': None,
            'r2': None,
            'mae': None,
            'mape': None,
            'status': 'failed',
            'error': f"Instance-based model family failed: {str(e)}"
        }])


# instance_based_model(df, 'price')