import sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np 
import warnings
warnings.filterwarnings("ignore")

# df = '/home/yashgajjar/Desktop/ml-pipelines/dataset/Housing.csv'

def neural_network(df, target) :
    
    try :
        # Input validation
        if not isinstance(df, pd.DataFrame) :
            raise TypeError("Input must be a pandas DataFrame")
        
        if target not in df.columns :
            raise ValueError(f"Target column '{target}' not found in dataframe")
        
        if df.empty :
            raise ValueError("DataFrame is empty")
        
        # df = pd.read_csv(df)
        X = df.drop(columns=[target])
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
                    ('num', StandardScaler(), num_features),
                ]
            )
        else :
            raise ValueError("Dataframe Columns is not proper numeric")

        model = MLPRegressor(
            hidden_layer_sizes=(100, 100),
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            tol=1e-4
        )

        pipeline = Pipeline(
            steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ]
        )

        param_grid = {
            'model__hidden_layer_sizes': [(100,), (50, 50)],
            'model__alpha': [0.001],
            'model__learning_rate': ['adaptive']
        }

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=3,  # Reduced for faster training
            scoring='neg_mean_squared_error',
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

        results = {
            'model_name': 'Neural Network MLP Regressor',
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
        }
        # print(results)
        return pd.DataFrame([results])
    
    except Exception as e :
        # Return error as dataframe for consistency
        return pd.DataFrame([{
            'model_name': 'Neural Network MLP Regressor',
            'best_params': None,
            'pipeline': None,
            'mse': None,
            'rmse': None,
            'r2': None,
            'mae': None,
            'mape': None,
            'status': 'failed',
            'error': f"Neural network model failed: {str(e)}"
        }])

# neural_network(df, 'price')