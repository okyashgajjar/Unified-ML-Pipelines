from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import numpy  as np 
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV


# remove warnings
import warnings
warnings.filterwarnings("ignore")

# df = '/home/yashgajjar/Desktop/ml-pipelines/dataset/Housing.csv'

def tree_based_model(df, target) :
    
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

        models = {
            'Decision Tree Regressor' : DecisionTreeRegressor(),
            'Extra Tree Regressor' : ExtraTreeRegressor(),
            'Extra Trees Regressor (different)' : ExtraTreesRegressor(),
            'Random Forest Regressor' : RandomForestRegressor(),
            'XGBoost Regressor' : XGBRegressor(),
            # 'LightGBM Regressor' : LGBMRegressor( objective='regression', random_state=42, n_jobs=-1 ),
            'Gradient Boosting Regressor' : GradientBoostingRegressor()
        }

        all_results = []

        for model_name, model in models.items() :
            
            try :
                # Use smaller param grids for faster training
                if model_name == "Decision Tree Regressor" : 
                    param_grid = {
                        'model__max_depth' : [5, 10, None],
                        'model__min_samples_split' : [2, 5]
                    }
                elif model_name == "Extra Tree Regressor" :
                    param_grid = {
                        'model__max_depth' : [5, 10, None],
                        'model__min_samples_split': [2, 5]
                    }
                elif model_name == "Extra Trees Regressor (different)" :
                    param_grid = {
                        'model__n_estimators' : [50, 100],
                        'model__max_depth' : [5, 10],
                    }
                elif model_name == "Random Forest Regressor" :
                    param_grid = {
                        'model__n_estimators' : [50, 100],
                        'model__max_depth' : [5, 10],
                    }
                elif model_name == "XGBoost Regressor" :
                    param_grid = {
                        "model__n_estimators": [50, 100],
                        "model__max_depth": [3, 5],
                        "model__learning_rate": [0.1]
                    }
                elif model_name == 'Gradient Boosting Regressor' :
                    param_grid = {
                        "model__n_estimators": [50, 100],
                        "model__learning_rate": [0.1],
                        "model__max_depth": [3, 5]
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
                    n_iter=5,  # Reduced for faster training
                    scoring='neg_root_mean_squared_error',
                    cv=3,  # Reduced for faster training
                    n_jobs=-1,
                    random_state=42
                )

                random_search.fit(X_train, y_train)

                best_model = random_search.best_estimator_
                best_params = random_search.best_params_

                y_pred = best_model.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                mape = mean_absolute_percentage_error(y_test, y_pred)

                all_results.append ({
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
            'model_name': 'Tree-Based Models',
            'best_params': None,
            'pipeline': None,
            'mse': None,
            'rmse': None,
            'r2': None,
            'mae': None,
            'mape': None,
            'status': 'failed',
            'error': f"Tree-based model family failed: {str(e)}"
        }])


# tree_based_model(df, 'price')