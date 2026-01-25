# not using this -- too no

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, RationalQuadratic, ExpSineSquared, DotProduct, ConstantKernel
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV
# PCA
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np 
import warnings
# warnings.filterwarnings("ignore", category=ConvergenceWarning)


# df = '/home/yashgajjar/Desktop/ml-pipelines/dataset/Housing.csv'

def probabilistic_model(df, target) :
    df = pd.read_csv(df)
    df = df.select_dtypes(exclude=['object', 'category'])
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    num_features = X.select_dtypes(include='number').columns

    if len(num_features) > 0 :
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_features)
            ]
        )
    else :
        raise ValueError("Dataframe Columns is not proper numeric")

    

    kernels = [
        RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e5)),
        Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e5), nu=1.5),
        RationalQuadratic(length_scale=1.0, alpha=1.0,
                        length_scale_bounds=(1e-2, 1e5)),
        ConstantKernel(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e5))
    ]



    model = GaussianProcessRegressor()


    pipeline = Pipeline(
            steps=[
                ('preprocessor', preprocessor),
                ('pca', PCA(n_components=0.95)),
                ('model', model)
            ]
        )

    param_grid = {
        "model__kernel": kernels,
        "model__alpha": [1e-2, 1e-1, 1e0, 1e1],
        "model__n_restarts_optimizer": [0, 2]
    }
  
    grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    results = {
            'model_name': model,
            'best_model' : grid.best_estimator_,
            'best_params': grid.best_params_,
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2,
            'mae': mae,
            'mape': f'{mape*100:.2f}%'
        }

    print(results)
    return results

# probabilistic_model(df, 'price')