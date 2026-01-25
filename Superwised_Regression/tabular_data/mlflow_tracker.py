import mlflow
import mlflow.sklearn
from typing import Dict, Any
import pandas as pd

class MLFlowTracker:
    """
    MLFlow tracking utilities for model family experiments.
    """
    
    def __init__(self, experiment_name: str = "ML-Pipelines-Regression"):
        """
        Initialize MLFlow tracker.
        
        Args:
            experiment_name: Name of the MLFlow experiment
        """
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
    
    def log_model_run(self, model_name: str, params: Dict[str, Any], metrics: Dict[str, float], model_pipeline=None):
        """
        Log a single model run to MLFlow.
        
        Args:
            model_name: Name of the model
            params: Hyperparameters dictionary
            metrics: Metrics dictionary (mse, rmse, r2, mae, mape)
            model_pipeline: Trained sklearn pipeline (optional)
        """
        
        with mlflow.start_run(run_name=model_name):
            # Log parameters
            if params:
                for param_name, param_value in params.items():
                    # Remove 'model__' prefix for cleaner logging
                    clean_param_name = param_name.replace('model__', '')
                    mlflow.log_param(clean_param_name, param_value)
            
            # Log metrics
            if metrics:
                for metric_name, metric_value in metrics.items():
                    # Skip non-numeric metrics
                    if isinstance(metric_value, (int, float)):
                        mlflow.log_metric(metric_name, metric_value)
            
            # Log model to registry
            if model_pipeline is not None:
                try:
                    mlflow.sklearn.log_model(
                        model_pipeline,
                        artifact_path="model",
                        registered_model_name=model_name.replace(' ', '_')
                    )
                except Exception as e:
                    print(f"Warning: Could not log model {model_name}: {str(e)}")
    
    def log_family_results(self, results_df: pd.DataFrame, family_name: str):
        """
        Log all models from a family to MLFlow.
        
        Args:
            results_df: DataFrame with model results
            family_name: Name of the model family
        """
        
        for _, row in results_df.iterrows():
            # Skip failed models
            if row.get('status') == 'failed':
                print(f"Skipping failed model: {row['model_name']}")
                continue
            
            # Prepare metrics
            metrics = {
                'mse': row.get('mse'),
                'rmse': row.get('rmse'),
                'r2': row.get('r2'),
                'mae': row.get('mae')
            }
            
            # Handle MAPE (might be string with %)
            mape_value = row.get('mape')
            if isinstance(mape_value, str) and '%' in mape_value:
                try:
                    metrics['mape'] = float(mape_value.replace('%', ''))
                except:
                    pass
            elif isinstance(mape_value, (int, float)):
                metrics['mape'] = mape_value
            
            # Log to MLFlow
            self.log_model_run(
                model_name=row['model_name'],
                params=row.get('best_params', {}),
                metrics=metrics,
                model_pipeline=row.get('pipeline')
            )
        
        print(f"✓ Logged {family_name} results to MLFlow")
    
    @staticmethod
    def get_experiment_runs(experiment_name: str) -> pd.DataFrame:
        """
        Get all runs from an experiment.
        
        Args:
            experiment_name: Name of the experiment
        
        Returns:
            DataFrame with run information
        """
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            return runs
        else:
            return pd.DataFrame()
