from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from typing import List, Callable, Tuple
import time

def execute_model_families_parallel(df: pd.DataFrame, target: str, model_functions: List[Tuple[str, Callable]]) -> pd.DataFrame:
    """
    Execute multiple model family functions in parallel using ThreadPoolExecutor.
    
    This achieves O(n) time complexity where n is the longest-running family,
    instead of O(4n) for sequential execution.
    
    Note: Using ThreadPoolExecutor instead of ProcessPoolExecutor for better
    compatibility with Python 3.14 and to avoid pickling issues with sklearn objects.
    
    Args:
        df: Input dataframe
        target: Target column name
        model_functions: List of tuples (family_name, function)
    
    Returns:
        Combined dataframe with results from all families
    """
    
    all_results = []
    
    # Use ThreadPoolExecutor for compatibility (sklearn releases GIL during training)
    with ThreadPoolExecutor(max_workers=len(model_functions)) as executor:
        # Submit all tasks
        future_to_family = {
            executor.submit(func, df, target): family_name 
            for family_name, func in model_functions
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_family):
            family_name = future_to_family[future]
            try:
                result_df = future.result()
                all_results.append(result_df)
                print(f"✓ {family_name} completed successfully")
            except Exception as e:
                print(f"✗ {family_name} failed: {str(e)}")
                # Create error dataframe
                error_df = pd.DataFrame([{
                    'model_name': family_name,
                    'best_params': None,
                    'pipeline': None,
                    'mse': None,
                    'rmse': None,
                    'r2': None,
                    'mae': None,
                    'mape': None,
                    'status': 'failed',
                    'error': str(e)
                }])
                all_results.append(error_df)
    
    # Combine all results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        return combined_results
    else:
        return pd.DataFrame()



def execute_model_families_sequential(df: pd.DataFrame, target: str, model_functions: List[Tuple[str, Callable]]) -> pd.DataFrame:
    """
    Execute model family functions sequentially (fallback option).
    
    Args:
        df: Input dataframe
        target: Target column name
        model_functions: List of tuples (family_name, function)
    
    Returns:
        Combined dataframe with results from all families
    """
    
    all_results = []
    
    for family_name, func in model_functions:
        try:
            print(f"Running {family_name}...")
            result_df = func(df, target)
            all_results.append(result_df)
            print(f"✓ {family_name} completed successfully")
        except Exception as e:
            print(f"✗ {family_name} failed: {str(e)}")
            error_df = pd.DataFrame([{
                'model_name': family_name,
                'best_params': None,
                'pipeline': None,
                'mse': None,
                'rmse': None,
                'r2': None,
                'mae': None,
                'mape': None,
                'status': 'failed',
                'error': str(e)
            }])
            all_results.append(error_df)
    
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        return combined_results
    else:
        return pd.DataFrame()
