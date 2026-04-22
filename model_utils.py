import os
import pickle
import zipfile
import pandas as pd
from datetime import datetime
import shutil

def save_models_as_zip(results_df, job_id=None, folder_name="models"):
    """
    Saves all successful models from the results dataframe into a zip file using pickle.
    
    Args:
        results_df: Pandas DataFrame containing the training results.
        job_id: Optional job identifier to name the zip file.
        folder_name: Directory where the zip file will be saved.
    
    Returns:
        Path to the generated zip file, or None if no models were saved.
    """
    if 'pipeline' not in results_df.columns:
        return None
    
    # Create export directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    id_str = job_id if job_id else timestamp
    export_dir = os.path.join("temp_exports", f"models_{id_str}")
    os.makedirs(export_dir, exist_ok=True)
    os.makedirs(folder_name, exist_ok=True)
    
    saved_count = 0
    for _, row in results_df.iterrows():
        if row.get('status') == 'success' and row.get('pipeline') is not None:
            # Create a clean filename
            safe_name = "".join([c if c.isalnum() else "_" for c in row['model_name']]).lower()
            file_path = os.path.join(export_dir, f"{safe_name}.pkl")
            
            try:
                with open(file_path, 'wb') as f:
                    pickle.dump(row['pipeline'], f)
                saved_count += 1
            except Exception as e:
                print(f"Error saving model {row['model_name']}: {e}")
    
    if saved_count == 0:
        shutil.rmtree(export_dir, ignore_errors=True)
        return None
    
    # Create zip file
    zip_path = os.path.join(folder_name, f"models_{id_str}.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(export_dir):
            for file in files:
                zipf.write(os.path.join(root, file), file)
    
    # Clean up temp directory
    shutil.rmtree(os.path.dirname(export_dir), ignore_errors=True)
    
    return zip_path
