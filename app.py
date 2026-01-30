from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import pandas as pd
import uuid
import os
from datetime import datetime

from Superwised_Regression.tabular_data.weight_reg import weight_based_model
from Superwised_Regression.tabular_data.tree_reg import tree_based_model
from Superwised_Regression.tabular_data.nn_reg import neural_network
from Superwised_Regression.tabular_data.instance_reg import instance_based_model
from Superwised_Regression.tabular_data.parallel_executor import execute_model_families_parallel
from Superwised_Regression.tabular_data.mlflow_tracker import MLFlowTracker
from Superwised_Regression.preprocessing import datacleaning

# Classification imports
from Superwised_Classification.tabular_data.weight_class import weight_based as weight_based_classifier
from Superwised_Classification.tabular_data.tree_class import tree_based as tree_based_classifier
from Superwised_Classification.tabular_data.nn_class import neural_network as nn_classifier
from Superwised_Classification.tabular_data.kernel_class import kernel_based as kernel_based_classifier
from Superwised_Classification.tabular_data.instance_class import instance_based as instance_based_classifier

# Initialize FastAPI app
app = FastAPI(
    title="ML Pipelines API",
    description="Mathematics-Driven Parallel ML Pipelines for Regression & Classification",
    version="2.0.0"
)

# Add CORS middleware for Streamlit integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for training jobs (in production, use Redis or database)
training_jobs: Dict[str, Dict[str, Any]] = {}

# Pydantic models for request/response validation
class TrainRequest(BaseModel):
    target_column: str
    learning_type: str = "regression"  # "regression" or "classification"
    use_parallel: bool = True
    enable_mlflow: bool = True

class TrainResponse(BaseModel):
    job_id: str
    status: str
    message: str
    timestamp: str

class ResultsResponse(BaseModel):
    job_id: str
    status: str
    results: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str

# Background task for model training
def train_models_background(job_id: str, df: pd.DataFrame, target: str, use_parallel: bool, enable_mlflow: bool):
    """
    Background task to train models and update job status.
    """
    try:
        training_jobs[job_id]['status'] = 'running'
        training_jobs[job_id]['started_at'] = datetime.now().isoformat()
        
        print(f"[Job {job_id}] Starting training...")
        print(f"[Job {job_id}] Dataset shape: {df.shape}")
        print(f"[Job {job_id}] Target column: '{target}'")
        print(f"[Job {job_id}] Columns: {list(df.columns)}")
        
        # Define model families
        model_functions = [
            ('Weight-Based Models', weight_based_model),
            ('Tree-Based Models', tree_based_model),
            ('Neural Network Models', neural_network),
            ('Instance-Based Models', instance_based_model)
        ]
        
        # Execute models sequentially with progress output
        from Superwised_Regression.tabular_data.parallel_executor import execute_model_families_sequential
        results_df = execute_model_families_sequential(df, target, model_functions)
        
        # Sort by MAE (ascending)
        results_df = results_df.sort_values(by='mae', ascending=True)
        
        # Log to MLFlow if enabled
        if enable_mlflow:
            try:
                tracker = MLFlowTracker(experiment_name=f"Job_{job_id}")
                for family_name, func in model_functions:
                    family_results = results_df[results_df['model_name'].str.contains(family_name.split()[0], case=False, na=False)]
                    if not family_results.empty:
                        tracker.log_family_results(family_results, family_name)
            except Exception as mlflow_error:
                print(f"MLFlow logging failed: {str(mlflow_error)}")
        
        # Convert results to dict (excluding pipeline objects)
        results_dict = results_df.drop(columns=['pipeline'], errors='ignore').to_dict(orient='records')
        
        # Update job status
        training_jobs[job_id]['status'] = 'completed'
        training_jobs[job_id]['results'] = results_dict
        training_jobs[job_id]['completed_at'] = datetime.now().isoformat()
        
        print(f"[Job {job_id}] Training completed successfully!")
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[Job {job_id}] Training failed with error:")
        print(error_details)
        
        training_jobs[job_id]['status'] = 'failed'
        training_jobs[job_id]['error'] = str(e)
        training_jobs[job_id]['error_details'] = error_details
        training_jobs[job_id]['completed_at'] = datetime.now().isoformat()

# Background task for classification model training
def train_classification_background(job_id: str, df: pd.DataFrame, target: str, enable_mlflow: bool):
    """
    Background task to train classification models and update job status.
    """
    try:
        training_jobs[job_id]['status'] = 'running'
        training_jobs[job_id]['started_at'] = datetime.now().isoformat()
        
        print(f"[Job {job_id}] Starting classification training...")
        print(f"[Job {job_id}] Dataset shape: {df.shape}")
        print(f"[Job {job_id}] Target column: '{target}'")
        print(f"[Job {job_id}] Columns: {list(df.columns)}")
        
        # Define classification model families
        model_functions = [
            ('Weight-Based Classifiers', weight_based_classifier),
            ('Tree-Based Classifiers', tree_based_classifier),
            ('Neural Network Classifiers', nn_classifier),
            ('Kernel-Based Classifiers', kernel_based_classifier),
            ('Instance-Based Classifiers', instance_based_classifier)
        ]
        
        # Execute each model family and collect results
        all_results = []
        for family_name, func in model_functions:
            print(f"[Job {job_id}] Training {family_name}...")
            try:
                family_results = func(df, target)
                all_results.append(family_results)
                print(f"[Job {job_id}] {family_name} completed.")
            except Exception as family_error:
                print(f"[Job {job_id}] {family_name} failed: {str(family_error)}")
                all_results.append(pd.DataFrame([{
                    'model_name': family_name,
                    'best_params': None,
                    'pipeline': None,
                    'accuracy': None,
                    'precision': None,
                    'status': 'failed',
                    'error': str(family_error)
                }]))
        
        # Combine all results
        results_df = pd.concat(all_results, ignore_index=True)
        
        # Sort by accuracy (descending) - higher is better
        results_df = results_df.sort_values(by='accuracy', ascending=False, na_position='last')
        
        # Log to MLFlow if enabled
        if enable_mlflow:
            try:
                tracker = MLFlowTracker(experiment_name=f"Classification_Job_{job_id}")
                for family_name, func in model_functions:
                    family_results = results_df[results_df['model_name'].str.contains(family_name.split()[0], case=False, na=False)]
                    if not family_results.empty:
                        tracker.log_family_results(family_results, family_name)
            except Exception as mlflow_error:
                print(f"MLFlow logging failed: {str(mlflow_error)}")
        
        # Convert results to dict (excluding pipeline objects)
        results_dict = results_df.drop(columns=['pipeline'], errors='ignore').to_dict(orient='records')
        
        # Update job status
        training_jobs[job_id]['status'] = 'completed'
        training_jobs[job_id]['results'] = results_dict
        training_jobs[job_id]['completed_at'] = datetime.now().isoformat()
        
        print(f"[Job {job_id}] Classification training completed successfully!")
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[Job {job_id}] Classification training failed with error:")
        print(error_details)
        
        training_jobs[job_id]['status'] = 'failed'
        training_jobs[job_id]['error'] = str(e)
        training_jobs[job_id]['error_details'] = error_details
        training_jobs[job_id]['completed_at'] = datetime.now().isoformat()

# API Endpoints

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        timestamp=datetime.now().isoformat()
    )

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        timestamp=datetime.now().isoformat()
    )

@app.post("/api/train", response_model=TrainResponse)
async def train_models(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_column: str = Form(...),  # Required - user selects from their data
    learning_type: str = Form("regression"),  # "regression" or "classification"
    use_parallel: str = Form("true"),
    enable_mlflow: str = Form("true")
):
    """
    Upload a CSV dataset and trigger model training.
    
    Args:
        file: CSV file upload
        target_column: Name of the target column
        learning_type: Type of learning - "regression" or "classification"
        use_parallel: Whether to use parallel execution (default: True)
        enable_mlflow: Whether to log to MLFlow (default: True)
    
    Returns:
        Training job ID and status
    """
    
    # Convert string form data to boolean
    use_parallel_bool = use_parallel.lower() == "true" if isinstance(use_parallel, str) else use_parallel
    enable_mlflow_bool = enable_mlflow.lower() == "true" if isinstance(enable_mlflow, str) else enable_mlflow
    learning_type_str = learning_type.lower() if isinstance(learning_type, str) else "regression"
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        # Save uploaded file temporarily
        job_id = str(uuid.uuid4())
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, f"{job_id}.csv")
        
        # Write file
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Load and clean data
        df = datacleaning(temp_file_path)
        
        # Validate target column
        if target_column not in df.columns:
            os.remove(temp_file_path)
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{target_column}' not found in dataset. Available columns: {list(df.columns)}"
            )
        
        # Create job entry
        training_jobs[job_id] = {
            'status': 'queued',
            'created_at': datetime.now().isoformat(),
            'target_column': target_column,
            'learning_type': learning_type_str,
            'use_parallel': use_parallel_bool,
            'enable_mlflow': enable_mlflow_bool,
            'filename': file.filename
        }
        
        # Start background training based on learning type
        if learning_type_str == "classification":
            background_tasks.add_task(
                train_classification_background,
                job_id,
                df,
                target_column,
                enable_mlflow_bool
            )
        else:
            background_tasks.add_task(
                train_models_background,
                job_id,
                df,
                target_column,
                use_parallel_bool,
                enable_mlflow_bool
            )
        
        # Clean up temp file
        os.remove(temp_file_path)
        
        return TrainResponse(
            job_id=job_id,
            status="queued",
            message="Training job started successfully",
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.get("/api/results/{job_id}", response_model=ResultsResponse)
async def get_results(job_id: str):
    """
    Get training results for a specific job ID.
    
    Args:
        job_id: Training job ID
    
    Returns:
        Training results and status
    """
    
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job ID not found")
    
    job = training_jobs[job_id]
    
    return ResultsResponse(
        job_id=job_id,
        status=job['status'],
        results=job.get('results'),
        error=job.get('error'),
        timestamp=datetime.now().isoformat()
    )

@app.get("/api/jobs")
async def list_jobs():
    """
    List all training jobs.
    
    Returns:
        List of all jobs with their status
    """
    
    jobs_summary = []
    for job_id, job_data in training_jobs.items():
        jobs_summary.append({
            'job_id': job_id,
            'status': job_data['status'],
            'created_at': job_data.get('created_at'),
            'target_column': job_data.get('target_column'),
            'learning_type': job_data.get('learning_type', 'regression'),
            'filename': job_data.get('filename')
        })
    
    return {"jobs": jobs_summary, "total": len(jobs_summary)}

@app.get("/api/experiments")
async def list_experiments():
    """
    List MLFlow experiments.
    
    Returns:
        List of MLFlow experiments
    """
    
    try:
        import mlflow
        experiments = mlflow.search_experiments()
        
        experiments_list = []
        for exp in experiments:
            experiments_list.append({
                'experiment_id': exp.experiment_id,
                'name': exp.name,
                'artifact_location': exp.artifact_location,
                'lifecycle_stage': exp.lifecycle_stage
            })
        
        return {"experiments": experiments_list, "total": len(experiments_list)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch experiments: {str(e)}")

@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a training job from memory.
    
    Args:
        job_id: Training job ID
    
    Returns:
        Deletion confirmation
    """
    
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job ID not found")
    
    del training_jobs[job_id]
    
    return {"message": f"Job {job_id} deleted successfully", "timestamp": datetime.now().isoformat()}

# Run with: uvicorn app:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
