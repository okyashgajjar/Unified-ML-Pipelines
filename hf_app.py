import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List, Dict, Any
import time
import uuid
from datetime import datetime
import os

# --- Imports from Backend Logic ---
# Importing directly from the project structure
try:
    from Superwised_Regression.tabular_data.weight_reg import weight_based_model
    from Superwised_Regression.tabular_data.tree_reg import tree_based_model
    from Superwised_Regression.tabular_data.nn_reg import neural_network
    from Superwised_Regression.tabular_data.instance_reg import instance_based_model
    # We use sequential execution for safer deployment on shared resources like HF Spaces
    from Superwised_Regression.tabular_data.parallel_executor import execute_model_families_sequential
    from Superwised_Regression.tabular_data.mlflow_tracker import MLFlowTracker
    from Superwised_Regression.preprocessing import datacleaning
    
    # Classification imports
    from Superwised_Classification.tabular_data.weight_class import weight_based as weight_based_classifier
    from Superwised_Classification.tabular_data.tree_class import tree_based as tree_based_classifier
    from Superwised_Classification.tabular_data.nn_class import neural_network as nn_classifier
    from Superwised_Classification.tabular_data.kernel_class import kernel_based as kernel_based_classifier
    from Superwised_Classification.tabular_data.instance_class import instance_based as instance_based_classifier
except ImportError as e:
    st.error(f"Critical Error: Failed to import ML modules. Make sure you are in the root directory. Error: {e}")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="Unified ML Pipelines",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #ffd700 0%, #ffb347 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        color: #1a1a2e;
        border: 2px solid #e6a800;
    }
    .metric-card h4 {
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }
    .metric-card p {
        color: #2d2d44;
        margin: 0.3rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- State Management ---
if 'jobs' not in st.session_state:
    st.session_state['jobs'] = {}
if 'current_job_id' not in st.session_state:
    st.session_state['current_job_id'] = ''

# --- Core Logic Functions (Adapted from app.py) ---

def run_training_pipeline(df: pd.DataFrame, target_column: str, use_parallel: bool, enable_mlflow: bool, job_id: str):
    """
    Executes the training pipeline synchronously.
    """
    try:
        # Update status
        st.session_state['jobs'][job_id]['status'] = 'running'
        st.session_state['jobs'][job_id]['started_at'] = datetime.now().isoformat()
        
        # Define model families
        model_functions = [
            ('Weight-Based Models', weight_based_model),
            ('Tree-Based Models', tree_based_model),
            ('Neural Network Models', neural_network),
            ('Instance-Based Models', instance_based_model)
        ]
        
        # Execute models
        # Note: We force sequential execution for stability on HF Spaces free tier
        results_df = execute_model_families_sequential(df, target_column, model_functions)
        
        # Sort by MAE
        results_df = results_df.sort_values(by='mae', ascending=True)
        
        # MLFlow Logging (Optional)
        if enable_mlflow:
            try:
                # Set local tracking URI explicitly to avoid connection errors if no server
                # import mlflow
                # mlflow.set_tracking_uri("file:./mlruns") 
                
                tracker = MLFlowTracker(experiment_name=f"Job_{job_id}")
                for family_name, func in model_functions:
                    family_results = results_df[results_df['model_name'].str.contains(family_name.split()[0], case=False, na=False)]
                    if not family_results.empty:
                        tracker.log_family_results(family_results, family_name)
            except Exception as e:
                print(f"MLFlow logging warning: {e}")
                # Don't fail the job just because MLFlow failed
        
        # Convert results
        results_dict = results_df.drop(columns=['pipeline'], errors='ignore').to_dict(orient='records')
        
        # Update Job Success
        st.session_state['jobs'][job_id]['status'] = 'completed'
        st.session_state['jobs'][job_id]['results'] = results_dict
        st.session_state['jobs'][job_id]['completed_at'] = datetime.now().isoformat()
        
        return True, "Training completed successfully"

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        st.session_state['jobs'][job_id]['status'] = 'failed'
        st.session_state['jobs'][job_id]['error'] = str(e)
        st.session_state['jobs'][job_id]['completed_at'] = datetime.now().isoformat()
        return False, str(e)

def run_classification_pipeline(df: pd.DataFrame, target_column: str, enable_mlflow: bool, job_id: str):
    """
    Executes the classification training pipeline synchronously.
    """
    try:
        # Update status
        st.session_state['jobs'][job_id]['status'] = 'running'
        st.session_state['jobs'][job_id]['started_at'] = datetime.now().isoformat()
        
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
            try:
                family_results = func(df, target_column)
                all_results.append(family_results)
            except Exception as family_error:
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
        
        # MLFlow Logging (Optional)
        if enable_mlflow:
            try:
                tracker = MLFlowTracker(experiment_name=f"Classification_Job_{job_id}")
                for family_name, func in model_functions:
                    family_results = results_df[results_df['model_name'].str.contains(family_name.split()[0], case=False, na=False)]
                    if not family_results.empty:
                        tracker.log_family_results(family_results, family_name)
            except Exception as e:
                print(f"MLFlow logging warning: {e}")
        
        # Convert results
        results_dict = results_df.drop(columns=['pipeline'], errors='ignore').to_dict(orient='records')
        
        # Update Job Success
        st.session_state['jobs'][job_id]['status'] = 'completed'
        st.session_state['jobs'][job_id]['results'] = results_dict
        st.session_state['jobs'][job_id]['completed_at'] = datetime.now().isoformat()
        
        return True, "Classification training completed successfully"

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        st.session_state['jobs'][job_id]['status'] = 'failed'
        st.session_state['jobs'][job_id]['error'] = str(e)
        st.session_state['jobs'][job_id]['completed_at'] = datetime.now().isoformat()
        return False, str(e)

# --- UI Functions ---

def main():
    # Header
    st.markdown('<div class="main-header">🤖 Unified ML Pipelines</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Mathematics-Driven Hyperparameter tuned ML Pipelines</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        page = st.radio("Navigate", ["🏠 Home", "📊 Train Models", "📈 View Results", "📋 Job History"])
        
        st.divider()
        st.caption("Running in Standalone Mode")

    # Routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Train Models":
        show_train_page()
    elif page == "📈 View Results":
        show_results_page()
    elif page == "📋 Job History":
        show_history_page()

def show_home_page():
    st.header("Welcome to Unified ML Pipelines v2.0")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📚 About")
        st.markdown("""
        This app demonstrates **Mathematics driven Hyperparameter tuned ML Pipelines**
        Now supporting both **Regression** and **Classification** tasks!
        
        **Key Features:**
        - ✅ **Regression & Classification** support
        - ✅ Consolidated App (Frontend + Backend)
        - ✅ Multi-Family Model Training
        - ✅ Automatic Preprocessing
        - ✅ Interactive Visualizations
        """)
    with col2:
        st.subheader("🎯 Model Families")
        st.markdown("""
        **Regression Models:**
        - Weight-Based (Linear, Ridge, Lasso)
        - Tree-Based (RF, XGBoost, GBM)
        - Neural Networks (MLP)
        - Instance-Based (KNN)
        
        **Classification Models:**
        - Weight-Based (Logistic, Ridge)
        - Tree-Based (DT, RF, GBM, LightGBM)
        - Neural Networks (MLP)
        - Kernel-Based (SVC)
        - Instance-Based (KNN)
        """)

def show_train_page():
    st.header("📊 Train Models")
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
    
    if uploaded_file:
        try:
            df_preview = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)
            
            st.success(f"✅ File loaded: {uploaded_file.name}")
            st.write(f"Shape: {df_preview.shape}")
            with st.expander("📋 Data Preview"):
                st.dataframe(df_preview.head())
            
            # Config
            col1, col2, col3 = st.columns(3)
            with col1:
                target_column = st.selectbox("Target Column", df_preview.columns.tolist())
            with col2:
                learning_type = st.selectbox(
                    "Learning Type",
                    options=["Regression", "Classification"],
                    help="Choose Regression for continuous targets, Classification for categorical"
                )
            with col3:
                # Parallel might be unstable on free tiers, keep option but default to what's safe
                use_parallel = st.checkbox("Use Parallel Execution", value=False, help="May be slower on single-vCPU environments")
            
            enable_mlflow = st.checkbox("Enable MLFlow Tracking", value=False)
            
            if st.button("🚀 Start Training", type="primary", use_container_width=True):
                # Create Job
                job_id = str(uuid.uuid4())
                st.session_state['jobs'][job_id] = {
                    'status': 'queued',
                    'created_at': datetime.now().isoformat(),
                    'target_column': target_column,
                    'learning_type': learning_type.lower(),
                    'filename': uploaded_file.name
                }
                st.session_state['current_job_id'] = job_id
                
                # Save temp file for processing (datacleaning expects a path)
                temp_dir = "temp_uploads"
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, f"{job_id}.csv")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # Run Training (Synchronous with Spinner)
                with st.spinner(f"🔄 Preprocessing data and training {learning_type.lower()} models... This may take a minute..."):
                    try:
                        # Clean Data
                        df_clean = datacleaning(temp_path)
                        
                        # Train based on learning type
                        if learning_type.lower() == "classification":
                            success, message = run_classification_pipeline(
                                df_clean, target_column, enable_mlflow, job_id
                            )
                        else:
                            success, message = run_training_pipeline(
                                df_clean, target_column, use_parallel, enable_mlflow, job_id
                            )
                        
                        if success:
                            st.balloons()
                            st.success(f"✅ {learning_type} Training Completed!")
                            time.sleep(1)
                            st.info("Navigate to **📈 View Results** to see the analysis.")
                        else:
                            st.error(f"❌ Training Failed: {message}")
                            
                    except Exception as e:
                        st.error(f"Error during execution: {e}")
                    finally:
                        # Cleanup
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                            
        except Exception as e:
            st.error(f"Error reading file: {e}")

def show_results_page():
    st.header("📈 View Results")
    
    # Selector for Jobs
    job_ids = list(st.session_state['jobs'].keys())
    if not job_ids:
        st.info("No training jobs run yet.")
        return
        
    current_id = st.session_state.get('current_job_id')
    if current_id not in job_ids:
        current_id = job_ids[-1]
        
    selected_job_id = st.selectbox(
        "Select Job", 
        job_ids, 
        index=job_ids.index(current_id),
        format_func=lambda x: f"{x[:8]}... ({st.session_state['jobs'][x].get('status', 'unknown')})"
    )
    
    job_data = st.session_state['jobs'][selected_job_id]
    status = job_data['status']
    
    if status == 'running':
        st.warning("Training in progress...")
    elif status == 'failed':
        st.error(f"Job Failed: {job_data.get('error')}")
    elif status == 'completed':
        results = job_data.get('results', [])
        display_training_results(results)

def display_training_results(results: list):
    """Reuse the visualization logic - auto-detects classification vs regression"""
    if not results:
        st.warning("No results to display.")
        return
        
    df_results = pd.DataFrame(results)
    df_success = df_results[df_results['status'] == 'success'].copy()
    
    if df_success.empty:
        st.error("No models trained successfully.")
        return

    # Detect result type based on available columns
    is_classification = 'accuracy' in df_success.columns and 'mae' not in df_success.columns
    
    if is_classification:
        display_classification_results(df_success)
    else:
        display_regression_results(df_success)

def display_classification_results(df_success: pd.DataFrame):
    """Display classification results with accuracy/precision metrics"""
    # Metrics
    st.subheader("🏆 Top Performing Models (by Accuracy)")
    top_3 = df_success.nlargest(3, 'accuracy')
    
    cols = st.columns(3)
    for idx, (_, row) in enumerate(top_3.iterrows()):
        with cols[idx]:
            acc_val = row['accuracy'] if row['accuracy'] is not None else 0
            prec_val = row['precision'] if row['precision'] is not None else 0
            st.markdown(f"""
            <div class="metric-card">
                <h4>#{idx+1} {row['model_name']}</h4>
                <p><b>Accuracy:</b> {acc_val:.4f}</p>
                <p><b>Precision:</b> {prec_val:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
            
    st.divider()
    
    # Charts
    col1, col2 = st.columns(2)
    with col1:
        fig_acc = px.bar(
            df_success.sort_values('accuracy', ascending=True),
            x='accuracy',
            y='model_name',
            orientation='h',
            title='Accuracy - Higher is Better',
            labels={'accuracy': 'Accuracy', 'model_name': 'Model'},
            color='accuracy',
            color_continuous_scale='RdYlGn'
        )
        fig_acc.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_acc, use_container_width=True)
    with col2:
        fig_prec = px.bar(
            df_success.sort_values('precision', ascending=True),
            x='precision',
            y='model_name',
            orientation='h',
            title='Precision - Higher is Better',
            labels={'precision': 'Precision', 'model_name': 'Model'},
            color='precision',
            color_continuous_scale='RdYlGn'
        )
        fig_prec.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_prec, use_container_width=True)

    # Radar chart
    st.subheader("📈 Multi-Metric Comparison")
    top_5 = df_success.nlargest(5, 'accuracy')
    
    fig_radar = go.Figure()
    for _, row in top_5.iterrows():
        acc_val = row['accuracy'] if row['accuracy'] is not None else 0
        prec_val = row['precision'] if row['precision'] is not None else 0
        fig_radar.add_trace(go.Scatterpolar(
            r=[acc_val, prec_val],
            theta=['Accuracy', 'Precision'],
            fill='toself',
            name=row['model_name']
        ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Top 5 Models - Classification Metrics",
        height=500
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Detailed results
    st.subheader("📋 Detailed Results")
    display_cols = ['model_name', 'accuracy', 'precision', 'best_params']
    available_cols = [c for c in display_cols if c in df_success.columns]
    display_df = df_success[available_cols].copy()
    display_df = display_df.sort_values('accuracy', ascending=False)
    
    if 'best_params' in display_df.columns:
        display_df['best_params'] = display_df['best_params'].apply(
            lambda x: str(x).replace('model__', '') if x else 'N/A'
        )
    
    st.dataframe(
        display_df.style.format({
            'accuracy': '{:.4f}',
            'precision': '{:.4f}'
        }),
        use_container_width=True,
        height=400
    )
    
    st.divider()
    
    # Download
    csv = df_success.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="classification_results.csv",
        mime="text/csv"
    )

def display_regression_results(df_success: pd.DataFrame):
    """Display regression results with MAE/R² metrics"""
    # Metrics
    st.subheader("🏆 Top Performing Models")
    top_3 = df_success.nsmallest(3, 'mae')
    
    cols = st.columns(3)
    for idx, (_, row) in enumerate(top_3.iterrows()):
        with cols[idx]:
            st.markdown(f"""
            <div class="metric-card">
                <h4>#{idx+1} {row['model_name']}</h4>
                <p><b>MAE:</b> {row['mae']:.4f}</p>
                <p><b>R²:</b> {row['r2']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
            
    st.divider()
    
    # Charts
    col1, col2 = st.columns(2)
    with col1:
        fig_mae = px.bar(
            df_success.sort_values('mae'),
            x='mae',
            y='model_name',
            orientation='h',
            title='Mean Absolute Error (MAE) - Lower is Better',
            labels={'mae': 'MAE', 'model_name': 'Model'},
            color='mae',
            color_continuous_scale='RdYlGn_r'
        )
        fig_mae.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_mae, use_container_width=True)
    with col2:
        fig_r2 = px.bar(
            df_success.sort_values('r2', ascending=False),
            x='r2',
            y='model_name',
            orientation='h',
            title='R² Score - Higher is Better',
            labels={'r2': 'R² Score', 'model_name': 'Model'},
            color='r2',
            color_continuous_scale='RdYlGn'
        )
        fig_r2.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_r2, use_container_width=True)

    # Radar chart
    st.subheader("📈 Multi-Metric Comparison")
    df_radar = df_success.copy()
    df_radar['mae_norm'] = 1 - (df_radar['mae'] - df_radar['mae'].min()) / (df_radar['mae'].max() - df_radar['mae'].min() + 1e-10)
    df_radar['rmse_norm'] = 1 - (df_radar['rmse'] - df_radar['rmse'].min()) / (df_radar['rmse'].max() - df_radar['rmse'].min() + 1e-10)
    df_radar['r2_norm'] = (df_radar['r2'] - df_radar['r2'].min()) / (df_radar['r2'].max() - df_radar['r2'].min() + 1e-10)
    
    top_5 = df_radar.nsmallest(5, 'mae')
    fig_radar = go.Figure()
    
    for _, row in top_5.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row['mae_norm'], row['rmse_norm'], row['r2_norm']],
            theta=['MAE', 'RMSE', 'R²'],
            fill='toself',
            name=row['model_name']
        ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Top 5 Models - Normalized Metrics",
        height=500
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Detailed results
    st.subheader("📋 Detailed Results")
    display_cols = ['model_name', 'mae', 'rmse', 'r2', 'mse', 'mape', 'best_params']
    available_cols = [c for c in display_cols if c in df_success.columns]
    display_df = df_success[available_cols].copy()
    display_df = display_df.sort_values('mae')
    
    if 'best_params' in display_df.columns:
        display_df['best_params'] = display_df['best_params'].apply(
            lambda x: str(x).replace('model__', '') if x else 'N/A'
        )
    
    st.dataframe(
        display_df.style.format({
            'mae': '{:.4f}',
            'rmse': '{:.4f}',
            'r2': '{:.4f}',
            'mse': '{:.4f}'
        }),
        use_container_width=True,
        height=400
    )

    st.divider()
    
    # Download
    csv = df_success.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="model_results.csv",
        mime="text/csv"
    )


def show_history_page():
    st.header("Job History")
    jobs = st.session_state['jobs']
    if not jobs:
        st.info("No history.")
        return
        
    history_data = []
    for jid, data in jobs.items():
        history_data.append({
            "Job ID": jid,
            "Date": data.get('created_at'),
            "Status": data.get('status'),
            "Type": data.get('learning_type', 'regression').capitalize(),
            "Target": data.get('target_column'),
            "File": data.get('filename')
        })
        
    st.dataframe(pd.DataFrame(history_data))

if __name__ == "__main__":
    main()
