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
    st.header("Welcome to Unified ML Pipelines")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📚 About")
        st.markdown("""
        This app demonstrates **Mathematics-driven Hyperparameter tuned Parallel ML Pipelines**.
        
        **Key Features:**
        - ✅ Consolidated App (Frontend + Backend)
        - ✅ Multi-Family Model Training
        - ✅ Automatic Preprocessing
        - ✅ Interactive Visualizations
        """)
    with col2:
        st.subheader("🎯 Model Families")
        st.markdown("""
        1. **Weight-Based** (Linear, Ridge, Lasso)
        2. **Tree-Based** (RF, XGBoost, GBM)
        3. **Neural Networks** (MLP)
        4. **Instance-Based** (KNN)
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
            col1, col2 = st.columns(2)
            with col1:
                target_column = st.selectbox("Target Column", df_preview.columns.tolist())
            with col2:
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
                with st.spinner("🔄 Preprocessing data and training models... This may take a minute..."):
                    try:
                        # Clean Data
                        df_clean = datacleaning(temp_path)
                        
                        # Train
                        success, message = run_training_pipeline(
                            df_clean, target_column, use_parallel, enable_mlflow, job_id
                        )
                        
                        if success:
                            st.balloons()
                            st.success("✅ Training Completed! Redirecting to results...")
                            time.sleep(1)
                            # Ideally we'd auto-redirect, but user can click tab
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
    """Reuse the visualization logic"""
    if not results:
        st.warning("No results to display.")
        return
        
    df_results = pd.DataFrame(results)
    df_success = df_results[df_results['status'] == 'success'].copy()
    
    if df_success.empty:
        st.error("No models trained successfully.")
        return

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

    # Metrics comparison radar chart
    st.subheader("📈 Multi-Metric Comparison")
    
    # Normalize metrics for radar chart (0-1 scale)
    df_radar = df_success.copy()
    df_radar['mae_norm'] = 1 - (df_radar['mae'] - df_radar['mae'].min()) / (df_radar['mae'].max() - df_radar['mae'].min() + 1e-10)
    df_radar['rmse_norm'] = 1 - (df_radar['rmse'] - df_radar['rmse'].min()) / (df_radar['rmse'].max() - df_radar['rmse'].min() + 1e-10)
    df_radar['r2_norm'] = (df_radar['r2'] - df_radar['r2'].min()) / (df_radar['r2'].max() - df_radar['r2'].min() + 1e-10)
    
    # Create radar chart for top 5 models
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

    # Detailed results table with hyperparameters
    st.subheader("📋 Detailed Results")
    
    # Format display columns - include best_params
    display_cols = ['model_name', 'mae', 'rmse', 'r2', 'mse', 'mape', 'best_params']
    available_cols = [c for c in display_cols if c in df_success.columns]
    display_df = df_success[available_cols].copy()
    display_df = display_df.sort_values('mae')
    
    # Format hyperparameters as readable string
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

    # --- Error Analysis Section ---
    st.subheader("🔍 Error & Outlier Analysis")
    
    st.markdown("""
    This section helps identify models sensitive to outliers and data quality issues.
    - **RMSE vs MAE Gap**: Large gap indicates sensitivity to outliers (RMSE penalizes large errors more)
    - **High MAPE**: Indicates issues with small target values or percentage-based errors
    """)
    
    # Calculate error metrics
    df_analysis = df_success.copy()
    df_analysis['rmse_mae_ratio'] = df_analysis['rmse'] / (df_analysis['mae'] + 1e-10)
    df_analysis['outlier_sensitivity'] = (df_analysis['rmse_mae_ratio'] - 1) * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        # RMSE vs MAE Scatter
        fig_scatter = px.scatter(
            df_analysis,
            x='mae', 
            y='rmse',
            color='model_name',
            hover_data=['r2', 'mape'],
            title='RMSE vs MAE (Closer to diagonal = Less Outlier Sensitivity)',
            labels={'mae': 'MAE', 'rmse': 'RMSE', 'model_name': 'Model'}
        )
        max_val = max(df_analysis['rmse'].max(), df_analysis['mae'].max())
        fig_scatter.add_shape(
            type='line', x0=0, y0=0, x1=max_val, y1=max_val,
            line=dict(color='gray', dash='dash'), name='Ideal (No Outliers)'
        )
        fig_scatter.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with col2:
        # Outlier Sensitivity Bar
        fig_outlier = px.bar(
            df_analysis.sort_values('outlier_sensitivity'),
            x='outlier_sensitivity',
            y='model_name',
            orientation='h',
            title='Outlier Sensitivity Score (Lower = Better)',
            labels={'outlier_sensitivity': 'Sensitivity %', 'model_name': 'Model'},
            color='outlier_sensitivity',
            color_continuous_scale='RdYlGn_r'
        )
        fig_outlier.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_outlier, use_container_width=True)
        
    # Issue Detection
    st.subheader("⚠️ Potential Issues Detected")
    issues_found = False
    
    # High Sensitivity
    high_sensitivity = df_analysis[df_analysis['outlier_sensitivity'] > 20]
    if not high_sensitivity.empty:
        issues_found = True
        with st.expander("🎯 High Outlier Sensitivity Detected", expanded=True):
            st.warning(f"**{len(high_sensitivity)} model(s)** show high sensitivity to outliers:")
            for _, row in high_sensitivity.iterrows():
                st.write(f"- **{row['model_name']}**: {row['outlier_sensitivity']:.1f}% above ideal")
            st.info("💡 **Suggestion**: Consider removing outliers from your data or use robust models like Ridge/Lasso regression.")
                
    # Negative R2
    negative_r2 = df_analysis[df_analysis['r2'] < 0]
    if not negative_r2.empty:
        issues_found = True
        with st.expander("❌ Poor Model Fit (Negative R²)", expanded=True):
            st.error(f"**{len(negative_r2)} model(s)** have negative R² (worse than mean baseline):")
            for _, row in negative_r2.iterrows():
                st.write(f"- **{row['model_name']}**: R² = {row['r2']:.4f}")
            st.info("💡 **Suggestion**: These models are not suitable for this dataset. Check for data quality issues or feature engineering needs.")

    # Check for high MAPE
    if 'mape' in df_analysis.columns:
        try:
            df_analysis['mape_val'] = df_analysis['mape'].apply(
                lambda x: float(str(x).replace('%', '')) if x else 0
            )
            high_mape = df_analysis[df_analysis['mape_val'] > 50]
            if not high_mape.empty:
                issues_found = True
                with st.expander("📊 High Percentage Error", expanded=False):
                    st.warning(f"**{len(high_mape)} model(s)** have MAPE > 50%:")
                    for _, row in high_mape.iterrows():
                        st.write(f"- **{row['model_name']}**: MAPE = {row['mape']}")
                    st.info("💡 **Suggestion**: The target variable may have small values near zero, causing inflated percentage errors. Consider log transformation.")
        except:
            pass

    if not issues_found:
        st.success("✅ No major issues detected! All models show reasonable error patterns.")

    st.divider()
    
    # Download results
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
            "Target": data.get('target_column'),
            "File": data.get('filename')
        })
        
    st.dataframe(pd.DataFrame(history_data))

if __name__ == "__main__":
    main()
