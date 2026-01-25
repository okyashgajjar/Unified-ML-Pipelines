import streamlit as st
import requests
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Unified ML Pipelines",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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

# Helper functions
def check_api_health() -> bool:
    """Check if FastAPI backend is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def upload_and_train(file, target_column: str, use_parallel: bool, enable_mlflow: bool) -> Optional[dict]:
    """Upload file and trigger training"""
    try:
        files = {"file": (file.name, file.getvalue(), "text/csv")}
        data = {
            "target_column": target_column,
            "use_parallel": use_parallel,
            "enable_mlflow": enable_mlflow
        }
        response = requests.post(f"{API_BASE_URL}/api/train", files=files, data=data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Training failed: {str(e)}")
        return None

def get_results(job_id: str) -> Optional[dict]:
    """Get training results for a job"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/results/{job_id}")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to fetch results: {str(e)}")
        return None

def get_all_jobs() -> Optional[dict]:
    """Get all training jobs"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/jobs")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to fetch jobs: {str(e)}")
        return None

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">🤖 Unified ML Pipelines</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Mathematics-Driven Parallel ML Pipelines</div>', unsafe_allow_html=True)
    
    # Check API health
    if not check_api_health():
        st.error("⚠️ FastAPI backend is not running! Please start it with: `uvicorn app:app --reload`")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Navigation
        page = st.radio("Navigate", ["🏠 Home", "📊 Train Models", "📈 View Results", "📋 Job History"])
        
        st.divider()
        
        # API Status
        st.success("✅ API Connected")
        st.caption(f"Endpoint: {API_BASE_URL}")
    
    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Train Models":
        show_train_page()
    elif page == "📈 View Results":
        show_results_page()
    elif page == "📋 Job History":
        show_history_page()

def show_home_page():
    """Home page with project overview"""
    st.header("Welcome to Unified ML Pipelines v1.1")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📚 About")
        st.markdown("""
        This system trains **multiple ML model families** in parallel, each with 
        mathematically correct preprocessing pipelines.
        
        **Key Features:**
        - ✅ Parallel execution (2x faster)
        - ✅ Error handling at all levels
        - ✅ MLFlow experiment tracking
        - ✅ RESTful API backend
        - ✅ Interactive Streamlit UI
        """)
    
    with col2:
        st.subheader("🎯 Model Families")
        st.markdown("""
        1. **Weight-Based** (Linear, Ridge, Lasso)
        2. **Tree-Based** (DT, RF, XGBoost, GBM)
        3. **Neural Networks** (MLP Regressor)
        4. **Instance-Based** (KNN, Radius Neighbors)
        
        Each family uses preprocessing optimized for its mathematical learning behavior.
        """)
    
    st.divider()
    
    st.subheader("🚀 Quick Start")
    st.markdown("""
    1. Navigate to **📊 Train Models**
    2. Upload your CSV dataset
    3. Select target column
    4. Click **Start Training**
    5. View results in **📈 View Results**
    """)

def show_train_page():
    """Training page with file upload"""
    st.header("📊 Train Models")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
    
    if uploaded_file:
        # Preview data
        try:
            df_preview = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)  # Reset file pointer
            
            st.success(f"✅ File loaded: {uploaded_file.name}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows", df_preview.shape[0])
            with col2:
                st.metric("Columns", df_preview.shape[1])
            
            # Show preview
            with st.expander("📋 Data Preview"):
                st.dataframe(df_preview.head(10))
            
            # Configuration
            st.subheader("⚙️ Training Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                target_column = st.selectbox(
                    "Target Column",
                    options=df_preview.columns.tolist(),
                    help="Select the column you want to predict"
                )
            
            with col2:
                use_parallel = st.checkbox("Use Parallel Execution", value=True, help="Train model families in parallel (faster)")
            
            enable_mlflow = st.checkbox("Enable MLFlow Tracking", value=True, help="Log experiments to MLFlow")
            
            # Train button
            if st.button("🚀 Start Training", type="primary", use_container_width=True):
                with st.spinner("Uploading dataset and starting training..."):
                    result = upload_and_train(uploaded_file, target_column, use_parallel, enable_mlflow)
                    
                    if result:
                        st.session_state['current_job_id'] = result['job_id']
                        st.markdown(f'<div class="success-box">✅ Training started successfully!<br>Job ID: <code>{result["job_id"]}</code></div>', unsafe_allow_html=True)
                        
                        # Auto-navigate to results
                        st.info("💡 Navigate to **📈 View Results** to monitor progress")
                        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

def show_results_page():
    """Results page with job monitoring"""
    st.header("📈 View Results")
    
    # Job ID input
    job_id = st.text_input(
        "Job ID",
        value=st.session_state.get('current_job_id', ''),
        help="Enter the job ID from training"
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("🔍 Fetch Results", type="primary", use_container_width=True):
            if job_id:
                fetch_and_display_results(job_id)
            else:
                st.warning("Please enter a job ID")
    
    with col2:
        if st.button("🔄 Auto-Refresh"):
            if job_id:
                st.session_state['auto_refresh'] = True
                st.rerun()
    
    # Auto-refresh logic
    if st.session_state.get('auto_refresh') and job_id:
        fetch_and_display_results(job_id, auto_refresh=True)

def fetch_and_display_results(job_id: str, auto_refresh: bool = False):
    """Fetch and display training results"""
    result = get_results(job_id)
    
    if not result:
        return
    
    status = result['status']
    
    # Status indicator
    if status == 'queued':
        st.info("⏳ Job is queued...")
    elif status == 'running':
        st.warning("🔄 Training in progress...")
        if auto_refresh:
            time.sleep(3)
            st.rerun()
    elif status == 'failed':
        st.markdown(f'<div class="error-box">❌ Training failed<br>Error: {result.get("error", "Unknown error")}</div>', unsafe_allow_html=True)
        st.session_state['auto_refresh'] = False
    elif status == 'completed':
        st.session_state['auto_refresh'] = False
        st.success("✅ Training completed!")
        st.balloons()  # Celebrate completion
        
        # Display results
        display_training_results(result['results'])

def display_training_results(results: list):
    """Display training results with visualizations"""
    if not results:
        st.warning("No results available")
        return
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # Filter successful models
    df_success = df_results[df_results['status'] == 'success'].copy()
    
    if df_success.empty:
        st.error("All models failed to train")
        st.dataframe(df_results)
        return
    
    # Metrics overview
    st.subheader("📊 Model Performance Overview")
    
    # Top 3 models by MAE
    st.markdown("### 🏆 Top 3 Models (by MAE)")
    top_3 = df_success.nsmallest(3, 'mae')
    
    cols = st.columns(3)
    for idx, (_, row) in enumerate(top_3.iterrows()):
        with cols[idx]:
            st.markdown(f"""
            <div class="metric-card">
                <h4>#{idx+1} {row['model_name']}</h4>
                <p><b>MAE:</b> {row['mae']:.4f}</p>
                <p><b>R²:</b> {row['r2']:.4f}</p>
                <p><b>RMSE:</b> {row['rmse']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # MAE comparison
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
        # R² comparison
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
    
    # Error Analysis Section - Outlier Detection
    st.subheader("🔍 Error & Outlier Analysis")
    
    st.markdown("""
    This section helps identify models sensitive to outliers and data quality issues.
    - **RMSE vs MAE Gap**: Large gap indicates sensitivity to outliers (RMSE penalizes large errors more)
    - **High MAPE**: Indicates issues with small target values or percentage-based errors
    """)
    
    # Calculate error metrics for analysis
    df_analysis = df_success.copy()
    df_analysis['rmse_mae_ratio'] = df_analysis['rmse'] / (df_analysis['mae'] + 1e-10)
    df_analysis['outlier_sensitivity'] = (df_analysis['rmse_mae_ratio'] - 1) * 100  # % above ideal
    
    # RMSE vs MAE Scatter Plot
    col1, col2 = st.columns(2)
    
    with col1:
        fig_scatter = px.scatter(
            df_analysis,
            x='mae',
            y='rmse',
            color='model_name',
            hover_data=['r2', 'mape'],
            title='RMSE vs MAE (Closer to diagonal = Less Outlier Sensitivity)',
            labels={'mae': 'MAE', 'rmse': 'RMSE', 'model_name': 'Model'}
        )
        
        # Add ideal line (RMSE = MAE when no outliers)
        max_val = max(df_analysis['rmse'].max(), df_analysis['mae'].max())
        fig_scatter.add_shape(
            type='line',
            x0=0, y0=0,
            x1=max_val, y1=max_val,
            line=dict(color='gray', dash='dash'),
            name='Ideal (No Outliers)'
        )
        fig_scatter.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Outlier Sensitivity Bar Chart
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
    
    # Issue Highlights
    st.subheader("⚠️ Potential Issues Detected")
    
    issues_found = False
    
    # Check for high outlier sensitivity
    high_sensitivity = df_analysis[df_analysis['outlier_sensitivity'] > 20]
    if not high_sensitivity.empty:
        issues_found = True
        with st.expander("🎯 High Outlier Sensitivity Detected", expanded=True):
            st.warning(f"**{len(high_sensitivity)} model(s)** show high sensitivity to outliers:")
            for _, row in high_sensitivity.iterrows():
                st.write(f"- **{row['model_name']}**: {row['outlier_sensitivity']:.1f}% above ideal")
            st.info("💡 **Suggestion**: Consider removing outliers from your data or use robust models like Ridge/Lasso regression.")
    
    # Check for negative R²
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
        # Convert MAPE string to float if needed
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
    """Job history page"""
    st.header("📋 Job History")
    
    if st.button("🔄 Refresh", type="primary"):
        st.rerun()
    
    jobs_data = get_all_jobs()
    
    if jobs_data and jobs_data.get('total', 0) > 0:
        jobs = jobs_data['jobs']
        
        st.metric("Total Jobs", jobs_data['total'])
        
        # Display jobs
        for job in jobs:
            with st.expander(f"Job: {job['job_id'][:8]}... - {job['status'].upper()}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Status:** {job['status']}")
                    st.write(f"**Target Column:** {job.get('target_column', 'N/A')}")
                    st.write(f"**Filename:** {job.get('filename', 'N/A')}")
                
                with col2:
                    st.write(f"**Created:** {job.get('created_at', 'N/A')}")
                    st.write(f"**Job ID:** `{job['job_id']}`")
                
                if st.button(f"View Results", key=f"view_{job['job_id']}"):
                    st.session_state['current_job_id'] = job['job_id']
                    st.info("✅ Job ID copied! Navigate to **📈 View Results** in the sidebar.")
    else:
        st.info("No training jobs found. Start a new training job!")

# Run app
if __name__ == "__main__":
    # Initialize session state
    if 'current_job_id' not in st.session_state:
        st.session_state['current_job_id'] = ''
    if 'auto_refresh' not in st.session_state:
        st.session_state['auto_refresh'] = False
    
    main()
