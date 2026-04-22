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
    from Superwised_Regression.preprocessing import datacleaning
    from model_utils import save_models_as_zip
    
    # Classification imports
    from Superwised_Classification.tabular_data.weight_class import weight_based as weight_based_classifier
    from Superwised_Classification.tabular_data.tree_class import tree_based as tree_based_classifier
    from Superwised_Classification.tabular_data.nn_class import neural_network as nn_classifier
    from Superwised_Classification.tabular_data.kernel_class import kernel_based as kernel_based_classifier
    from Superwised_Classification.tabular_data.instance_class import instance_based as instance_based_classifier
    from Superwised_Classification.nlp_data.nlp_class import nlp_classification
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
# Custom CSS for Modern UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .main {
        background: radial-gradient(circle at top right, #1e1e2f, #121212);
    }
    
    /* Glassmorphism containers */
    .stApp {
        background: #0e1117;
    }
    
    [data-testid="stSidebar"] {
        background-color: rgba(17, 25, 40, 0.75);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #a0a0a0;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 300;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(5px);
        padding: 1.5rem;
        border-radius: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(0, 242, 254, 0.5);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
    }
    
    .metric-card h4 {
        color: #00f2fe;
        margin-top: 0;
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    .metric-card p {
        color: #e0e0e0;
        margin: 0.4rem 0;
        font-size: 0.95rem;
    }
    
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #ffffff;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(79, 172, 254, 0.4);
    }

    .success-box {
        background: rgba(0, 255, 127, 0.1);
        border-left: 5px solid #00ff7f;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #00ff7f;
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
        
        # Save models as ZIP
        zip_path = save_models_as_zip(results_df, job_id=job_id)
        
        # Update Job Success
        st.session_state['jobs'][job_id]['status'] = 'completed'
        st.session_state['jobs'][job_id]['results'] = results_dict
        st.session_state['jobs'][job_id]['models_zip'] = zip_path
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
            ('Instance-Based Classifiers', instance_based_classifier),
            ('NLP Classification (if applicable)', nlp_classification)
        ]
        
        # Execute each model family and collect results
        all_results = []
        for family_name, func in model_functions:
            try:
                print(f"Running {family_name}...")
                family_results = func(df, target_column)
                all_results.append(family_results)
                print(f"✓ {family_name} completed successfully")
            except Exception as family_error:
                print(f"✗ {family_name} failed: {str(family_error)}")
                all_results.append(pd.DataFrame([{
                    'model_name': family_name,
                    'best_params': None,
                    'pipeline': None,
                    'accuracy': None,
                    'precision': None,
                    'f1_score': None,
                    'status': 'failed',
                    'error': str(family_error)
                }]))
        
        # Combine all results
        if not all_results:
            print("Warning: No results collected from any model family")
            return False, "No model families could be executed"
        
        results_df = pd.concat(all_results, ignore_index=True)
        print(f"Classification results collected: {len(results_df)} models")
        
        # Sort by accuracy (descending) - higher is better
        if 'accuracy' in results_df.columns:
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
        
        # Save models as ZIP
        zip_path = save_models_as_zip(results_df, job_id=job_id)
        
        # Update Job Success
        st.session_state['jobs'][job_id]['status'] = 'completed'
        st.session_state['jobs'][job_id]['results'] = results_dict
        st.session_state['jobs'][job_id]['models_zip'] = zip_path
        st.session_state['jobs'][job_id]['completed_at'] = datetime.now().isoformat()
        
        return True, "Classification training completed successfully"

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        st.session_state['jobs'][job_id]['status'] = 'failed'
        st.session_state['jobs'][job_id]['error'] = str(e)
        st.session_state['jobs'][job_id]['completed_at'] = datetime.now().isoformat()
        return False, str(e)

def run_nlp_pipeline(df: pd.DataFrame, target_column: str, enable_mlflow: bool, job_id: str):
    """
    Executes the NLP-specific classification training pipeline.
    """
    try:
        # Update status
        st.session_state['jobs'][job_id]['status'] = 'running'
        st.session_state['jobs'][job_id]['started_at'] = datetime.now().isoformat()
        
        # Run NLP classification
        print(f"Running NLP Classification...")
        results_df = nlp_classification(df, target_column)
        print(f"✓ NLP Classification completed successfully")
        
        # Sort by accuracy
        if 'accuracy' in results_df.columns:
            results_df = results_df.sort_values(by='accuracy', ascending=False, na_position='last')
        
        # MLFlow Logging (Optional)
        if enable_mlflow:
            try:
                tracker = MLFlowTracker(experiment_name=f"NLP_Job_{job_id}")
                tracker.log_family_results(results_df, "NLP Classification")
            except Exception as e:
                print(f"MLFlow logging warning: {e}")
        
        # Convert results
        results_dict = results_df.drop(columns=['pipeline'], errors='ignore').to_dict(orient='records')
        
        # Save models as ZIP
        zip_path = save_models_as_zip(results_df, job_id=job_id)
        
        # Update Job Success
        st.session_state['jobs'][job_id]['status'] = 'completed'
        st.session_state['jobs'][job_id]['results'] = results_dict
        st.session_state['jobs'][job_id]['models_zip'] = zip_path
        st.session_state['jobs'][job_id]['completed_at'] = datetime.now().isoformat()
        
        return True, "NLP training completed successfully"

    except Exception as e:
        import traceback
        st.session_state['jobs'][job_id]['status'] = 'failed'
        st.session_state['jobs'][job_id]['error'] = str(e)
        st.session_state['jobs'][job_id]['completed_at'] = datetime.now().isoformat()
        return False, str(e)

# --- UI Functions ---

def main():
    # Header
    st.markdown('<div class="main-header">Unified ML Pipelines</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced Parallel Neural Engine • v3.0</div>', unsafe_allow_html=True)
    
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
    st.header("Welcome to Unified ML Pipelines v3.0")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.03); padding: 2rem; border-radius: 1.5rem; border: 1px solid rgba(255,255,255,0.05);">
            <h3 style="color: #00f2fe; margin-top: 0;">🚀 Project Genesis</h3>
            <p style="color: #a0a0a0; font-size: 1.1rem; line-height: 1.6;">
                Unified ML Pipelines is a premium machine learning ecosystem designed for extreme performance. 
                Our engine parallelizes complex mathematical operations to deliver hyper-optimized models 
                in record time.
            </p>
            <hr style="opacity: 0.1; margin: 1.5rem 0;">
            <ul style="color: #e0e0e0; list-style: none; padding-left: 0;">
                <li style="margin-bottom: 0.8rem;">✨ <b>Neural Architecture:</b> Parallel family-based execution</li>
                <li style="margin-bottom: 0.8rem;">✨ <b>Smart Preprocessing:</b> Context-aware feature scaling</li>
                <li style="margin-bottom: 0.8rem;">✨ <b>NLP Engine:</b> Semantic vectorization with Word2Vec</li>
                <li style="margin-bottom: 0.8rem;">✨ <b>MLOps Ready:</b> One-click model export and deployment</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.03); padding: 2rem; border-radius: 1.5rem; border: 1px solid rgba(255,255,255,0.05);">
            <h3 style="color: #4facfe; margin-top: 0;">🎯 Engine Core</h3>
            <div style="margin-bottom: 1.5rem;">
                <b style="color: #00f2fe;">REGRESSION</b><br>
                <span style="color: #a0a0a0; font-size: 0.9rem;">Weight-Based • Tree-Based • Neural Nets • Instance</span>
            </div>
            <div style="margin-bottom: 1.5rem;">
                <b style="color: #00f2fe;">CLASSIFICATION</b><br>
                <span style="color: #a0a0a0; font-size: 0.9rem;">Logistic • GBM • XGBoost • LightGBM • SVC</span>
            </div>
            <div>
                <b style="color: #00f2fe;">NATURAL LANGUAGE</b><br>
                <span style="color: #a0a0a0; font-size: 0.9rem;">Semantic Embeddings • TF-IDF • N-Grams</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

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
                    options=["Regression", "Classification", "NLP Classification"],
                    help="Regression for numbers, Classification for categories, NLP for text data"
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
                progress_placeholder = st.empty()
                progress_placeholder.info(f"🔄 Preprocessing data for {learning_type.lower()} training...")
                
                try:
                    # Clean Data
                    df_clean = datacleaning(temp_path)
                    progress_placeholder.info(f"✓ Data cleaned. Shape: {df_clean.shape}. Starting {learning_type.lower()} training...")
                    
                    # Train based on learning type
                    if learning_type.lower() == "classification":
                        progress_placeholder.info(f"🔄 Training classification models on target '{target_column}'...")
                        success, message = run_classification_pipeline(
                            df_clean, target_column, enable_mlflow, job_id
                        )
                    elif learning_type.lower() == "nlp classification":
                        progress_placeholder.info(f"🔄 Training NLP models on text for target '{target_column}'...")
                        success, message = run_nlp_pipeline(
                            df_clean, target_column, enable_mlflow, job_id
                        )
                    else:
                        progress_placeholder.info(f"🔄 Training regression models on target '{target_column}'...")
                        success, message = run_training_pipeline(
                            df_clean, target_column, use_parallel, enable_mlflow, job_id
                        )
                    
                    if success:
                        progress_placeholder.empty()  # Clear the progress message
                        st.balloons()
                        st.success(f"✅ {learning_type} Training Completed!")
                        time.sleep(1)
                        st.info("Navigate to **📈 View Results** to see the analysis.")
                    else:
                        progress_placeholder.empty()
                        st.error(f"❌ Training Failed: {message}")
                        
                except Exception as e:
                    import traceback
                    progress_placeholder.empty()
                    st.error(f"Error during execution: {e}")
                    st.code(traceback.format_exc(), language='python')
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
    """Display classification results with accuracy/precision metrics and enhanced visualizations"""
    # Metrics
    st.subheader("🏆 Top Performing Models (by Accuracy)")
    top_3 = df_success.nlargest(3, 'accuracy')
    
    cols = st.columns(3)
    for idx, (_, row) in enumerate(top_3.iterrows()):
        with cols[idx]:
            acc_val = row['accuracy'] if row['accuracy'] is not None else 0
            prec_val = row['precision'] if row['precision'] is not None else 0
            f1_val = row.get('f1_score', 0) if row.get('f1_score') is not None else 0
            st.markdown(f"""
            <div class="metric-card">
                <h4>{row['model_name']}</h4>
                <p>Accuracy</p>
                <div class="metric-value">{acc_val:.4f}</div>
                <p>F1 Score: <b>{f1_val:.4f}</b></p>
            </div>
            """, unsafe_allow_html=True)
            
    st.divider()
    
    # Charts - Accuracy and Precision bars
    st.subheader("📊 Model Performance Comparison")
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

    st.divider()

    # NEW: Accuracy vs Precision Scatter Plot
    st.subheader("🎯 Accuracy vs Precision Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Scatter plot showing accuracy vs precision trade-off
        fig_scatter = px.scatter(
            df_success,
            x='accuracy',
            y='precision',
            text='model_name',
            title='Accuracy vs Precision Trade-off',
            labels={'accuracy': 'Accuracy', 'precision': 'Precision'},
            color='accuracy',
            size='precision',
            color_continuous_scale='Viridis'
        )
        fig_scatter.update_traces(textposition='top center', textfont_size=9)
        fig_scatter.update_layout(height=450, showlegend=False)
        # Add diagonal reference line (ideal: accuracy = precision)
        fig_scatter.add_shape(
            type='line', x0=0, y0=0, x1=1, y1=1,
            line=dict(color='gray', dash='dash', width=1)
        )
        fig_scatter.add_annotation(
            x=0.85, y=0.75, text="Ideal Line", showarrow=False,
            font=dict(size=10, color='gray')
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Accuracy-Precision Gap Analysis
        df_gap = df_success.copy()
        df_gap['gap'] = abs(df_gap['accuracy'] - df_gap['precision'])
        df_gap = df_gap.sort_values('gap', ascending=True)
        
        fig_gap = px.bar(
            df_gap,
            x='gap',
            y='model_name',
            orientation='h',
            title='Accuracy-Precision Gap (Lower = More Balanced)',
            labels={'gap': 'Gap', 'model_name': 'Model'},
            color='gap',
            color_continuous_scale='RdYlGn_r'
        )
        fig_gap.update_layout(showlegend=False, height=450)
        st.plotly_chart(fig_gap, use_container_width=True)
    
    st.divider()

    # NEW: Model Family Performance Comparison
    st.subheader("👨‍👩‍👧‍👦 Model Family Performance")
    
    # Extract model family from model name
    df_family = df_success.copy()
    def get_family(name):
        name_lower = name.lower()
        if 'logistic' in name_lower or 'ridge' in name_lower:
            return 'Weight-Based'
        elif any(x in name_lower for x in ['tree', 'forest', 'gradient', 'ada', 'lgbm', 'lightgbm', 'xgb']):
            return 'Tree-Based'
        elif 'mlp' in name_lower or 'neural' in name_lower:
            return 'Neural Network'
        elif 'svc' in name_lower or 'svm' in name_lower:
            return 'Kernel-Based'
        elif 'knn' in name_lower or 'neighbor' in name_lower:
            return 'Instance-Based'
        else:
            return 'Other'
    
    df_family['family'] = df_family['model_name'].apply(get_family)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Grouped bar chart by family
        family_stats = df_family.groupby('family').agg({
            'accuracy': 'mean',
            'precision': 'mean'
        }).reset_index()
        
        fig_family = go.Figure(data=[
            go.Bar(name='Accuracy', x=family_stats['family'], y=family_stats['accuracy'], marker_color='#2ecc71'),
            go.Bar(name='Precision', x=family_stats['family'], y=family_stats['precision'], marker_color='#3498db')
        ])
        fig_family.update_layout(
            barmode='group',
            title='Average Performance by Model Family',
            xaxis_title='Model Family',
            yaxis_title='Score',
            height=400,
            yaxis=dict(range=[0, 1])
        )
        st.plotly_chart(fig_family, use_container_width=True)
    
    with col2:
        # Box plot showing performance distribution by family
        fig_box = px.box(
            df_family,
            x='family',
            y='accuracy',
            color='family',
            title='Accuracy Distribution by Model Family',
            labels={'family': 'Model Family', 'accuracy': 'Accuracy'}
        )
        fig_box.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    st.divider()

    # Radar chart
    st.subheader("📈 Multi-Metric Comparison (Top 5)")
    top_5 = df_success.nlargest(5, 'accuracy')
    
    fig_radar = go.Figure()
    for _, row in top_5.iterrows():
        acc_val = row['accuracy'] if row['accuracy'] is not None else 0
        prec_val = row['precision'] if row['precision'] is not None else 0
        # Add F1 approximation for richer radar
        f1_approx = 2 * (acc_val * prec_val) / (acc_val + prec_val + 1e-10)
        fig_radar.add_trace(go.Scatterpolar(
            r=[acc_val, prec_val, f1_approx],
            theta=['Accuracy', 'Precision', 'F1 (approx)'],
            fill='toself',
            name=row['model_name']
        ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Top 5 Models - Classification Metrics Radar",
        height=500
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()

    # NEW: Performance Heatmap
    st.subheader("🔥 Performance Heatmap")
    
    # Create heatmap data
    heatmap_df = df_success[['model_name', 'accuracy', 'precision']].copy()
    heatmap_df['f1_approx'] = 2 * (heatmap_df['accuracy'] * heatmap_df['precision']) / (heatmap_df['accuracy'] + heatmap_df['precision'] + 1e-10)
    heatmap_df = heatmap_df.set_index('model_name')
    heatmap_df.columns = ['Accuracy', 'Precision', 'F1 (approx)']
    
    fig_heatmap = px.imshow(
        heatmap_df.values,
        x=heatmap_df.columns.tolist(),
        y=heatmap_df.index.tolist(),
        color_continuous_scale='RdYlGn',
        aspect='auto',
        title='Model Performance Heatmap',
        labels=dict(x='Metric', y='Model', color='Score')
    )
    fig_heatmap.update_layout(height=max(300, len(df_success) * 35))
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.divider()

    # Detailed results table
    st.subheader("📋 Detailed Results")
    display_cols = ['model_name', 'accuracy', 'precision', 'f1_score', 'best_params']
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
    
    # Model Download ZIP
    zip_path = st.session_state['jobs'][st.session_state['current_job_id']].get('models_zip')
    if zip_path and os.path.exists(zip_path):
        with open(zip_path, "rb") as f:
            st.download_button(
                label="📦 Download All Trained Models (ZIP)",
                data=f,
                file_name=os.path.basename(zip_path),
                mime="application/zip",
                use_container_width=True
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
    
    # Model Download ZIP
    zip_path = st.session_state['jobs'][st.session_state['current_job_id']].get('models_zip')
    if zip_path and os.path.exists(zip_path):
        with open(zip_path, "rb") as f:
            st.download_button(
                label="📦 Download All Trained Models (ZIP)",
                data=f,
                file_name=os.path.basename(zip_path),
                mime="application/zip",
                use_container_width=True
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
