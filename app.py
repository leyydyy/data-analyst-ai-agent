import streamlit as st
import pandas as pd
import os
import pickle
from config import init_session
from components.sidebar import render_sidebar
from components.cleaning_agent import render_cleaning_agent
from components.insights import render_insights
from components.visualization import render_visualization
from components.qa import render_qa

# PAGE CONFIG
st.set_page_config(
    page_title="AI Data Analyst Agent",
    layout="wide",
)

# GLOBAL STYLES
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem !important; }
    .upload-hero { text-align: center; padding: 4rem 2rem; }
    .upload-hero h1 {
        font-size: 2rem;
        font-weight: 800;
        color: #e2e8f0;
        margin-bottom: 0.5rem;
    }
    .upload-hero p { color: #94a3b8; margin-bottom: 2rem; }
    button[data-baseweb="tab"][aria-disabled="true"] {
        opacity: 0.4;
        cursor: not-allowed;
        pointer-events: none;
    }
</style>
""", unsafe_allow_html=True)

init_session()

# PERSISTENCE HELPERS
CACHE_PATH = "/tmp/analyst_agent_cache.pkl"

def save_to_cache(df, filename, insights_generated):
    """Persist dataframe and key state to disk."""
    with open(CACHE_PATH, "wb") as f:
        pickle.dump({
            "df": df,
            "filename": filename,
            "insights_generated": insights_generated
        }, f)

def load_from_cache():
    """Load persisted dataframe and state from disk if available."""
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    return None

def clear_cache():
    """Delete the cache file."""
    if os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)

# DATASET HELPERS
def dataset_is_clean(df):
    return df.isnull().sum().sum() == 0 and df.duplicated().sum() == 0

def insights_are_ready():
    return st.session_state.get("insights_generated", False)

# AUTO-RESTORE ON RELOAD
if st.session_state.get("df") is None:
    cached = load_from_cache()
    if cached is not None:
        st.session_state.df = cached["df"]
        st.session_state.current_file = cached["filename"]
        st.session_state.insights_generated = cached["insights_generated"]
        st.session_state.active_tab = 1 if cached["insights_generated"] else 0
        st.session_state.cleaning_done = False
        st.session_state.tab_switched = False

# UPLOAD SCREEN
if st.session_state.get("df") is None:
    st.markdown("""
    <div class="upload-hero">
        <h1>AI Data Analyst</h1>
        <p>Upload a CSV or Excel file to clean, visualize, and explore your data with AI.</p>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel File",
            type=["csv", "xlsx"],
            key="uploaded_file",
            label_visibility="collapsed"
        )
        if uploaded_file is not None:
            from utils.file_io import load_uploaded_file
            df_loaded = load_uploaded_file(uploaded_file)
            st.session_state.df = df_loaded
            st.session_state.current_file = uploaded_file.name

            if dataset_is_clean(df_loaded):
                st.session_state.active_tab = 1
                st.session_state.insights_generated = True
            else:
                st.session_state.active_tab = 0
                st.session_state.insights_generated = False

            st.session_state.cleaning_done = False
            st.session_state.tab_switched = False

            # Persist to disk
            save_to_cache(df_loaded, uploaded_file.name, st.session_state.insights_generated)

            st.rerun()
        st.info("Supported formats: CSV, XLSX", icon=None)

# MAIN DASHBOARD
else:
    render_sidebar()

    df = st.session_state.df

    # Page title
    title_col, badge_col = st.columns([5, 1])
    with title_col:
        st.markdown("## AI Data Analyst")
    with badge_col:
        st.markdown(
            f"<div style='text-align:right; color:#94a3b8; font-size:0.8rem; padding-top:0.6rem;'>"
            f"📄 {st.session_state.get('current_file', 'dataset')}</div>",
            unsafe_allow_html=True
        )

    # Dataset overview metrics
    st.subheader("🗂️ Dataset Overview")
    m1, m2, m3 = st.columns(3)
    m1.metric("Rows",       len(df))
    m2.metric("Missing",    int(df.isnull().sum().sum()))
    m3.metric("Duplicates", int(df.duplicated().sum()))

    # Data Preview
    st.dataframe(df.head(10), use_container_width=True)

    st.divider()

    # Tab LOgic
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = 0

    visuals_ready = insights_are_ready()

    tab1, tab2, tab3 = st.tabs([
        "🧹 Data Cleaning",
        "📈 Visuals & Insights",
        "💬 Ask About Your Dataset"
    ])

    # Tab 1: Cleaning
    with tab1:
        render_cleaning_agent(df)

    # Tab 2: Visuals & Insights 
    with tab2:
        st.subheader("📈 Visualization")
        render_visualization(df)

        st.divider()

        st.subheader("💡 AI Insights")
        render_insights(df)

    # Tab 3: Ask About Your Dataset
    with tab3:
        render_qa(df)