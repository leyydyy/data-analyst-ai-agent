"""
app.py
------
Entry point for the AI Data Analyst Agent.

Responsibilities:
  - Page config
  - Session state initialization
  - Orchestrating all UI components in order

All logic lives in utils/ and components/ — this file is intentionally thin.
Run with:  streamlit run app.py
"""

import streamlit as st
from config import init_session_state
from components.sidebar import render_sidebar
from components.cleaning_agent import render_cleaning_agent
from components.manual_cleaning import (
    render_manual_cleaning,
    render_outlier_detection,
    render_change_log,
)
from components.insights import render_insights
from components.visualization import render_visualization
from components.qa import render_qa

# ---------------------------
# Page Config (must be first Streamlit call)
# ---------------------------
st.set_page_config(page_title="AI Data Analyst Agent", layout="wide")

# ---------------------------
# Session State
# ---------------------------
init_session_state()

# ---------------------------
# Sidebar (upload + export)
# ---------------------------
render_sidebar()

# ---------------------------
# Main Content
# ---------------------------
if st.session_state.df is not None:
    df = st.session_state.df

    st.title("📊 AI Data Analyst Agent")

    # Data preview + health metrics
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📄 Data Preview")
        st.dataframe(df.head())
    with col2:
        st.subheader("📊 Data Health")
        st.metric("Rows",          len(df))
        st.metric("Duplicates",    df.duplicated().sum())
        st.metric("Missing Cells", df.isnull().sum().sum())

    # Agentic cleaning flow
    st.divider()
    render_cleaning_agent(df)

    # Explainability log
    render_change_log()

    # Manual cleaning tools + outlier detection
    st.divider()
    render_manual_cleaning(df)
    render_outlier_detection(df)

    # AI insights
    st.divider()
    render_insights(df)

    # Visualization explorer
    st.divider()
    render_visualization(df)

    # Q&A
    st.divider()
    render_qa(df)

else:
    st.title("📊 AI Data Analyst Agent")
    st.info("Upload a CSV or Excel file to begin.")