import streamlit as st
from config import init_session
from components.sidebar import render_sidebar
from components.cleaning_agent import render_cleaning_agent
from components.insights import render_insights
from components.visualization import render_visualization
from components.qa import render_qa

# ---------------------------
# INIT
# ---------------------------
init_session()

st.set_page_config(page_title="AI Data Analyst Agent", layout="wide")

st.title("📊 AI Data Analyst Agent")

# ---------------------------
# SIDEBAR (UPLOAD)
# ---------------------------
render_sidebar()
df = st.session_state.df

# ---------------------------
# MAIN FLOW
# ---------------------------
if df is not None:

    st.session_state.df = df

    # ---------------------------
    # DATA PREVIEW
    # ---------------------------
    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", len(df))
    col2.metric("Missing", df.isnull().sum().sum())
    col3.metric("Duplicates", df.duplicated().sum())

    # CLEANING AGENT
    render_cleaning_agent(st.session_state.df)

    # INSIGHTS (AUTO + MANUAL)
    render_insights(st.session_state.df)

    # VISUALIZATION
    render_visualization(st.session_state.df)

    # Q&A
    render_qa(st.session_state.df)

else:
    st.info("Upload a dataset to begin.")