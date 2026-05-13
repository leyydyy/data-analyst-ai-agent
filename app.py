import streamlit as st
from config import init_session
from components.sidebar import render_sidebar
from components.cleaning_agent import render_cleaning_agent
from components.insights import render_insights
from components.visualization import render_visualization
from components.qa import render_qa

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="AI Data Analyst Agent",
    layout="wide"
)

# ---------------------------
# INIT SESSION
# ---------------------------
init_session()
st.title("AI Data Analyst Agent")

# ---------------------------
# SHOW UPLOADER IN MAIN SCREEN
# ONLY WHEN NO DATASET
# ---------------------------
if st.session_state.get("df") is None:
    st.markdown("## Upload Your Dataset")
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel File",
        type=["csv", "xlsx"],
        key="uploaded_file"
    )
    if uploaded_file is not None:
        from utils.file_io import load_uploaded_file
        st.session_state.df = load_uploaded_file(uploaded_file)
        st.session_state.current_file = uploaded_file.name
        st.rerun()
    st.info("Upload a dataset to begin.")

# ---------------------------
# MAIN DASHBOARD
# ---------------------------
else:
    # show sidebar only after upload
    render_sidebar()

    # Always read LIVE from session state — never use a stale local variable.
    # Each component (especially cleaning_agent) may update st.session_state.df,
    # so we re-read it from session state before passing it to the next component.

    st.subheader("Data Preview")
    st.dataframe(st.session_state.df.head())

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows",       len(st.session_state.df))
    col2.metric("Missing",    int(st.session_state.df.isnull().sum().sum()))
    col3.metric("Duplicates", int(st.session_state.df.duplicated().sum()))

    # CLEANING AGENT — may overwrite st.session_state.df
    render_cleaning_agent(st.session_state.df)

    # Re-read after cleaning so downstream components get the cleaned df
    render_insights(st.session_state.df)

    render_visualization(st.session_state.df)

    render_qa(st.session_state.df)