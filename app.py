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

st.title("📊 AI Data Analyst Agent")

# ---------------------------
# GET DATAFRAME
# ---------------------------
df = st.session_state.get("df", None)

# ---------------------------
# SHOW UPLOADER IN MAIN SCREEN
# ONLY WHEN NO DATASET
# ---------------------------
if df is None:

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

    st.subheader("📄 Data Preview")

    st.dataframe(df.head())

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", len(df))
    col2.metric("Missing", int(df.isnull().sum().sum()))
    col3.metric("Duplicates", int(df.duplicated().sum()))

    # CLEANING AGENT
    render_cleaning_agent(df)

    # INSIGHTS
    render_insights(df)

    # VISUALIZATION
    render_visualization(df)

    # Q&A
    render_qa(df)