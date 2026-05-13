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
    layout="wide",
    page_icon="📊"
)

# ---------------------------
# GLOBAL STYLES
# ---------------------------
st.markdown("""
<style>
    /* ── Hide default Streamlit top padding ── */
    .block-container { padding-top: 1.5rem !important; }

    /* ── Upload hero ── */
    .upload-hero {
        text-align: center;
        padding: 4rem 2rem;
    }
    .upload-hero h1 {
        font-size: 2rem;
        font-weight: 800;
        color: #e2e8f0;
        margin-bottom: 0.5rem;
    }
    .upload-hero p {
        color: #94a3b8;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# INIT SESSION
# ---------------------------
init_session()

# ---------------------------
# UPLOAD SCREEN
# ---------------------------
if st.session_state.get("df") is None:
    st.markdown("""
    <div class="upload-hero">
        <h1>📊 AI Data Analyst Agent</h1>
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
            st.session_state.df = load_uploaded_file(uploaded_file)
            st.session_state.current_file = uploaded_file.name
            st.rerun()
        st.info("📁 Supported formats: CSV, XLSX", icon=None)

# ---------------------------
# MAIN DASHBOARD
# ---------------------------
else:
    render_sidebar()

    # ── Page title + file badge ──
    title_col, badge_col = st.columns([5, 1])
    with title_col:
        st.markdown("## 📊 AI Data Analyst Agent")
    with badge_col:
        st.markdown(
            f"<div style='text-align:right; color:#94a3b8; font-size:0.8rem; padding-top:0.6rem;'>"
            f"📄 {st.session_state.get('current_file', 'dataset')}</div>",
            unsafe_allow_html=True
        )

    # ── Dataset overview ──
    st.subheader("🗂️ Dataset Overview")

    m1, m2, m3 = st.columns(3)
    m1.metric("Rows",       len(st.session_state.df))
    m2.metric("Missing",    int(st.session_state.df.isnull().sum().sum()))
    m3.metric("Duplicates", int(st.session_state.df.duplicated().sum()))

    with st.expander("Preview data", expanded=False):
        st.dataframe(st.session_state.df.head(10), use_container_width=True)

    st.divider()

    # ── Cleaning Agent ──
    st.subheader("🧹 AI Cleaning Agent")
    render_cleaning_agent(st.session_state.df)

    st.divider()

    # ── Visualization ──
    st.subheader("📈 Visualization")
    render_visualization(st.session_state.df)

    st.divider()

    # ── AI Insights ──
    st.subheader("💡 AI Insights")
    render_insights(st.session_state.df)

    st.divider()

    # ── Ask Your Data ──
    st.subheader("💬 Ask About Your Dataset")
    render_qa(st.session_state.df)