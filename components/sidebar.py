import streamlit as st
from utils.file_io import load_uploaded_file, to_csv_bytes, make_export_filename


def render_sidebar():
    st.sidebar.title("📁 Upload your data")

    # ---------------------------
    # RESET BUTTON
    # ---------------------------
    if st.session_state.get("df") is not None:
        if st.sidebar.button("🔄 Load Different Dataset", use_container_width=True):
            # Clear all state
            keys_to_reset = [
                "df", "current_file", "cleaned", "pending_plan",
                "change_log", "original_df", "auto_plan_generated",
                "auto_insights", "data_quality"
            ]
            for key in keys_to_reset:
                st.session_state[key] = None if "df" in key or "file" in key or "plan" in key or "log" in key else False
            st.session_state.data_quality = "unknown"
            st.session_state.change_log = []
            st.rerun()

        st.sidebar.divider()

    # ---------------------------
    # FILE UPLOADER
    # ---------------------------
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV or Excel", type=["csv", "xlsx", "xls"]
    )

    if uploaded_file and st.session_state.get("current_file") != uploaded_file.name:
        st.session_state.current_file        = uploaded_file.name
        st.session_state.cleaned             = False
        st.session_state.pending_plan        = None
        st.session_state.change_log          = []
        st.session_state.df                  = load_uploaded_file(uploaded_file)
        st.session_state.data_quality        = "unknown"
        st.session_state.auto_plan_generated = False
        st.session_state.auto_insights       = False
        st.session_state.original_df         = None
        st.rerun()

    # ---------------------------
    # EXPORT
    # ---------------------------
    if st.session_state.get("df") is not None and st.session_state.get("cleaned"):
        st.sidebar.divider()
        st.sidebar.subheader("📥 Export Data")
        csv_bytes = to_csv_bytes(st.session_state.df)
        filename  = make_export_filename(st.session_state.current_file or "data.csv")
        st.sidebar.download_button(
            label="Download Cleaned CSV",
            data=csv_bytes,
            file_name=filename,
            mime="text/csv",
        )