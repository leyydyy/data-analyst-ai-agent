import streamlit as st
from utils.file_io import load_uploaded_file, to_csv_bytes, make_export_filename


def render_sidebar():
    """
    Draw the sidebar and handle all file-upload / export logic.
    Mutates st.session_state directly so the rest of the app reacts.
    """
    st.sidebar.title("📁 Upload your data")

    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV or Excel", type=["csv", "xlsx", "xls"]
    )

    # Only re-parse when a new file is uploaded
    if uploaded_file and st.session_state.current_file != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        st.session_state.cleaned      = False
        st.session_state.pending_plan = None
        st.session_state.change_log   = []
        st.session_state.df           = load_uploaded_file(uploaded_file)

    # Export section — visible only after at least one cleaning action
    if st.session_state.df is not None and st.session_state.cleaned:
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