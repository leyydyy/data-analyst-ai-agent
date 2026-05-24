import streamlit as st
import os
import pickle
from utils.file_io import to_csv_bytes, make_export_filename

CACHE_PATH = "/tmp/analyst_agent_cache.pkl"

def clear_cache():
    if os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)

def render_sidebar():
    st.sidebar.title("Dataset Controls")

    # DATASET INFO
    st.sidebar.success(
        f"Loaded: {st.session_state.get('current_file', 'Dataset')}"
    )

    if st.session_state.get("cleaned"):
        st.sidebar.success("Dataset has been cleaned!")

    st.sidebar.divider()

    # LOAD DIFFERENT DATASET
    if st.sidebar.button("Load Different Dataset", use_container_width=True):
        clear_cache()  # ← clears disk so data isn't restored on rerun
        keys_to_clear = [
            "df",
            "uploaded_file",
            "current_file",
            "cleaned",
            "pending_plan",
            "change_log",
            "original_df",
            "auto_plan_generated",
            "auto_insights",
            "data_quality",
            "insights_generated",
            "active_tab",
            "cleaning_done",
            "tab_switched",
            "saved_insights",
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    # EXPORT
    if st.session_state.get("cleaned"):
        st.sidebar.divider()
        st.sidebar.subheader("Export Data")

        export_df = st.session_state.df

        csv_bytes = to_csv_bytes(export_df)
        filename  = make_export_filename(
            st.session_state.get("current_file") or "data.csv"
        )

        st.sidebar.download_button(
            label="Download Cleaned Dataset",
            data=csv_bytes,
            file_name=filename,
            mime="text/csv",
            use_container_width=True,
        )


def on_dataset_loaded(quality: str):
    """
    Call this right after uploading a file and assessing its quality.
    Sets auto_insights so render_insights() triggers on the next render.

    Usage:
        quality = assess_data_quality(df)  # your existing quality check
        st.session_state.data_quality = quality
        on_dataset_loaded(quality)
    """
    st.session_state.data_quality = quality

    st.session_state.auto_insights = True