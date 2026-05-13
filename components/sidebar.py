import streamlit as st
from utils.file_io import to_csv_bytes, make_export_filename

def render_sidebar():
    st.sidebar.title("Dataset Controls")

    # ---------------------------
    # DATASET INFO
    # ---------------------------
    st.sidebar.success(
        f"Loaded: {st.session_state.get('current_file', 'Dataset')}"
    )

    # Show a cleaned badge if applicable
    if st.session_state.get("cleaned"):
        st.sidebar.success("Dataset has been cleaned!")

    st.sidebar.divider()

    # ---------------------------
    # RESET BUTTON
    # ---------------------------
    if st.sidebar.button(
        "Load Different Dataset",
        use_container_width=True
    ):
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
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    # ---------------------------
    # EXPORT
    # ---------------------------
    if st.session_state.get("cleaned"):
        st.sidebar.divider()
        st.sidebar.subheader("Export Data")

        # ✅ Always export from st.session_state.df (the live, cleaned df)
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

        # Show a quick summary of what was cleaned
        if st.session_state.get("change_log"):
            with st.sidebar.expander("View Change Log"):
                for entry in st.session_state.change_log:
                    st.write(f"- {entry}")