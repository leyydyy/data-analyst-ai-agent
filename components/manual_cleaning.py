import streamlit as st
from utils.cleaning import (
    standardize_categorical,
    fix_data_types,
    detect_outliers,
)


def render_manual_cleaning(df):
    """Draw the manual cleaning buttons. Each button mutates session state."""
    st.subheader("🛠 Manual Cleaning Tools")

    colA, colB, colC, colD = st.columns(4)

    # --- Standardize Categories ---
    if colA.button("Standardize Categories"):
        log = []
        st.session_state.df = standardize_categorical(df, log=log)
        st.session_state.change_log += (
            log or ["✏️ Standardized categorical columns (no changes needed)"]
        )
        st.session_state.cleaned = True
        st.success("Categorical values standardized.")
        st.rerun()

    # --- Fix Data Types ---
    if colB.button("Fix Data Types"):
        log = []
        st.session_state.df = fix_data_types(df, log=log)
        st.session_state.change_log += (
            log or ["🔧 Checked data types (no changes needed)"]
        )
        st.session_state.cleaned = True
        st.success("Data types fixed.")
        st.rerun()

    # --- Remove Duplicates ---
    if colC.button("Remove Duplicates"):
        before  = len(df)
        deduped = df.drop_duplicates()
        removed = before - len(deduped)
        st.session_state.df = deduped
        st.session_state.change_log.append(f"🔁 Removed {removed} duplicate rows")
        st.session_state.cleaned = True
        st.success(f"Removed {removed} duplicate rows.")
        st.rerun()

    # --- Smart Fill Missing ---
    if colD.button("Smart Fill Missing"):
        temp = df.copy()
        log  = []
        for col in temp.columns:
            missing = int(temp[col].isnull().sum())
            if missing == 0:
                continue
            if hasattr(temp[col], "dtype") and temp[col].dtype.kind in "iufc":  # numeric
                val = temp[col].median()
                temp[col] = temp[col].fillna(val)
                log.append(
                    f"📊 Filled {missing} missing in **{col}** with median ({val:.2f})"
                )
            else:
                mode_vals = temp[col].mode()
                if not mode_vals.empty:
                    temp[col] = temp[col].fillna(mode_vals[0])
                    log.append(
                        f"📊 Filled {missing} missing in **{col}** "
                        f"with mode ('{mode_vals[0]}')"
                    )
        st.session_state.df = temp
        st.session_state.change_log += log
        st.session_state.cleaned = True
        st.success("Missing values filled.")
        st.rerun()


def render_outlier_detection(df):
    """Display an IQR-based outlier summary for all numeric columns."""
    st.subheader("⚠️ Outlier Detection")
    outliers = detect_outliers(df)
    if outliers:
        st.warning("Potential outliers detected:")
        st.write(outliers)
    else:
        st.success("No major outliers detected.")


def render_change_log():
    """Display the cumulative audit trail of all cleaning actions."""
    if st.session_state.change_log:
        st.divider()
        st.subheader("📋 What Changed")
        st.caption("Full audit trail of every cleaning action applied to your data.")
        for entry in st.session_state.change_log:
            st.markdown(f"- {entry}")