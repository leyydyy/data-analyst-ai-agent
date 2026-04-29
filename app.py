import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI
from dotenv import load_dotenv
import os

# ---------------------------
# Setup
# ---------------------------
load_dotenv()
client = OpenAI()

st.set_page_config(page_title="AI Data Analyst Agent", layout="wide")

# ---------------------------
# Session State
# ---------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None
if "cleaned" not in st.session_state:
    st.session_state.cleaned = False

# ---------------------------
# Helper Functions
# ---------------------------

def sanitize_data(df):
    junk = ["", " ", "NULL", "null", "?", "N/A", "n/a", "NA"]
    return df.replace(junk, pd.NA)

def standardize_categorical(df):
    df = df.copy()
    for col in df.select_dtypes(include='object').columns:
        # Standardize casing and strip whitespace
        df[col] = df[col].astype(str).str.strip().str.lower()
        # Common boolean/categorical normalization
        df[col] = df[col].replace({
            "yes": "yes", "y": "yes", "1": "yes", "true": "yes",
            "no": "no", "n": "no", "0": "no", "false": "no",
            "x": pd.NA, "unknown": pd.NA
        })
    return df

def fix_data_types(df):
    df = df.copy()
    for col in df.columns:
        try:
            # Try numeric first
            df[col] = pd.to_numeric(df[col], errors='ignore')
        except:
            pass
        try:
            # Try datetime
            df[col] = pd.to_datetime(df[col], errors='ignore')
        except:
            pass
    return df

def detect_outliers(df):
    report = {}
    for col in df.select_dtypes(include='number').columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
        if not outliers.empty:
            report[col] = len(outliers)
    return report

# ---------------------------
# Sidebar Upload & Export
# ---------------------------
st.sidebar.title("📁 Data Management")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel", type=["csv", "xlsx", "xls"]
)

if uploaded_file:
    if st.session_state.current_file != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        # Reset cleaning state for new file
        st.session_state.cleaned = False 

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        df = sanitize_data(df)
        st.session_state.df = df

# Export Section
if st.session_state.df is not None and st.session_state.cleaned:
    st.sidebar.divider()
    st.sidebar.subheader("📥 Export Data")
    csv = st.session_state.df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="Download Cleaned CSV",
        data=csv,
        file_name=f"cleaned_{st.session_state.current_file if st.session_state.current_file.endswith('.csv') else st.session_state.current_file + '.csv'}",
        mime="text/csv",
    )

# ---------------------------
# Main App
# ---------------------------
if st.session_state.df is not None:

    df = st.session_state.df

    st.title("📊 AI Data Analyst Agent")

    # ---------------------------
    # Preview + Metrics
    # ---------------------------
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📄 Data Preview")
        st.dataframe(df.head())

    with col2:
        st.subheader("📊 Data Health")
        st.metric("Rows", len(df))
        st.metric("Duplicates", df.duplicated().sum())
        st.metric("Missing Cells", df.isnull().sum().sum())

    # ---------------------------
    # AI CLEANING STRATEGY
    # ---------------------------
    st.divider()
    st.subheader("🤖 AI Cleaning Strategy")

    if st.button("Analyze Cleaning Strategy (AI)"):
        with st.spinner("Analyzing dataset..."):
            summary = df.describe(include='all').fillna("").to_string()
            missing = df.isnull().sum().to_string()

            prompt = f"""
            Analyze dataset and recommend cleaning strategy:

            Dataset summary:
            {summary}

            Missing values:
            {missing}

            Identify:
            - Missing handling
            - Categorical inconsistencies
            - Duplicates
            - Data types
            - Outliers

            Provide step-by-step plan.
            """

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a data analyst."},
                    {"role": "user", "content": prompt}
                ]
            )

            st.markdown(response.choices[0].message.content)

    # ---------------------------
    # APPLY AI CLEANING
    # ---------------------------
    if st.button("Apply AI Cleaning (Auto)"):
        temp = df.copy()

        temp = standardize_categorical(temp)
        temp = fix_data_types(temp)

        for col in temp.columns:
            missing_ratio = temp[col].isnull().mean()

            if missing_ratio < 0.05:
                if pd.api.types.is_numeric_dtype(temp[col]):
                    temp[col] = temp[col].fillna(temp[col].mean())
                else:
                    temp[col] = temp[col].fillna(temp[col].mode()[0])

            elif missing_ratio < 0.4:
                if pd.api.types.is_numeric_dtype(temp[col]):
                    temp[col] = temp[col].fillna(temp[col].median())
                else:
                    temp[col] = temp[col].fillna(temp[col].mode()[0])

            elif missing_ratio > 0.6:
                if temp[col].nunique() < 5:
                    temp = temp.drop(columns=[col])

        temp = temp.drop_duplicates()

        st.session_state.df = temp
        st.session_state.cleaned = True
        st.success("AI cleaning applied successfully. Download available in sidebar.")
        st.rerun()

    # ---------------------------
    # MANUAL CLEANING
    # ---------------------------
    st.divider()
    st.subheader("🛠 Manual Cleaning Tools")

    colA, colB, colC, colD = st.columns(4)

    if colA.button("Standardize Categories"):
        st.session_state.df = standardize_categorical(df)
        st.session_state.cleaned = True
        st.success("Categorical standardized")
        st.rerun()

    if colB.button("Fix Data Types"):
        st.session_state.df = fix_data_types(df)
        st.session_state.cleaned = True
        st.success("Types fixed")
        st.rerun()

    if colC.button("Remove Duplicates"):
        before = len(df)
        df = df.drop_duplicates()
        st.session_state.df = df
        st.session_state.cleaned = True
        st.success(f"Removed {before - len(df)} duplicates")
        st.rerun()

    if colD.button("Smart Fill Missing"):
        temp = df.copy()
        for col in temp.columns:
            if temp[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(temp[col]):
                    temp[col] = temp[col].fillna(temp[col].median())
                else:
                    temp[col] = temp[col].fillna(temp[col].mode()[0])
        st.session_state.df = temp
        st.session_state.cleaned = True
        st.success("Missing values filled")
        st.rerun()

    # ---------------------------
    # Outlier Detection
    # ---------------------------
    st.subheader("⚠ Outlier Detection")
    outliers = detect_outliers(df)

    if outliers:
        st.warning("Potential outliers:")
        st.write(outliers)
    else:
        st.success("No major outliers detected")

    # ---------------------------
    # AI INSIGHTS
    # ---------------------------
    st.divider()
    st.subheader("🧠 AI Insights")

    if st.button("Generate Insights"):
        summary = df.describe(include='all').to_string()

        prompt = f"""
        Analyze dataset and provide:
        - Key trends
        - 3 insights
        - 2 recommendations

        Dataset summary:
        {summary}
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a senior data analyst."},
                {"role": "user", "content": prompt}
            ]
        )

        st.write(response.choices[0].message.content)

    # ---------------------------
    # Visualization (FIXED)
    # ---------------------------
    st.divider()
    st.subheader("📊 Visualization")

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    if len(numeric_cols) >= 2:
        x = st.selectbox("X-axis", numeric_cols)
        y = st.selectbox("Y-axis", numeric_cols, index=1)

        fig = px.scatter(df, x=x, y=y, trendline="ols")
        st.plotly_chart(fig, use_container_width=True)

    elif len(numeric_cols) == 1:
        st.info("Only one numeric column → showing distribution")
        col = numeric_cols[0]
        fig = px.histogram(df, x=col)
        st.plotly_chart(fig, use_container_width=True)

    elif len(categorical_cols) > 0:
        st.info("No numeric data → showing categorical distribution")
        col = categorical_cols[0]
        fig = px.bar(df[col].value_counts().reset_index(),
                     x='index', y=col)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No suitable data for visualization.")

    # ---------------------------
    # Q&A
    # ---------------------------
    st.divider()
    st.subheader("❓ Ask Questions")

    question = st.text_input("Ask about your data")

    if question:
        summary = df.describe(include='all').to_string()

        prompt = f"""
        Dataset summary:
        {summary}

        Question: {question}
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Answer using dataset only."},
                {"role": "user", "content": prompt}
            ]
        )

        st.write(response.choices[0].message.content)

else:
    st.title("📊 AI Data Analyst Agent")
    st.info("Upload a dataset to begin.")