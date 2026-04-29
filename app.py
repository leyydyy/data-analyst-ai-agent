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

# ---------------------------
# Helper Functions
# ---------------------------

def sanitize_data(df):
    junk = ["", " ", "NULL", "null", "?", "N/A", "n/a", "NA"]
    return df.replace(junk, pd.NA)

def standardize_categorical(df):
    df = df.copy()
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip().str.lower()
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
            df[col] = pd.to_numeric(df[col])
        except:
            pass
        try:
            df[col] = pd.to_datetime(df[col])
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
# Sidebar Upload
# ---------------------------
st.sidebar.title("📁 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel", type=["csv", "xlsx", "xls"]
)

if uploaded_file:
    if st.session_state.current_file != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        df = sanitize_data(df)
        st.session_state.df = df

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
    # AI Data Analysis
    # ---------------------------
    st.divider()
    st.subheader("🤖 AI Data Assessment")

    if st.button("Analyze Dataset"):
        summary = df.describe(include='all').fillna("").to_string()

        prompt = f"""
        You are a data analyst.

        Analyze dataset and identify:
        - Missing data issues
        - Inconsistent categorical values
        - Possible outliers
        - Data type issues
        - Any suspicious values

        Dataset summary:
        {summary}

        Provide structured recommendations.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional data analyst."},
                {"role": "user", "content": prompt}
            ]
        )

        st.write(response.choices[0].message.content)

    # ---------------------------
    # Cleaning Tools
    # ---------------------------
    st.divider()
    st.subheader("🛠 Data Cleaning Tools")

    colA, colB, colC, colD = st.columns(4)

    if colA.button("Standardize Categories"):
        st.session_state.df = standardize_categorical(df)
        st.success("Categorical values standardized")
        st.rerun()

    if colB.button("Fix Data Types"):
        st.session_state.df = fix_data_types(df)
        st.success("Data types corrected")
        st.rerun()

    if colC.button("Remove Duplicates"):
        before = len(df)
        df = df.drop_duplicates()
        st.session_state.df = df
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
        st.success("Missing values filled")
        st.rerun()

    # ---------------------------
    # Outlier Detection
    # ---------------------------
    st.subheader("⚠ Outlier Detection")
    outliers = detect_outliers(df)

    if outliers:
        st.warning("Potential outliers found:")
        st.write(outliers)
    else:
        st.success("No major outliers detected")

    # ---------------------------
    # Insights
    # ---------------------------
    st.divider()
    st.subheader("🧠 AI Insights")

    if st.button("Generate Insights"):
        summary = df.describe(include='all').to_string()

        prompt = f"""
        Analyze this dataset and provide:
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
    # Visualization
    # ---------------------------
    st.divider()
    st.subheader("📊 Visualization")

    numeric_cols = df.select_dtypes(include='number').columns

    if len(numeric_cols) >= 2:
        x = st.selectbox("X-axis", numeric_cols)
        y = st.selectbox("Y-axis", numeric_cols, index=1)

        fig = px.scatter(df, x=x, y=y, trendline="ols")
        st.plotly_chart(fig, use_container_width=True)

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