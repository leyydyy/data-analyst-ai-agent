import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI
from dotenv import load_dotenv
import os

# ---------------------------
# Setup & API Configuration
# ---------------------------
load_dotenv()
client = OpenAI()

st.set_page_config(page_title="Pro AI Data Analyst", layout="wide")

# Initialize Session States
if "df" not in st.session_state:
    st.session_state.df = None
if "cleaned" not in st.session_state:
    st.session_state.cleaned = False
if "current_file" not in st.session_state:
    st.session_state.current_file = None

# ---------------------------
# ADVANCED SYSTEM PROMPTS
# ---------------------------
SYSTEM_PERSONA = """
You are a Senior Data Scientist and IT Project Consultant. 
Your goal is to provide high-level, actionable insights. 
When analyzing data:
1. Prioritize technical accuracy and statistical significance.
2. Look for operational bottlenecks or optimization opportunities.
3. Keep your tone professional, concise, and structured using Markdown.
"""

# ---------------------------
# SIDEBAR - File Upload
# ---------------------------
st.sidebar.title("🛠 Controls")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

# Reset logic: If a new file is uploaded, clear the old session data
if uploaded_file:
    if st.session_state.current_file != uploaded_file.name:
        st.session_state.df = None
        st.session_state.cleaned = False
        st.session_state.current_file = uploaded_file.name

    if st.session_state.df is None:
        try:
            if uploaded_file.name.endswith(".csv"):
                raw_df = pd.read_csv(uploaded_file)
            else:
                raw_df = pd.read_excel(uploaded_file)
            
            # --- CRITICAL: DATA SANitization ON LOAD ---
            # Convert common "dirty" strings to actual NaN so metrics are accurate
            junk_values = ["", " ", "NULL", "null", "?", "N/A", "n/a", "NA"]
            raw_df = raw_df.replace(junk_values, pd.NA)
            
            st.session_state.df = raw_df
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

# ---------------------------
# MAIN INTERFACE
# ---------------------------
if st.session_state.df is not None:
    df = st.session_state.df

    st.title("📊 Pro AI Data Analyst Agent")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📄 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

    with col2:
        st.subheader("🔍 Data Health Metrics")
        # Calculate fresh metrics every time
        current_rows = len(df)
        total_missing = df.isnull().sum().sum()
        duplicate_count = df.duplicated().sum()
        
        st.metric("Total Rows", current_rows)
        st.metric("Duplicate Rows", duplicate_count)
        st.metric("Missing Cells (NaN)", total_missing, 
                  delta="- Needs Cleaning" if total_missing > 0 else "Clean", 
                  delta_color="inverse")

    # ---------------------------
    # STEP 1: AI CLEANING ADVICE
    # ---------------------------
    st.divider()
    st.subheader("🤖 AI Data Sanitization Strategy")
    
    if st.button("Analyze Data Integrity"):
        with st.spinner("Agent is inspecting schema..."):
            summary_stats = df.describe(include='all').fillna("N/A").to_string()
            cleaning_prompt = f"""
            Perform a data integrity audit on this dataset:
            
            SUMMARY STATS:
            {summary_stats}
            
            MISSING VALUES PER COLUMN:
            {df.isnull().sum().to_string()}
            
            Provide a 3-step 'Sanitization Strategy'. 
            Identify if any columns should be dropped. 
            Mention if any numeric columns look like they contain outliers.
            """
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PERSONA},
                    {"role": "user", "content": cleaning_prompt}
                ]
            )
            st.markdown(response.choices[0].message.content)

    # ---------------------------
    # STEP 2: STRATEGIC CLEANING (Analyst-Led)
    # ---------------------------
    st.divider()
    st.subheader("🛠 Strategic Cleaning Center")

    if st.session_state.df.isnull().sum().sum() > 0:
        st.warning("Action Required: Missing data detected. Choose a strategy below.")
        
        # AI Strategy Advice
        if st.button("Get AI Cleaning Strategy"):
            with st.spinner("Agent is evaluating data distribution..."):
                null_report = df.isnull().sum()[df.isnull().sum() > 0].to_string()
                advice_prompt = f"""
                As a Senior Data Analyst, evaluate these missing values:
                {null_report}

                1. Analyze the risk of bias for each column.
                2. RECOMMEND one specific Option (A, B, or C) as the best statistical choice.
                3. Explain WHY that option is the most scientifically sound for an IT student project.
                """
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": SYSTEM_PERSONA},
                              {"role": "user", "content": advice_prompt}]
                )
                st.info(response.choices[0].message.content)

        # The Human Choice
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        if col_btn1.button("Option A: Drop All Missing Rows"):
            st.session_state.df = df.dropna()
            st.session_state.cleaned = True
            st.success("Analyst Choice: Rows with missing values removed.")
            st.rerun()

        if col_btn2.button("Option B: Smart Fill (Median/Mode)"):
            temp_df = df.advice_pcopy()
            for col in temp_df.columns:
                if temp_df[col].isnull().sum() > 0:
                    if pd.api.types.is_numeric_dtype(temp_df[col]):
                        temp_df[col] = temp_df[col].fillna(temp_df[col].median())
                    else:
                        temp_df[col] = temp_df[col].fillna(temp_df[col].mode()[0])
            st.session_state.df = temp_df
            st.session_state.cleaned = True
            st.success("Analyst Choice: Imputation applied via Median/Mode.")
            st.rerun()
            
        if col_btn3.button("Option C: Keep as is (Analyze 'as-is')"):
            st.session_state.cleaned = True
            st.success("Analyst Choice: Proceeding with missing values.")
            st.rerun()
    else:
        st.success("✅ Data is pristine. No cleaning actions required.")

    # ---------------------------
    # STEP 3: ANALYTICS & VISUALS
    # ---------------------------
    st.divider()
    st.subheader("🧠 Executive Insights")

    if st.button("Generate Strategic Analysis"):
        with st.spinner("Synthesizing trends..."):
            analysis_prompt = f"""
            Analyze this dataset for executive-level decision making:
            DATASET OVERVIEW:
            {df.describe(include='all').to_string()}
            
            Deliver:
            1. 'The Big Picture'
            2. 'Critical Patterns'
            3. 'Actionable Recommendations'
            """
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PERSONA},
                    {"role": "user", "content": analysis_prompt}
                ]
            )
            st.info(response.choices[0].message.content)

    # Visualizer
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if len(numeric_cols) >= 2:
        st.subheader("📊 Interactive Distribution Explorer")
        c1, c2 = st.columns(2)
        x_var = c1.selectbox("Select X-Axis", numeric_cols, index=0)
        y_var = c2.selectbox("Select Y-Axis", numeric_cols, index=min(1, len(numeric_cols)-1))
        
        fig = px.scatter(df, x=x_var, y=y_var, trendline="ols", 
                         title=f"Correlation: {x_var} vs {y_var}",
                         template="plotly_dark", color_discrete_sequence=['#00CC96'])
        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # STEP 4: Q&A
    # ---------------------------
    st.divider()
    st.subheader("💬 Ask the Data Analyst")
    user_query = st.text_input("Ask a specific question about these results")

    if user_query:
        with st.spinner("Consulting the data..."):
            qa_prompt = f"Data Summary:\n{df.describe(include='all').to_string()}\n\nQuestion: {user_query}"
            answer = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PERSONA},
                    {"role": "user", "content": qa_prompt}
                ]
            )
            st.chat_message("assistant").write(answer.choices[0].message.content)

    # Sidebar Download
    if st.session_state.cleaned:
        st.sidebar.divider()
        csv = st.session_state.df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button("📥 Export Cleaned CSV", data=csv, file_name="analyst_export.csv", mime="text/csv")

else:
    st.title("📊 AI Data Analyst Agent")
    st.warning("Please upload a dataset in the sidebar to begin.")