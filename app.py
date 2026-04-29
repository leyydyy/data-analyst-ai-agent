import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI
from dotenv import load_dotenv
import os

# API CONFIG
load_dotenv()
client = OpenAI()

st.set_page_config(page_title="Pro AI Data Analyst", layout="wide")

# SESSION STATE INITILIZATION
if "cleaned" not in st.session_state:
    st.session_state.cleaned = False
if "df" not in st.session_state:
    st.session_state.df = None

# SYSTEM PROMPT
SYSTEM_PERSONA = """
You are a Senior Data Scientist and IT Project Consultant. 
Your goal is to provide high-level, actionable insights. 
When analyzing data:
1. Prioritize technical accuracy and statistical significance.
2. Look for operational bottlenecks or optimization opportunities.
3. Keep your tone professional, concise, and structured using Markdown.
"""

# FILE UPLOAD AND EXPORT
st.sidebar.title("🛠 Controls")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

if uploaded_file:
    if st.session_state.df is None:
        try:
            if uploaded_file.name.endswith(".csv"):
                st.session_state.df = pd.read_csv(uploaded_file)
            else:
                st.session_state.df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

    df = st.session_state.df

    # MAIN INTERFACE
    st.title("AI Data Analyst Agent")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📄 Raw Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

    with col2:
        st.subheader("🔍 Data Health Metrics")
        st.metric("Total Rows", len(df))
        st.metric("Duplicates", df.duplicated().sum())
        missing = df.isnull().sum().sum()
        st.metric("Total Missing Cells", missing, delta="- Needs Attention" if missing > 0 else "Clean", delta_color="inverse")

    # DATA CLEANSING
    st.divider()
    st.subheader("🤖 AI Data Sanitization Strategy")
    
    if st.button("Analyze Data Integrity"):
        with st.spinner("Agent is inspecting schema..."):
            summary_stats = df.describe(include='all').fillna("N/A").to_string()
            
            # Prompt for cleaning the dataset
            cleaning_prompt = f"""
            Perform a data integrity audit on this dataset:
            
            SUMMARY STATS:
            {summary_stats}
            
            MISSING VALUES:
            {df.isnull().sum().to_string()}
            
            Provide a 3-step 'Sanitization Strategy'. 
            Identify if any columns should be dropped due to low variance or high missingness.
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

    # APPLY CLEANSING
    if st.button("🚀 Execute Auto-Cleaning"):
        temp_df = df.copy()
        # Drop columns with more than 50% missing data
        cols_to_drop = [col for col in temp_df.columns if temp_df[col].isnull().mean() > 0.5]
        temp_df.drop(columns=cols_to_drop, inplace=True)
        
        # Simple imputation
        for col in temp_df.columns:
            if temp_df[col].isnull().sum() > 0:
                if temp_df[col].dtype in ['float64', 'int64']:
                    temp_df[col] = temp_df[col].fillna(temp_df[col].median())
                else:
                    temp_df[col] = temp_df[col].fillna(temp_df[col].mode()[0])
        
        temp_df.drop_duplicates(inplace=True)
        st.session_state.df = temp_df
        st.session_state.cleaned = True
        st.success("Data sanitized! Schema updated.")
        st.rerun()

    # ANALYTICS AND VISUALS
    st.divider()
    st.subheader("🧠 Executive Insights")

    if st.button("Generate Strategic Analysis"):
        with st.spinner("Synthesizing trends..."):
            # BI-focused prompt
            analysis_prompt = f"""
            Analyze this dataset for executive-level decision making:
            
            DATASET OVERVIEW:
            {df.describe(include='all').to_string()}
            
            Deliver:
            1. 'The Big Picture' (High-level summary)
            2. 'Critical Patterns' (Correlations or anomalies)
            3. 'Predictive Thoughts' (What might happen next?)
            4. 'Actionable Recommendations' (2 specific steps for the business)
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
        x_var = c1.selectbox("Select Independent Variable (X)", numeric_cols, index=0)
        y_var = c2.selectbox("Select Dependent Variable (Y)", numeric_cols, index=1)
        
        fig = px.scatter(df, x=x_var, y=y_var, trendline="ols", 
                         title=f"Correlation: {x_var} vs {y_var}",
                         template="plotly_dark", color_discrete_sequence=['#00CC96'])
        st.plotly_chart(fig, use_container_width=True)

    # ASK QUESTIONS
    st.divider()
    st.subheader("💬 Ask the Data Analyst")
    user_query = st.text_input("Ask a specific question (e.g., 'Which category has the highest variance?')")

    if user_query:
        with st.spinner("Consulting the data..."):
            # The Prompt: Focused on context-awareness
            qa_prompt = f"""
            The user has a question about the following dataset summary:
            {df.describe(include='all').to_string()}
            
            USER QUESTION: {user_query}
            
            If the question is mathematical, explain the steps to find the answer. 
            If the data summary is insufficient to answer, state clearly what additional data is needed.
            """
            
            answer = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PERSONA},
                    {"role": "user", "content": qa_prompt}
                ]
            )
            st.chat_message("assistant").write(answer.choices[0].message.content)

    # DOWNLOAD CLEANED
    if st.session_state.cleaned:
        st.sidebar.divider()
        csv = st.session_state.df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button("📥 Export Cleaned CSV", data=csv, file_name="analyst_export.csv", mime="text/csv")

else:
    st.title("📊 AI Data Analyst Agent")
    st.warning("Welcome! Please upload a dataset in the sidebar to begin your analysis.")