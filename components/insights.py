import json
import streamlit as st
from config import client
from utils.data_summary import build_dataset_summary

_SYSTEM_PROMPT = (
    "You are a senior data analyst. Be specific — always reference "
    "actual column names and values from the data provided."
)

_USER_PROMPT_TEMPLATE = """
Based on the structured dataset summary below, provide:
1. A 2-sentence overview of the data quality and content.
2. Three specific, actionable insights (reference column names and values).
3. Two concrete recommendations for next analytical steps.

Dataset Summary:
{summary}

Format your response with clear headers. Avoid generic statements.
"""


def render_insights(df):
    """Draw the AI Insights section for the given DataFrame."""
    st.subheader("🧠 AI Insights")

    if st.button("Generate Insights"):
        with st.spinner("Generating insights..."):
            summary = build_dataset_summary(df)
            prompt  = _USER_PROMPT_TEMPLATE.format(
                summary=json.dumps(summary, indent=2, default=str)
            )
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                )
                st.write(response.choices[0].message.content)
            except Exception as e:
                st.error(f"AI Error: {e}")