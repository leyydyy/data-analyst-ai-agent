import json
import streamlit as st
from config import client
from utils.data_summary import build_dataset_summary

_SYSTEM_PROMPT = (
    "You are a data analyst. Answer the user's question using ONLY "
    "the dataset information provided. If the answer cannot be determined "
    "from the data, say so clearly. Be concise and specific."
)

_USER_PROMPT_TEMPLATE = """
Dataset Summary:
{summary}

Question: {question}
"""


def render_qa(df):
    """Draw the Q&A section for the given DataFrame."""
    st.subheader("❓ Ask Questions About Your Data")

    question = st.text_input("Ask anything about your dataset")

    if question:
        with st.spinner("Thinking..."):
            summary = build_dataset_summary(df)
            prompt  = _USER_PROMPT_TEMPLATE.format(
                summary=json.dumps(summary, indent=2, default=str),
                question=question,
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