import json
import streamlit as st
from config import client
from utils.data_summary import build_dataset_summary

_SYSTEM_PROMPT = (
    "You are a data analyst assistant. Your ONLY job is to answer factual questions "
    "about the dataset provided — such as summarizing values, identifying trends, "
    "comparing figures, or explaining what the data shows.\n\n"
    "You must REFUSE any request that asks you to:\n"
    "- Write, generate, or explain code in any language\n"
    "- Create scripts, functions, or programs\n"
    "- Perform tasks unrelated to interpreting the dataset\n"
    "- Give instructions on how to do something programmatically\n\n"
    "If the user asks for code or anything outside of dataset analysis, respond with:\n"
    "'I can only answer questions about the dataset. I cannot perform tasks outside of data interpretation.'\n\n"
    "If the answer cannot be determined from the data, say so clearly. Be concise and specific."
)

_USER_PROMPT_TEMPLATE = """
Dataset Summary:
{summary}

Question: {question}
"""


def render_qa(df):
    """Draw the Q&A section with conversation memory."""

    # Display previous conversation history
    for msg in st.session_state.qa_history:
        if msg["role"] != "system": # Dont print the hidden system prompt
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    #Capture new user input
    question = st.chat_input("Ask anything about your dataset")

    if question:
        # Display the new question immediately
        with st.chat_message("user"):
            st.markdown(question)

        with st.spinner("Thinking..."):
            summary = build_dataset_summary(df)
            
            #Initialize  history with the System Prompt if its empty
            if not st.session_state.qa_history:
                st.session_state.qa_history.append({
                    "role": "system", 
                    "content": _SYSTEM_PROMPT + "\n\nDataset Summary:\n" + json.dumps(summary, indent=2, default=str)
                })

            #Append the new user question to the history
            st.session_state.qa_history.append({"role": "user", "content": question})

            try:
                #Pass the wntire history to the AI
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=st.session_state.qa_history,
                )
                
                answer = response.choices[0].message.content
                
                # Display AI response
                with st.chat_message("assistant"):
                    st.markdown(answer)
                
                #Save AI response to history
                st.session_state.qa_history.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"AI Error: {e}")