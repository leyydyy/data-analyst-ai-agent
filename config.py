"""
config.py
---------
Centralizes app-wide configuration:
- OpenAI client initialization
- Streamlit session state defaults
"""

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Single shared OpenAI client — import this everywhere instead of re-creating it
client = OpenAI()

# All session state keys and their initial values in one place.
# Call init_session_state() once at the top of app.py.
SESSION_DEFAULTS = {
    "df": None,            # The active DataFrame
    "current_file": None,  # Filename of the uploaded file
    "cleaned": False,      # Whether any cleaning has been applied
    "pending_plan": None,  # AI-generated cleaning plan awaiting user approval
    "change_log": [],      # Explainability audit trail
}


def init_session_state():
    """Initialize all session state keys that are not yet set."""
    import streamlit as st
    for key, val in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = val