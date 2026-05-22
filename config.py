"""
config.py
---------
Centralizes app-wide configuration:
- OpenAI client initialization
- Streamlit session state defaults
"""

from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
client = OpenAI()

# Session Defaults
SESSION_DEFAULTS = {
    "df": None,                 # Active DataFrame
    "current_file": None,       # Uploaded filename
    "cleaned": False,           # Whether cleaning has been applied
    "pending_plan": None,       # AI cleaning plan
    "change_log": [],           # Explainability log
    "original_df": None,        # Before cleaning
    "auto_plan_generated": False,
    "auto_insights": False,
    "data_quality": "unknown",
    "insights_generated": False,
    "qa_history": [],
}

# Initialize Session State
def init_session_state():
    """Initialize all session state keys that are not yet set."""
    for key, val in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = val


def init_session():
    """
    Alias for compatibility with app.py.
    Prevents ImportError: cannot import name 'init_session'
    """
    init_session_state()