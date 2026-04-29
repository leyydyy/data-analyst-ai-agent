import json
import streamlit as st
from config import client
from utils.data_summary import build_dataset_summary
from utils.cleaning import apply_cleaning_plan

# Maps action keys to display emojis for the review panel
_ACTION_ICONS = {
    "drop_column":            "🗑️",
    "fill_median":            "📊",
    "fill_mean":              "📊",
    "fill_mode":              "📊",
    "remove_duplicates":      "🔁",
    "fix_dtypes":             "🔧",
    "standardize_categories": "✏️",
}

_SYSTEM_PROMPT = (
    "You are a precise data cleaning agent. Output only valid JSON — "
    "no markdown, no explanation, no code fences."
)

_USER_PROMPT_TEMPLATE = """
Analyze the following dataset metadata and return a structured JSON cleaning plan.

Dataset Info:
{summary}

Return ONLY a valid JSON object in this exact format:
{{
  "summary": "Brief 2-sentence overview of dataset quality",
  "steps": [
    {{
      "action": "fill_median" | "fill_mean" | "fill_mode" | "drop_column" | "remove_duplicates" | "fix_dtypes" | "standardize_categories",
      "column": "column_name or 'all'",
      "reason": "plain English explanation"
    }}
  ]
}}

Rules:
- Only include steps that are genuinely necessary.
- Use drop_column only if missing > 60% or the column has no analytical value.
- fill_median for skewed numerics; fill_mean for normally distributed ones.
- fill_mode for categorical columns with missing values.
- Always include remove_duplicates if duplicates > 0.
- Include fix_dtypes and standardize_categories when relevant.
"""


def _generate_plan(df):
    """Call the AI and store the parsed plan in session state."""
    summary = build_dataset_summary(df)
    prompt  = _USER_PROMPT_TEMPLATE.format(
        summary=json.dumps(summary, indent=2, default=str)
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        )
        raw  = response.choices[0].message.content.strip()
        raw  = raw.replace("```json", "").replace("```", "").strip()
        plan = json.loads(raw)
        st.session_state.pending_plan = plan

    except json.JSONDecodeError as e:
        st.error(f"AI returned malformed JSON — try again. ({e})")
    except Exception as e:
        st.error(f"AI Error: {e}")


def _render_plan_review(df):
    """Show the pending plan and Approve / Reject controls."""
    plan  = st.session_state.pending_plan
    steps = plan.get("steps", [])

    st.info(f"**AI Assessment:** {plan.get('summary', '')}")
    st.write("**Proposed Steps:**")

    for i, step in enumerate(steps, start=1):
        icon   = _ACTION_ICONS.get(step["action"], "•")
        target = (
            f" on **{step['column']}**"
            if step.get("column") and step["column"] != "all"
            else ""
        )
        st.markdown(f"{i}. {icon} `{step['action']}`{target} — {step['reason']}")

    st.write("")
    col_approve, col_reject = st.columns(2)

    with col_approve:
        if st.button("✅ Approve & Execute Plan", type="primary"):
            with st.spinner("Applying cleaning steps..."):
                cleaned_df, change_log = apply_cleaning_plan(df, steps)
            st.session_state.df           = cleaned_df
            st.session_state.change_log   = change_log
            st.session_state.cleaned      = True
            st.session_state.pending_plan = None
            st.success("Cleaning plan executed successfully!")
            st.rerun()

    with col_reject:
        if st.button("❌ Reject Plan"):
            st.session_state.pending_plan = None
            st.info("Plan discarded. You can regenerate or use the manual tools below.")
            st.rerun()


def render_cleaning_agent(df):
    """
    Main entry point — call this from app.py.
    Renders the full agentic cleaning section for the given DataFrame.
    """
    st.subheader("🤖 AI Cleaning Agent")
    st.caption(
        "The agent proposes a cleaning plan. You review and approve before anything changes."
    )

    if st.button("🔍 Generate Cleaning Plan"):
        with st.spinner("Analyzing your dataset..."):
            _generate_plan(df)

    if st.session_state.pending_plan:
        _render_plan_review(df)