import json
import streamlit as st
from config import client
from utils.data_summary import build_dataset_summary
from utils.cleaning import apply_cleaning_plan, _ALLOWED_ACTIONS

# Icons
_ACTION_ICONS = {
    "drop_column":            "🗑️",
    "fill_median":            "📊",
    "fill_mean":              "📊",
    "fill_mode":              "📊",
    "fill_constant":          "📝",
    "remove_duplicates":      "🔁",
    "fix_dtypes":             "🔧",
    "standardize_categories": "✏️",
    "standardize_dates":      "📅",
}

# Prompts — Plan generation
_SYSTEM_PROMPT = (
    "You are a precise data cleaning agent. Output only valid JSON — "
    "no markdown, no explanation, no code fences."
)

_USER_PROMPT_TEMPLATE = """
Analyze the dataset and return a structured JSON cleaning plan.

Dataset:
{summary}

The "action" field in every step should ideally be one of these built-in actions:
  - "drop_column"            → remove an entire column
  - "fill_median"            → fill numeric nulls with column median
  - "fill_mean"              → fill numeric nulls with column mean
  - "fill_mode"              → fill nulls with most frequent value
  - "fill_constant"          → fill nulls with a fixed value (add "value" key)
  - "remove_duplicates"      → remove duplicate rows
  - "fix_dtypes"             → cast column to correct dtype (add "dtype" key, e.g. "datetime64", "float", "int", "str")
  - "standardize_categories" → normalise inconsistent text values in a category column
  - "standardize_dates"      → parse and standardise inconsistent date formats to YYYY-MM-DD

If NONE of the built-in actions fit, you may use a short descriptive action name
(e.g. "remove_special_chars", "cap_outliers"). These will trigger code generation.
Prefer built-in actions when possible.

Return ONLY JSON:
{{
  "summary": "Brief dataset quality overview",
  "confidence": 0-100,
  "steps": [
    {{
      "action": "<built-in action or short custom name>",
      "column": "<column name, or 'all' for row-level ops>",
      "reason": "<one sentence explanation>",
      "value":  "<optional: used by fill_constant>",
      "dtype":  "<optional: used by fix_dtypes>"
    }}
  ]
}}
"""

# Prompts — Code-gen fallback
_CODEGEN_SYSTEM = (
    "You are a pandas code generation expert. "
    "Output only valid Python code — no markdown, no explanation, no code fences. "
    "The code will be executed with exec(). "
    "You have access to `df` (a pandas DataFrame) and `pd` (pandas). "
    "You MUST assign the result back to `df`. "
    "Do NOT import anything. Do NOT use os, sys, open, eval, exec, or any file I/O."
)

_CODEGEN_USER_TEMPLATE = """
Write a single pandas code block to perform this cleaning step on the DataFrame `df`.

Step details:
  action : {action}
  column : {column}
  reason : {reason}

DataFrame info:
  columns : {columns}
  dtypes  : {dtypes}
  sample  : {sample}

Rules:
- Assign result back to `df` (e.g. df['col'] = ... or df = df.drop(...))
- No imports, no print statements, no comments
- Keep it minimal — one logical operation only
"""


# Code-gen fallback function
def _request_codegen(step: dict, df) -> str | None:
    """
    Ask the AI to write a pandas snippet for a step whose action is not
    in the built-in list. Returns the code string, or None on failure.
    """
    import pandas as pd

    prompt = _CODEGEN_USER_TEMPLATE.format(
        action  = step.get("action", ""),
        column  = step.get("column", ""),
        reason  = step.get("reason", ""),
        columns = df.columns.tolist(),
        dtypes  = {col: str(dt) for col, dt in df.dtypes.items()},
        sample  = df.head(3).fillna("").to_dict(orient="records"),
    )

    try:
        response = client.chat.completions.create(
            model       = "gpt-4o-mini",
            temperature = 0.1,       # low temp → deterministic, safe code
            max_tokens  = 300,       # snippets are short; cap cost
            messages    = [
                {"role": "system", "content": _CODEGEN_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
        )
        code = response.choices[0].message.content.strip()
        # Strip accidental markdown fences
        code = code.replace("```python", "").replace("```", "").strip()
        return code if code else None

    except Exception as e:
        # Non-fatal — caller will log the skip
        return None


# Messiness Detection
def is_dataset_messy(df):
    total_cells     = df.shape[0] * df.shape[1]
    missing_ratio   = df.isnull().sum().sum() / total_cells if total_cells > 0 else 0
    duplicate_ratio = df.duplicated().sum() / len(df) if len(df) > 0 else 0
    object_ratio    = len(df.select_dtypes(include="object").columns) / len(df.columns)
    missing_count   = df.isnull().sum().sum()
    duplicate_count = df.duplicated().sum()

    return (
        missing_ratio   > 0.01 or
        missing_count   > 5    or
        duplicate_ratio > 0.01 or
        duplicate_count > 0    or
        object_ratio    > 0.6
    )


# Generate Plan
def _generate_plan(df):
    summary = build_dataset_summary(df)

    prompt = _USER_PROMPT_TEMPLATE.format(
        summary=json.dumps({
            "metadata": summary,
            "sample":   df.head(5).to_dict()
        }, indent=2, default=str)
    )

    try:
        response = client.chat.completions.create(
            model       = "gpt-4o-mini",
            temperature = 0.2,
            messages    = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        )

        raw  = response.choices[0].message.content.strip()
        raw  = raw.replace("```json", "").replace("```", "").strip()
        plan = json.loads(raw)

        if not isinstance(plan, dict) or "steps" not in plan:
            st.error("Invalid plan format returned by AI.")
            return

        # Split steps into built-in vs custom so the UI can label them
        for step in plan["steps"]:
            action = step.get("action", "").strip().lower()
            step["_is_custom"] = action not in _ALLOWED_ACTIONS

        st.session_state.pending_plan = plan

    except Exception as e:
        st.error(f"AI Error: {e}")


# Render Plan Review
def _render_plan_review(df):
    plan  = st.session_state.pending_plan
    steps = plan.get("steps", [])

    st.info(f"**AI Assessment:** {plan.get('summary', '')}")

    if "confidence" in plan:
        st.metric("AI Confidence", f"{plan['confidence']}%")

    # Count how many steps need code-gen so user knows what to expect
    custom_count = sum(1 for s in steps if s.get("_is_custom"))
    if custom_count:
        st.info(
            f"{custom_count} step(s) use a custom action not in the built-in list. "
            "The AI will generate pandas code for these when you approve."
        )

    st.write("**Proposed Steps:**")

    for i, step in enumerate(steps, start=1):
        action     = step.get("action", "")
        icon       = _ACTION_ICONS.get(action, "🤖" if step.get("_is_custom") else "•")
        col        = step.get("column", "")
        target     = f" on **{col}**" if col and col != "all" else ""
        custom_tag = " *(custom — code gen)*" if step.get("_is_custom") else ""
        st.markdown(f"{i}. {icon} `{action}`{target}{custom_tag} — {step.get('reason')}")

    col1, col2 = st.columns(2)

    # APPROVE
    with col1:
        if st.button("✅ Approve & Execute Plan", type="primary"):
            with st.spinner("Applying cleaning…"):

                current_df = st.session_state.get("df", df)

                if st.session_state.get("original_df") is None:
                    st.session_state.original_df = current_df.copy()

                steps = plan.get("steps", [])

                if not steps:
                    st.warning("No cleaning steps to apply.")
                    st.session_state.pending_plan = None
                    st.rerun()

                # Pass codegen_fn so cleaning.py can call back for custom steps
                cleaned_df, change_log = apply_cleaning_plan(
                    current_df,
                    steps,
                    codegen_fn=_request_codegen,
                )

            st.session_state.df            = cleaned_df
            st.session_state.change_log    = change_log
            st.session_state.cleaned       = True
            st.session_state.pending_plan  = None
            st.session_state.data_quality  = "clean"
            st.session_state.auto_insights = True
            st.session_state.insights_generated = True

            st.success("Cleaning applied successfully!")

            if change_log:
                st.write("### 🧾 Changes Applied")
                for c in change_log:
                    st.write(f"- {c}")

            st.rerun()

    # REJECT
    with col2:
        if st.button("❌ Reject Plan"):
            st.session_state.pending_plan = None
            st.rerun()

# MAIN RENDER
def render_cleaning_agent(df):

    if "auto_plan_generated" not in st.session_state:
        st.session_state.auto_plan_generated = False

    # AUTO DETECT DATA QUALITY
    if st.session_state.data_quality == "unknown":
        if is_dataset_messy(df):
            st.session_state.data_quality  = "unclean"
            st.session_state.auto_insights = True
        else:
            st.session_state.data_quality  = "clean"
            st.session_state.auto_insights = True
            st.session_state.insights_generated = True

    # AUTO PLAN GENERATION
    if (
        st.session_state.data_quality == "unclean"
        and not st.session_state.get("pending_plan")
        and not st.session_state.auto_plan_generated
    ):
        st.info("⚠️ Dataset is messy. Generating cleaning plan")
        _generate_plan(df)
        st.session_state.auto_plan_generated = True
        st.rerun()

    # MANUAL GENERATE
    if st.session_state.data_quality == "unclean":
        if (
            not st.session_state.get("pending_plan")
            and st.session_state.get("auto_plan_generated")
        ):
            if st.button("Generate Cleaning Plan"):
                _generate_plan(df)

    else:
        st.success("Dataset appears clean.")

    # SHOW PLAN
    if st.session_state.get("pending_plan"):
        _render_plan_review(df)