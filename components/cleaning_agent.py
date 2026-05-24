import json
import streamlit as st
from config import client
from utils.data_summary import build_dataset_summary
from utils.cleaning import apply_cleaning_plan, _ALLOWED_ACTIONS

_SYSTEM_PROMPT = (
    "You are a precise data cleaning agent. Output only valid JSON."
    "No markdown, no explanation, no code fences."
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

Return ONLY JSON in this exact shape:
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

## Example 1 — Mixed issues (nulls, duplicates, wrong dtype, messy categories)

Input dataset summary:
{{
  "shape": [500, 5],
  "columns": {{
    "Emp_ID":     {{"dtype": "int64",   "nulls": 0}},
    "Name":       {{"dtype": "object",  "nulls": 3}},
    "Department": {{"dtype": "object",  "nulls": 0,  "sample_values": ["HR", "hr", "Finance", "FINANCE"]}},
    "Salary":     {{"dtype": "object",  "nulls": 0,  "sample_values": ["50000", "60000", "N/A"]}},
    "JoinDate":   {{"dtype": "object",  "nulls": 0,  "sample_values": ["2020-01-15", "15/01/2020", "Jan 15 2020"]}}
  }},
  "duplicate_rows": 12
}}

Output:
{{
  "summary": "Dataset has 3 missing names, 12 duplicate rows, inconsistent department casing, Salary stored as text with invalid entries, and mixed date formats.",
  "confidence": 88,
  "steps": [
    {{"action": "fill_mode",              "column": "Name",       "reason": "Fill 3 missing names with the most frequent value as a safe placeholder."}},
    {{"action": "remove_duplicates",      "column": "all",        "reason": "Remove 12 duplicate rows to avoid skewed aggregations."}},
    {{"action": "standardize_categories", "column": "Department", "reason": "Normalize HR/hr/Finance/FINANCE to consistent casing."}},
    {{"action": "fix_dtypes",             "column": "Salary",     "reason": "Cast Salary from object to numeric; invalid entries like N/A will become NaN.", "dtype": "float"}},
    {{"action": "standardize_dates",      "column": "JoinDate",   "reason": "Unify mixed date formats to YYYY-MM-DD."}}
  ]
}}

## Example 2 — Mostly clean dataset

Input dataset summary:
{{
  "shape": [200, 3],
  "columns": {{
    "ProductID": {{"dtype": "int64",  "nulls": 0}},
    "Price":     {{"dtype": "float64","nulls": 0}},
    "Category":  {{"dtype": "object", "nulls": 0}}
  }},
  "duplicate_rows": 0
}}

Output:
{{
  "summary": "Dataset appears clean with no nulls, no duplicates, and correct dtypes.",
  "confidence": 97,
  "steps": []
}}

Notice: When the data is already clean, return an empty steps array — do not invent problems.
"""

_CODEGEN_SYSTEM = (
    "You are a pandas code generation expert. "
    "Output only valid Python code. No markdown, no explanation, no code fences. "
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

## Example 1 — Remove special characters from a text column

Step details:
  action : remove_special_chars
  column : Phone
  reason : Phone numbers contain dashes and parentheses that should be stripped.

Output:
df['Phone'] = df['Phone'].astype(str).str.replace(r'[^\\d]', '', regex=True)

## Example 2 — Cap outliers at the 99th percentile

Step details:
  action : cap_outliers
  column : Salary
  reason : A small number of extreme salary values are skewing the distribution.

Output:
df['Salary'] = df['Salary'].clip(upper=df['Salary'].quantile(0.99))

## Example 3 — Strip leading and trailing whitespace from a column

Step details:
  action : strip_whitespace
  column : City
  reason : City names have inconsistent leading/trailing spaces.

Output:
df['City'] = df['City'].str.strip()

Notice: Always assign back to df. One operation only. No imports or comments.
"""

def _request_codegen(step: dict, df) -> str | None:
    """
    Ask the AI to write a pandas snippet for a step whose action is not
    in the built-in list. Returns the code string, or None on failure.
    """
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
            temperature = 0.1,
            max_tokens  = 300,
            messages    = [
                {"role": "system", "content": _CODEGEN_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
        )
        code = response.choices[0].message.content.strip()
        code = code.replace("```python", "").replace("```", "").strip()
        return code if code else None

    except Exception:
        return None


def is_dataset_messy(df):
    total_rows = len(df)
    total_cells = df.shape[0] * df.shape[1]
    
    if total_rows == 0:
        return False
        
    missing_ratio   = df.isnull().sum().sum() / total_cells
    duplicate_ratio = df.duplicated().sum() / total_rows
    object_ratio    = len(df.select_dtypes(include="object").columns) / len(df.columns)
    missing_threshold = 0.01 if total_rows < 1000 else 0.05
    
    return (
        missing_ratio > missing_threshold or
        duplicate_ratio > 0.02 or
        object_ratio > 0.75
    )


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

        # Tag custom steps so the UI can label
        for step in plan["steps"]:
            action = step.get("action", "").strip().lower()
            step["_is_custom"] = action not in _ALLOWED_ACTIONS

        st.session_state.pending_plan = plan

    except Exception as e:
        st.error(f"AI Error: {e}")


def _render_plan_review(df):
    plan  = st.session_state.pending_plan
    steps = plan.get("steps", [])

    st.info(f"**AI Assessment:** {plan.get('summary', '')}")

    if "confidence" in plan:
        st.metric("AI Confidence", f"{plan['confidence']}%")

    custom_count = sum(1 for s in steps if s.get("_is_custom"))
    if custom_count:
        st.info(
            f"{custom_count} step(s) use a custom action not in the built-in list. "
            "The AI will generate pandas code for these when you approve."
        )

    st.write("**Proposed Steps:**")

    for i, step in enumerate(steps, start=1):
        action     = step.get("action", "")
        col        = step.get("column", "")
        target     = f" on **{col}**" if col and col != "all" else ""
        custom_tag = " *(custom — code gen)*" if step.get("_is_custom") else ""
        st.markdown(f"{i}. `{action}`{target}{custom_tag} — {step.get('reason')}")

    st.warning(
        "**Review carefully before applying.** This plan is AI-generated and may not be "
        "fully accurate. It could miss issues, make incorrect assumptions, or modify data "
        "in unintended ways. Verify each step against your dataset before approving."
    )

    col1, col2 = st.columns(2)

    # APPROVE
    with col1:
        if st.button("Approve & Execute Plan", type="primary"):
            with st.spinner("Applying cleaning…"):

                current_df = st.session_state.get("df", df)

                if st.session_state.get("original_df") is None:
                    st.session_state.original_df = current_df.copy()

                steps = plan.get("steps", [])

                if not steps:
                    st.warning("No cleaning steps to apply.")
                    st.session_state.pending_plan = None
                    st.rerun()

                cleaned_df, change_log = apply_cleaning_plan(
                    current_df,
                    steps,
                    codegen_fn=_request_codegen,
                )

            st.session_state.df                 = cleaned_df
            st.session_state.change_log         = change_log
            st.session_state.cleaned            = True
            st.session_state.pending_plan       = None
            st.session_state.data_quality       = "clean"
            st.session_state.auto_insights      = True
            st.session_state.insights_generated = True

            st.success("Cleaning applied successfully!")

            if change_log:
                st.write("### Changes Applied")
                for c in change_log:
                    st.write(f"- {c}")

            st.rerun()

    # REGENERATE
    with col2:
        if st.button("Regenerate Plan"):
            st.session_state.pending_plan = None
            st.session_state.auto_plan_generated = False
            with st.spinner("AI is generating a new cleaning plan…"):
                _generate_plan(df)
            st.rerun()


def render_cleaning_agent(df):

    if "auto_plan_generated" not in st.session_state:
        st.session_state.auto_plan_generated = False

    # Autodetect data quality
    if st.session_state.data_quality == "unknown":
        if is_dataset_messy(df):
            st.session_state.data_quality  = "unclean"
            st.session_state.auto_insights = True
        else:
            st.session_state.data_quality       = "clean"
            st.session_state.auto_insights      = True
            st.session_state.insights_generated = True

    # Auto generate plan on first load for unclean data
    if (
        st.session_state.data_quality == "unclean"
        and not st.session_state.get("pending_plan")
        and not st.session_state.auto_plan_generated
    ):
        with st.spinner("AI is generating a cleaning plan"):
            _generate_plan(df)
        st.session_state.auto_plan_generated = True
        st.rerun()

    # Manual re-generate button (shown after auto-plan was rejected)
    if st.session_state.data_quality == "unclean":
        if (
            not st.session_state.get("pending_plan")
            and st.session_state.get("auto_plan_generated")
        ):
            if st.button("Generate Cleaning Plan"):
                with st.spinner("AI is generating a cleaning plan…"):
                    _generate_plan(df)
    else:
        st.success("Dataset appears clean.")

    # Show plan review if one is pending
    if st.session_state.get("pending_plan"):
        _render_plan_review(df)