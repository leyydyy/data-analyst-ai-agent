import json
import streamlit as st
from config import client
from utils.data_summary import build_dataset_summary
from utils.cleaning import apply_cleaning_plan

# ---------------------------
# Icons
# ---------------------------
_ACTION_ICONS = {
    "drop_column":            "🗑️",
    "fill_median":            "📊",
    "fill_mean":              "📊",
    "fill_mode":              "📊",
    "remove_duplicates":      "🔁",
    "fix_dtypes":             "🔧",
    "standardize_categories": "✏️",
}

# ---------------------------
# Prompts
# ---------------------------
_SYSTEM_PROMPT = (
    "You are a precise data cleaning agent. Output only valid JSON — "
    "no markdown, no explanation, no code fences."
)

_USER_PROMPT_TEMPLATE = """
Analyze the following dataset metadata and sample data and return a structured JSON cleaning plan.

Dataset Info:
{summary}

Return ONLY a valid JSON object in this exact format:
{{
  "summary": "Brief 2-sentence overview of dataset quality",
  "confidence": 0-100,
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

# ---------------------------
# Messiness Detection
# ---------------------------
def is_dataset_messy(df):
    total_cells = df.shape[0] * df.shape[1]

    missing_ratio = df.isnull().sum().sum() / total_cells if total_cells > 0 else 0
    duplicate_ratio = df.duplicated().sum() / len(df) if len(df) > 0 else 0
    object_ratio = len(df.select_dtypes(include="object").columns) / len(df.columns)

    return (
        missing_ratio > 0.1 or
        duplicate_ratio > 0.05 or
        object_ratio > 0.6
    )

# ---------------------------
# Generate Plan
# ---------------------------
def _generate_plan(df):

    summary = build_dataset_summary(df)
    sample = df.head(5).to_dict()

    prompt = _USER_PROMPT_TEMPLATE.format(
        summary=json.dumps({
            "metadata": summary,
            "sample": sample
        }, indent=2, default=str)
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )

        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        plan = json.loads(raw)

        # Validation
        if not isinstance(plan, dict):
            st.error("Invalid plan format returned by AI.")
            return

        if "steps" not in plan or not isinstance(plan["steps"], list):
            st.error("Invalid steps format in AI response.")
            return

        st.session_state.pending_plan = plan

    except json.JSONDecodeError as e:
        st.error(f"AI returned malformed JSON — try again. ({e})")
    except Exception as e:
        st.error(f"AI Error: {e}")

# ---------------------------
# Render Plan Review
# ---------------------------
def _render_plan_review(df):

    plan  = st.session_state.pending_plan
    steps = plan.get("steps", [])

    st.info(f"**AI Assessment:** {plan.get('summary', '')}")

    if "confidence" in plan:
        st.metric("AI Confidence", f"{plan['confidence']}%")

    st.write("**Proposed Steps:**")

    for i, step in enumerate(steps, start=1):
        icon   = _ACTION_ICONS.get(step.get("action"), "•")
        target = (
            f" on **{step.get('column')}**"
            if step.get("column") and step.get("column") != "all"
            else ""
        )
        st.markdown(f"{i}. {icon} `{step.get('action')}`{target} — {step.get('reason')}")

    st.write("")
    col_approve, col_reject = st.columns(2)

    # APPROVE
    with col_approve:
        if st.button("✅ Approve & Execute Plan", type="primary"):
            try:
                with st.spinner("Applying cleaning steps..."):

                    # Save original for evaluation
                    st.session_state.original_df = df.copy()

                    cleaned_df, change_log = apply_cleaning_plan(df, steps)

                st.session_state.df           = cleaned_df
                st.session_state.change_log   = change_log
                st.session_state.cleaned      = True
                st.session_state.pending_plan = None

                st.success("Cleaning plan executed successfully!")

                # Explainability
                if change_log:
                    st.write("### 🧾 What Changed")
                    for change in change_log:
                        st.write(f"- {change}")

                st.rerun()

            except Exception as e:
                st.error(f"Processing Error: {e}")

    # REJECT
    with col_reject:
        if st.button("❌ Reject Plan"):
            st.session_state.pending_plan = None
            st.info("Plan discarded. You can regenerate or use manual tools.")
            st.rerun()

# ---------------------------
# Main Render Function
# ---------------------------
def render_cleaning_agent(df):

    st.subheader("🤖 AI Cleaning Agent")
    st.caption(
        "The agent detects messy data, proposes a plan, and executes only after your approval."
    )

    # Init flags
    if "auto_plan_generated" not in st.session_state:
        st.session_state.auto_plan_generated = False

    # =========================
    # AUTO GENERATE IF MESSY
    # =========================
    if (
        not st.session_state.get("cleaned", False)
        and not st.session_state.get("pending_plan")
        and not st.session_state.auto_plan_generated
    ):
        if is_dataset_messy(df):
            st.info("⚠️ Dataset appears messy. AI is generating a cleaning plan...")

            with st.spinner("Analyzing dataset automatically..."):
                _generate_plan(df)

            st.session_state.auto_plan_generated = True
            st.rerun()

    # =========================
    # MANUAL GENERATE
    # =========================
    if not st.session_state.get("cleaned", False):

        if st.button("🔍 Generate Cleaning Plan Manually"):
            with st.spinner("Analyzing your dataset..."):
                _generate_plan(df)

    else:
        st.success("✅ Dataset already cleaned using AI agent.")

        col1, col2 = st.columns(2)

        if col1.button("🔄 Re-run AI Cleaning"):
            st.session_state.cleaned = False
            st.session_state.pending_plan = None
            st.session_state.auto_plan_generated = False
            st.info("You can generate a new plan.")
            st.rerun()

        if col2.button("📜 View Last Changes"):
            if st.session_state.get("change_log"):
                for change in st.session_state.change_log:
                    st.write(f"- {change}")

    # =========================
    # SHOW PLAN
    # =========================
    if st.session_state.get("pending_plan"):
        _render_plan_review(df)