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
Analyze the dataset and return a structured JSON cleaning plan.

Dataset:
{summary}

Return ONLY JSON:
{{
  "summary": "Brief dataset quality overview",
  "confidence": 0-100,
  "steps": [
    {{
      "action": "...",
      "column": "...",
      "reason": "..."
    }}
  ]
}}
"""

# ---------------------------
# Messiness Detection
# ---------------------------
def is_dataset_messy(df):
    total_cells = df.shape[0] * df.shape[1]

    missing_ratio = df.isnull().sum().sum() / total_cells if total_cells > 0 else 0
    duplicate_ratio = df.duplicated().sum() / len(df) if len(df) > 0 else 0
    object_ratio = len(df.select_dtypes(include="object").columns) / len(df.columns)

    # Absolute counts
    missing_count = df.isnull().sum().sum()
    duplicate_count = df.duplicated().sum()

    return (
        missing_ratio > 0.01 or      # lowered from 0.1
        missing_count > 5 or         # any meaningful missing values
        duplicate_ratio > 0.01 or    # lowered from 0.05
        duplicate_count > 0 or       # any duplicates at all
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

        if not isinstance(plan, dict) or "steps" not in plan:
            st.error("Invalid plan format returned by AI.")
            return

        st.session_state.pending_plan = plan

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
        icon = _ACTION_ICONS.get(step.get("action"), "•")
        col  = step.get("column")

        target = f" on **{col}**" if col and col != "all" else ""

        st.markdown(f"{i}. {icon} `{step.get('action')}`{target} — {step.get('reason')}")

    col1, col2 = st.columns(2)

    # APPROVE
    with col1:
        if st.button("✅ Approve & Execute Plan", type="primary"):

            with st.spinner("Applying cleaning..."):

                st.session_state.original_df = df.copy()

                cleaned_df, change_log = apply_cleaning_plan(df, steps)

            st.session_state.df = cleaned_df
            st.session_state.change_log = change_log
            st.session_state.cleaned = True
            st.session_state.pending_plan = None

            # ✅ KEY ADDITIONS
            st.session_state.data_quality = "clean"
            st.session_state.auto_insights = True

            st.success("Cleaning applied!")

            if change_log:
                st.write("### 🧾 Changes")
                for c in change_log:
                    st.write(f"- {c}")

            st.rerun()

    # REJECT
    with col2:
        if st.button("❌ Reject Plan"):
            st.session_state.pending_plan = None
            st.rerun()

# ---------------------------
# MAIN RENDER
# ---------------------------
def render_cleaning_agent(df):

    st.subheader("🤖 AI Cleaning Agent")

    if "auto_plan_generated" not in st.session_state:
        st.session_state.auto_plan_generated = False

    # ---------------------------
    # AUTO DETECT DATA QUALITY
    # ---------------------------
    if st.session_state.data_quality == "unknown": 

        if is_dataset_messy(df):
            st.session_state.data_quality = "unclean"
            st.session_state.auto_insights = True   # ✅ ADD THIS

        else:
            st.session_state.data_quality = "clean"
            st.session_state.auto_insights = True

    # ---------------------------
    # AUTO PLAN GENERATION
    # ---------------------------
    if (
        st.session_state.data_quality == "unclean"
        and not st.session_state.get("pending_plan")
        and not st.session_state.auto_plan_generated
    ):
        st.info("⚠️ Dataset is messy. Generating cleaning plan...")

        _generate_plan(df)

        st.session_state.auto_plan_generated = True
        st.rerun()

    # ---------------------------
    # MANUAL GENERATE
    # ---------------------------
    if st.session_state.data_quality == "unclean":

        # ✅ Only show button if no plan is pending AND auto-gen already ran
        if (
            not st.session_state.get("pending_plan")
            and st.session_state.get("auto_plan_generated")
        ):
            if st.button("🔍 Generate Cleaning Plan"):
                _generate_plan(df)

    else:
        st.success("✅ Dataset appears clean.")

    # ---------------------------
    # SHOW PLAN
    # ---------------------------
    if st.session_state.get("pending_plan"):
        _render_plan_review(df)