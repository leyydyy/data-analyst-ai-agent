import streamlit as st
from config import client
from utils.data_summary import build_dataset_summary


def render_insights(df):

    quality = st.session_state.get("data_quality", "unknown")
    cleaned = st.session_state.get("cleaned", False)

    # WARNINGS
    st.info(
        "ℹ️ AI-generated insights may contain errors. "
        "Always validate results before making decisions."
    )

    # AUTO INSIGHTS
    if st.session_state.get("auto_insights", False):

        if quality == "unclean":
            st.warning(
                "⚠️ These insights are based on raw, unclean data and may be "
                "inaccurate. Consider approving the cleaning plan above for "
                "more reliable results."
            )
        else:
            st.success("Generating insights from data")

        _generate_insights(df)

        st.session_state.auto_insights = False

    # MANUAL BUTTON
    if st.button("Regenerate Insights"):
        _generate_insights(df)


def _generate_insights(df):

    try:
        summary = build_dataset_summary(df)

        prompt = f"""
        Analyze dataset and provide:
        - Key trends
        - 3 insights
        - 2 recommendations

        Dataset:
        {summary}
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a senior data analyst."},
                {"role": "user", "content": prompt}
            ]
        )

        st.write(response.choices[0].message.content)

    except Exception as e:
        st.error(f"AI Error: {e}")