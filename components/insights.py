import streamlit as st
from config import client
from utils.data_summary import build_dataset_summary

def render_insights(df):

    quality = st.session_state.get("data_quality", "unknown")
    cleaned = st.session_state.get("cleaned", False)

    # WARNINGS
    st.info(
        "AI-generated insights may contain errors. "
        "Always validate results before making decisions."
    )

    # AUTO INSIGHTS
    if st.session_state.get("auto_insights", False):
        if quality == "unclean":
            st.warning(
                "These insights are based on raw, unclean data and may be "
                "inaccurate. Consider approving the cleaning plan above for "
                "more reliable results."
            )
        st.session_state.auto_insights = False
        with st.spinner("AI is analyzing your dataset"):
            _generate_insights(df)

    if st.button("Regenerate Insights"):
        with st.spinner("AI is analyzing your dataset"):
            _generate_insights(df)


def _generate_insights(df, quality="unknown"):
    try:
        summary = build_dataset_summary(df)

        quality_note = (
            "Note: This dataset has some quality issues. Where data is incomplete, "
            "derive insights from available records and flag approximations with '(approximate)'."
            if quality == "unclean" else ""
        )

        prompt = f"""
        Analyze the dataset summary and provide insights about the data itself — 
        its patterns, distributions, trends, and what the numbers reveal.
        {quality_note}

        Your response must follow these strict rules:
        1. Every insight MUST include at least one specific statistic (%, count, ratio, or named value).
           Example: "64% of sales came from the West region", "Age ranges from 18 to 72 with a mean of 34"
        2. Every recommendation MUST be actionable — start with a verb and reference a specific column or metric.
           Example: "Prioritize the West region in Q3 campaigns given its 64% sales share"
        3. Never write about data quality, nulls, duplicates, or missing values as the main topic of an insight.
           Focus on what the data is telling you about the subject matter.

        Use clear Markdown formatting with headings and bullet points.

        ## EXAMPLE
        Input Dataset Summary:
        {{"shape": {{"rows": 500, "columns": 3}}, "columns": ["Region", "Sales", "Profit"],
          "missing": {{"Profit": 90}}}}

        Output:
        ### Key Trends
        * **West Region Dominates Sales:** The West region accounts for the largest share of total sales volume across all regions.
        * **Profit Margins Vary by Region:** Profit figures differ significantly between regions despite similar sales numbers.
        * **Sales Concentration Risk:** The top 2 regions contribute the majority of revenue, leaving overall performance sensitive to regional shifts.

        ### Insights
        1. **Regional Sales Concentration:** The top 2 regions contribute approximately 68% of total sales, indicating a concentration risk.
        2. **Sales-to-Profit Gap:** Some regions show high sales but comparatively low profit, suggesting varying cost structures or pricing strategies.
        3. **Consistent Sales Volume:** Sales figures are present across all 500 rows, providing a complete picture of revenue distribution.

        ### Recommendations
        1. **Expand presence in the top-performing regions** to capitalize on the 68% sales concentration already observed there.
        2. **Audit pricing or cost structures in low-profit regions** to identify where margins can be improved relative to their sales volume.
        3. **Diversify revenue sources across underperforming regions** to reduce dependency on the top 2 regions driving most of the sales.

        ---
        Now, analyze the following dataset:
        {summary}
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior data analyst. You output clean, structured markdown. "
                        "Your insights are always about what the data reveals — patterns, trends, and distributions. "
                        "You always back every insight with a specific statistic. "
                        "Your recommendations are actionable and tied directly to the insights."
                    )
                },
                {"role": "user", "content": prompt}
            ]
        )

        st.write(response.choices[0].message.content)

    except Exception as e:
        st.error(f"AI Error: {e}")