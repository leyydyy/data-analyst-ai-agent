import streamlit as st
import plotly.express as px


def render_visualization(df):
    """Draw the chart explorer section for the given DataFrame."""
    st.subheader("📊 Interactive Visualization Explorer")

    numeric_cols     = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    date_cols        = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

    viz_type = st.radio(
        "Select Chart Type",
        [
            "Correlation (Scatter)",
            "Comparison (Bar)",
            "Distribution (Histogram)",
            "Trends (Line)",
        ],
        horizontal=True,
    )

    if viz_type == "Correlation (Scatter)":
        if len(numeric_cols) >= 2:
            x   = st.selectbox("X-axis (Numeric)", numeric_cols)
            y   = st.selectbox("Y-axis (Numeric)", numeric_cols, index=1)
            fig = px.scatter(df, x=x, y=y, trendline="ols", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Needs at least 2 numeric columns.")

    elif viz_type == "Comparison (Bar)":
        if categorical_cols and numeric_cols:
            cat    = st.selectbox("Category (X-axis)", categorical_cols)
            num    = st.selectbox("Value (Y-axis)", numeric_cols)
            df_agg = df.groupby(cat)[num].mean().reset_index()
            fig    = px.bar(
                df_agg, x=cat, y=num,
                title=f"Average {num} by {cat}",
                template="plotly_dark",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Needs at least 1 categorical and 1 numeric column.")

    elif viz_type == "Distribution (Histogram)":
        all_cols = numeric_cols + categorical_cols
        if all_cols:
            col_sel = st.selectbox("Select Column", all_cols)
            fig     = px.histogram(df, x=col_sel, template="plotly_dark", nbins=20)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No columns available for histogram.")

    elif viz_type == "Trends (Line)":
        if date_cols and numeric_cols:
            d_col = st.selectbox("Date Column", date_cols)
            n_col = st.selectbox("Value to Track", numeric_cols)
            fig   = px.line(
                df.sort_values(d_col), x=d_col, y=n_col,
                template="plotly_dark",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                "No date columns detected. "
                "Try **Fix Data Types** in the cleaning section first."
            )