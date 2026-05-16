import streamlit as st
import plotly.express as px


def render_visualization(df):
    """Draw the chart explorer section for the given DataFrame."""

    # Disclaimer
    st.info(
        "📌 Charts are generated directly from your dataset using Python/Plotly. "
        "The AI is not involved here — what you see is exactly what the data contains. "
        "If a chart looks wrong, the source data may still need manual review."
    )

    # Column classification 
    numeric_cols     = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    date_cols        = df.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns.tolist()

    if not numeric_cols and not categorical_cols:
        st.warning("No plottable columns found in this dataset.")
        return

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

    # ── Correlation (Scatter) ────────────────────────────────────────────────
    if viz_type == "Correlation (Scatter)":
        if len(numeric_cols) >= 2:
            x = st.selectbox("X-axis (Numeric)", numeric_cols, key="scatter_x")
            y = st.selectbox(
                "Y-axis (Numeric)", numeric_cols,
                index=min(1, len(numeric_cols) - 1),
                key="scatter_y",
            )
            color = st.selectbox(
                "Color by (optional)",
                ["None"] + categorical_cols,
                key="scatter_color",
            )
            if x == y:
                st.warning("⚠️ Please select different columns for X and Y axes.")
            else:
                try:
                    fig = px.scatter(
                        df,
                        x=x,
                        y=y,
                        color=None if color == "None" else color,
                        trendline="ols",
                        template="plotly_dark",
                        title=f"{y} vs {x}",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Unable to generate chart: {e}")
        else:
            st.warning("Needs at least 2 numeric columns for a scatter plot.")

    # ── Comparison (Bar) ────────────────────────────────────────────────────
    elif viz_type == "Comparison (Bar)":
        if categorical_cols and numeric_cols:
            cat = st.selectbox("Category (X-axis)", categorical_cols, key="bar_cat")
            val = st.selectbox("Value (Y-axis)",    numeric_cols,     key="bar_val")
            agg = st.selectbox(
                "Aggregation",
                ["mean", "sum", "count", "median", "max", "min"],
                key="bar_agg",
            )
            try:
                grouped = (
                    df.groupby(cat)[val]
                    .agg(agg)
                    .reset_index()
                    .sort_values(val, ascending=False)
                )
                fig = px.bar(
                    grouped,
                    x=cat,
                    y=val,
                    template="plotly_dark",
                    title=f"{agg.capitalize()} of {val} by {cat}",
                    text_auto=".2f",
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Unable to generate chart: {e}")
        else:
            st.warning("Needs at least one categorical and one numeric column.")

    # ── Distribution (Histogram) ─────────────────────────────────────────────
    elif viz_type == "Distribution (Histogram)":
        all_cols = numeric_cols + categorical_cols
        if all_cols:
            col_sel = st.selectbox("Select Column", all_cols, key="hist_col")
            bins    = st.slider("Number of bins", 5, 100, 20, key="hist_bins") \
                      if col_sel in numeric_cols else None
            try:
                fig = px.histogram(
                    df,
                    x=col_sel,
                    nbins=bins,
                    template="plotly_dark",
                    title=f"Distribution of {col_sel}",
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Unable to generate chart: {e}")
        else:
            st.warning("No columns available for histogram.")

    # ── Trends (Line) ────────────────────────────────────────────────────────
    elif viz_type == "Trends (Line)":
        if date_cols and numeric_cols:
            d_col = st.selectbox("Date Column",      date_cols,    key="line_date")
            n_col = st.selectbox("Value to Track",   numeric_cols, key="line_val")
            color = st.selectbox(
                "Group by (optional)",
                ["None"] + categorical_cols,
                key="line_color",
            )
            try:
                fig = px.line(
                    df.sort_values(d_col),
                    x=d_col,
                    y=n_col,
                    color=None if color == "None" else color,
                    template="plotly_dark",
                    title=f"{n_col} over {d_col}",
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Unable to generate chart: {e}")
        else:
            st.info(
                "No datetime columns detected. "
                "Try **Fix Data Types** in the cleaning section first, "
                "or use **Standardize Dates** if your date column is stored as text."
            )