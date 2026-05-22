import json
import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI

client = OpenAI()  # uses OPENAI_API_KEY from env

def _dataset_summary(df: pd.DataFrame) -> str:
    """Compact dataset description to send to the AI."""
    buf = []
    buf.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    buf.append("\nColumns and dtypes:")
    for col, dtype in df.dtypes.items():
        buf.append(f"  {col!r}: {dtype}")
    buf.append("\nSample (first 3 rows):")
    buf.append(df.head(3).to_string(index=False))
    buf.append("\nBasic stats:")
    buf.append(df.describe(include="all").to_string())
    return "\n".join(buf)

def _ask_ai_for_charts(df: pd.DataFrame) -> list[dict]:
    """
    Ask the AI which charts to render.
    Returns a list of chart spec dicts, e.g.:
      [
        {"type": "scatter",   "x": "age",    "y": "salary",   "color": "department", "title": "Salary vs Age"},
        {"type": "bar",       "cat": "region","val": "sales",  "agg": "sum",          "title": "Total Sales by Region"},
        {"type": "histogram", "col": "score", "bins": 20,                             "title": "Score Distribution"},
        {"type": "line",      "date": "date", "val": "revenue","color": "None",       "title": "Revenue Over Time"},
        {"type": "box",       "cat": "team",  "val": "score",                         "title": "Score Distribution by Team"},
        {"type": "pie",       "names": "category", "values": "amount",               "title": "Revenue Share by Category"},
      ]
    """
    summary = _dataset_summary(df)

    system_prompt = """You are a data visualization expert.
    Given a dataset summary, recommend 4-6 insightful charts.
    Reply ONLY with a valid JSON array. No markdown, no prose, no code fences.

    Each element must be one of these exact shapes:
    {"type":"scatter",   "x":"<num_col>",    "y":"<num_col>",    "color":"<cat_col or None>", "title":"..."}
    {"type":"bar",       "cat":"<cat_col>",  "val":"<num_col>",  "agg":"<mean|sum|count|median|max|min>", "title":"..."}
    {"type":"histogram", "col":"<col>",      "bins":20,                                        "title":"..."}
    {"type":"line",      "date":"<date_col>","val":"<num_col>",  "color":"<cat_col or None>",  "title":"..."}
    {"type":"box",       "cat":"<cat_col>",  "val":"<num_col>",                                "title":"..."}
    {"type":"pie",       "names":"<cat_col>","values":"<num_col>",                             "title":"..."}
    {"type":"pie_count", "names":"<cat_col>",                                                  "title":"..."}

    Rules:
    - Only reference columns that actually exist in the dataset.
    - For "pie", only use categorical columns with 10 or fewer unique values.
    - Vary the chart types — do not repeat the same type more than twice.
    - Prioritise charts that are genuinely informative given the data shape.
    - For "pie" and "bar", NEVER use ID columns, index columns, or any numeric column
    whose name contains 'id', 'ID', 'Id', 'key', 'code', or 'num' as the value.
    Use aggregated counts instead by setting "agg":"count" for bar,
    or use "pie_count" for pie.

    ## Example

    Input:
    Shape: 200 rows × 4 columns
    Columns: 'Department' (object), 'Salary' (float64), 'Age' (int64), 'Emp_ID' (int64)

    Output:
    [
    {"type":"bar",       "cat":"Department", "val":"Salary", "agg":"mean", "title":"Average Salary by Department"},
    {"type":"histogram", "col":"Age",        "bins":20,                    "title":"Age Distribution"},
    {"type":"box",       "cat":"Department", "val":"Age",                  "title":"Age Spread by Department"}
    ]

    Notice: Emp_ID was ignored because it is an identifier, not a meaningful measure.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": summary},
        ],
        temperature=0.3,
    )

    raw = response.choices[0].message.content.strip()

    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    return json.loads(raw)

def _render_chart(df: pd.DataFrame, spec: dict) -> None:
    """Render a single chart from an AI-produced spec dict."""
    try:
        t = spec.get("type")

        if t == "scatter":
            x, y  = spec["x"], spec["y"]
            color = spec.get("color")
            fig   = px.scatter(
                df, x=x, y=y,
                color=None if color in (None, "None") else color,
                trendline="ols",
                template="plotly_dark",
                title=spec.get("title", f"{y} vs {x}"),
            )

        elif t == "bar":
            cat, val, agg = spec["cat"], spec["val"], spec.get("agg", "mean")
            grouped = (
                df.groupby(cat)[val]
                .agg(agg)
                .reset_index()
                .sort_values(val, ascending=False)
            )
            fig = px.bar(
                grouped, x=cat, y=val,
                template="plotly_dark",
                title=spec.get("title", f"{agg.capitalize()} of {val} by {cat}"),
                text_auto=".2f",
            )

        elif t == "histogram":
            col  = spec["col"]
            bins = spec.get("bins", 20)
            fig  = px.histogram(
                df, x=col, nbins=bins,
                template="plotly_dark",
                title=spec.get("title", f"Distribution of {col}"),
            )

        elif t == "line":
            date_col = spec["date"]
            val      = spec["val"]
            color    = spec.get("color")
            plot_df  = df.copy()
            plot_df[date_col] = pd.to_datetime(plot_df[date_col], errors="coerce")
            fig = px.line(
                plot_df.sort_values(date_col),
                x=date_col, y=val,
                color=None if color in (None, "None") else color,
                template="plotly_dark",
                title=spec.get("title", f"{val} over {date_col}"),
            )

        elif t == "box":
            cat = spec.get("cat")
            val = spec["val"]
            fig = px.box(
                df,
                x=cat if cat and cat != "None" else None,
                y=val,
                template="plotly_dark",
                title=spec.get("title", f"Distribution of {val}"),
            )

        elif t == "pie":
            names  = spec["names"]
            values = spec["values"]
            pie_df = (
                df.groupby(names)[values]
                .sum()
                .reset_index()
                .sort_values(values, ascending=False)
                .head(10)
            )
            fig = px.pie(
                pie_df, names=names, values=values,
                template="plotly_dark",
                title=spec.get("title", f"{values} by {names}"),
            )

        elif t == "pie_count":
            names = spec["names"]
            pie_df = (
                df[names]
                .value_counts()
                .reset_index()
                .rename(columns={names: "Category", "count": "Count"})
                .head(10)
            )
            fig = px.pie(
                pie_df, names="Category", values="Count",
                template="plotly_dark",
                title=spec.get("title", f"Distribution of {names}"),
            )
        else:
            st.warning(f"Unknown chart type from AI: `{t}` — skipping.")
            return

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Could not render **{spec.get('title', str(spec))}**: {e}")

def _classify_columns(df: pd.DataFrame):
    """Return (numeric_cols, categorical_cols, date_cols)."""
    numeric_cols     = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    date_cols        = df.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns.tolist()

    # Also detect object columns whose values look like date strings
    for col in df.select_dtypes(include="object").columns:
        if col in date_cols:
            continue
        sample = df[col].dropna().head(10)
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().sum() >= len(sample) * 0.8:
                date_cols.append(col)
        except Exception:
            pass

    return numeric_cols, categorical_cols, date_cols

def _render_manual_explorer(df: pd.DataFrame) -> None:
    numeric_cols, categorical_cols, date_cols = _classify_columns(df)

    if not numeric_cols and not categorical_cols:
        st.warning("No plottable columns found in this dataset.")
        return

    viz_type = st.radio(
        "Select Chart Type",
        ["Correlation", "Comparison", "Distribution", "Trends", "Box Plot", "Pie Chart"],
        horizontal=True,
        key="manual_viz_type",
    )

    # Correlation — Scatter Plot
    if viz_type == "Correlation":
        if len(numeric_cols) >= 2:
            x     = st.selectbox("X-axis (Numeric)", numeric_cols, key="scatter_x")
            y     = st.selectbox(
                "Y-axis (Numeric)", numeric_cols,
                index=min(1, len(numeric_cols) - 1),
                key="scatter_y",
            )
            color = st.selectbox(
                "Color by (optional)", ["None"] + categorical_cols, key="scatter_color"
            )
            if x == y:
                st.warning("Please select different columns for X and Y axes.")
            else:
                try:
                    fig = px.scatter(
                        df, x=x, y=y,
                        color=None if color == "None" else color,
                        trendline="ols",
                        template="plotly_dark",
                        title=f"{y} vs {x}",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Unable to generate chart: {e}")
        else:
            st.warning("Needs at least 2 numeric columns for a scatter plot.")

    # Comparison — Bar Chart
    elif viz_type == "Comparison":
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
                    grouped, x=cat, y=val,
                    template="plotly_dark",
                    title=f"{agg.capitalize()} of {val} by {cat}",
                    text_auto=".2f",
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Unable to generate chart: {e}")
        else:
            st.warning("Needs at least one categorical and one numeric column.")

    # Distribution — Histogram
    elif viz_type == "Distribution":
        all_cols = numeric_cols + categorical_cols
        if all_cols:
            col_sel = st.selectbox("Select Column", all_cols, key="hist_col")
            bins    = (
                st.slider("Number of bins", 5, 100, 20, key="hist_bins")
                if col_sel in numeric_cols
                else None
            )
            try:
                fig = px.histogram(
                    df, x=col_sel, nbins=bins,
                    template="plotly_dark",
                    title=f"Distribution of {col_sel}",
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Unable to generate chart: {e}")
        else:
            st.warning("No columns available for histogram.")

    # Trends — Line Chart
    elif viz_type == "Trends":
        if date_cols and numeric_cols:
            d_col = st.selectbox("Date Column",    date_cols,    key="line_date")
            n_col = st.selectbox("Value to Track", numeric_cols, key="line_val")
            color = st.selectbox(
                "Group by (optional)", ["None"] + categorical_cols, key="line_color"
            )
            try:
                plot_df = df.copy()
                plot_df[d_col] = pd.to_datetime(plot_df[d_col], errors="coerce")
                fig = px.line(
                    plot_df.sort_values(d_col),
                    x=d_col, y=n_col,
                    color=None if color == "None" else color,
                    template="plotly_dark",
                    title=f"{n_col} over {d_col}",
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Unable to generate chart: {e}")
        else:
            st.info(
                "No datetime columns detected. "
                "Try **Fix Data Types** in the cleaning section first, "
                "or use **Standardize Dates** if your date column is stored as text."
            )

    # Box Plot
    elif viz_type == "Box Plot":
        if numeric_cols:
            val = st.selectbox("Value Column", numeric_cols, key="box_val")
            cat = st.selectbox(
                "Group by (optional)", ["None"] + categorical_cols, key="box_cat"
            )
            try:
                fig = px.box(
                    df,
                    x=None if cat == "None" else cat,
                    y=val,
                    template="plotly_dark",
                    title=f"Distribution of {val}" + (f" by {cat}" if cat != "None" else ""),
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Unable to generate chart: {e}")
        else:
            st.warning("Needs at least one numeric column for a box plot.")

    # Pie Chart
    elif viz_type == "Pie Chart":
        if categorical_cols and numeric_cols:
            names      = st.selectbox("Category (Slices)", categorical_cols, key="pie_names")
            values     = st.selectbox("Value (Size)",      numeric_cols,     key="pie_values")
            max_slices = st.slider("Max slices", 3, 20, 10, key="pie_slices")
            try:
                pie_df = (
                    df.groupby(names)[values]
                    .sum()
                    .reset_index()
                    .sort_values(values, ascending=False)
                    .head(max_slices)
                )
                fig = px.pie(
                    pie_df, names=names, values=values,
                    template="plotly_dark",
                    title=f"{values} by {names}",
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Unable to generate chart: {e}")
        else:
            st.warning("Needs at least one categorical and one numeric column for a pie chart.")


def _df_cache_key(df: pd.DataFrame) -> str:
    """A lightweight fingerprint of the dataframe used to bust the AI cache."""
    return f"{df.shape}_{list(df.columns)}_{df.iloc[0].tolist() if len(df) > 0 else []}"


def render_visualization(df: pd.DataFrame) -> None:
    """Draw the chart explorer section for the given DataFrame."""

    # Data quality banner
    quality = st.session_state.get("data_quality", "unknown")
    cleaned = st.session_state.get("cleaned", False)

    if quality == "unclean" and not cleaned:
        st.warning(
            "These visuals are based on raw, unclean data and may be inaccurate. "
            "Consider approving the cleaning plan above for more reliable results."
        )
    elif cleaned:
        st.success("Rendering charts from your cleaned dataset.")
    elif quality == "clean":
        st.success("Dataset looks clean — rendering charts.")

    st.divider()

    # Mode selector
    mode = st.radio(
        "Chart mode",
        ["AI-recommended", "Manual explorer"],
        horizontal=True,
        key="chart_mode",
    )

    st.divider()

    if mode == "AI-recommended":
        cache_key     = "ai_chart_specs"
        cache_key_sig = "ai_chart_specs_sig"  # tracks which df the cache belongs to
        current_sig   = _df_cache_key(df)

        # Bust cache if the dataframe has changed (e.g. after cleaning)
        if st.session_state.get(cache_key_sig) != current_sig:
            st.session_state.pop(cache_key, None)
            st.session_state.pop(cache_key_sig, None)

        # Autogenerate on first load 
        if cache_key not in st.session_state:
            with st.spinner("AI is analyzing your dataset and choosing the best charts…"):
                try:
                    specs = _ask_ai_for_charts(df)
                    st.session_state[cache_key]     = specs
                    st.session_state[cache_key_sig] = current_sig
                except json.JSONDecodeError as e:
                    st.error(f"AI returned malformed JSON: {e}")
                    return
                except Exception as e:
                    st.error(f"AI recommendation failed: {e}")
                    return

        specs = st.session_state[cache_key]

        # Header row with re-generate button
        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown(
                f"**{len(specs)} charts recommended by AI** — based on your dataset's structure and content."
            )
            st.caption("The AI analysed column types, cardinality, and data distributions to select these charts.")
        with col2:
            if st.button("Re-generate", help="Ask the AI for a fresh set of chart recommendations"):
                st.session_state.pop(cache_key, None)
                st.session_state.pop(cache_key_sig, None)
                st.rerun()

        # Render each chart
        for i, spec in enumerate(specs):
            chart_title = spec.get("title", f"Chart {i + 1}")
            chart_type  = spec.get("type", "unknown").capitalize()

            with st.expander(f"{chart_title}  ·  `{chart_type}`", expanded=True):
                _render_chart(df, spec)

    else:
        _render_manual_explorer(df)