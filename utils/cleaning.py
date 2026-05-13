import pandas as pd


# ---------------------------------------------------------------------------
# Sanitization
# ---------------------------------------------------------------------------

def sanitize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Replace common junk/null-like strings with pd.NA."""
    junk_values = ["", " ", "NULL", "null", "?", "N/A", "n/a", "NA"]
    return df.replace(junk_values, pd.NA)


# ---------------------------------------------------------------------------
# Standardization
# ---------------------------------------------------------------------------

def standardize_categorical(
    df: pd.DataFrame,
    col: str | None = None,
    log: list | None = None,
) -> pd.DataFrame:
    """
    Strip whitespace, lowercase, and normalize common boolean-like strings.

    If `col` is given, only that column is touched.
    If `col` is None (or "all"), every object column is processed.
    """
    df = df.copy()
    changed_cols = []

    targets = (
        [col]
        if col and col != "all" and col in df.columns
        else df.select_dtypes(include="object").columns.tolist()
    )

    for c in targets:
        if df[c].dtype != object:
            continue
        before = df[c].copy()
        df[c] = df[c].astype(str).str.strip().str.lower()
        df[c] = df[c].replace(
            {
                "yes":     "yes", "y":     "yes", "1": "yes", "true":  "yes",
                "no":      "no",  "n":     "no",  "0": "no",  "false": "no",
                "x":       pd.NA,
                "unknown": pd.NA,
                "nan":     pd.NA,   # str(pd.NA) artefact
            }
        )
        if not df[c].equals(before):
            changed_cols.append(c)

    if log is not None and changed_cols:
        log.append(
            f"✏️ Standardized categorical values in: {', '.join(changed_cols)}"
        )

    return df


# ---------------------------------------------------------------------------
# Type inference
# ---------------------------------------------------------------------------

def fix_data_types(
    df: pd.DataFrame,
    col: str | None = None,
    dtype: str | None = None,
    log: list | None = None,
) -> pd.DataFrame:
    """
    Attempt to infer / cast better dtypes.

    If `col` + `dtype` are provided (from an AI step), cast that column
    explicitly.  Otherwise fall back to auto-inference across all columns.
    """
    df = df.copy()
    fixed = []

    # --- Explicit cast requested by the AI ---
    if col and col != "all" and col in df.columns and dtype:
        try:
            if "datetime" in dtype:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                fixed.append(f"{col} → datetime")
            elif dtype in ("float", "float64"):
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
                fixed.append(f"{col} → float")
            elif dtype in ("int", "int64"):
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
                fixed.append(f"{col} → int")
            elif dtype == "str":
                df[col] = df[col].astype(str)
                fixed.append(f"{col} → str")
        except Exception as e:
            if log is not None:
                log.append(f"⚠️ Could not cast {col} to {dtype}: {e}")
        if log is not None and fixed:
            log.append(f"🔧 Fixed data types: {'; '.join(fixed)}")
        return df

    # --- Auto-inference across all columns ---
    for c in df.columns:
        original_dtype = str(df[c].dtype)

        try:
            converted = pd.to_numeric(df[c], errors="ignore")
            if str(converted.dtype) != original_dtype:
                df[c] = converted
                fixed.append(f"{c} → numeric")
                continue
        except Exception:
            pass

        try:
            converted = pd.to_datetime(df[c], errors="ignore")
            if str(converted.dtype) != original_dtype:
                df[c] = converted
                fixed.append(f"{c} → datetime")
        except Exception:
            pass

    if log is not None and fixed:
        log.append(f"🔧 Fixed data types: {'; '.join(fixed)}")

    return df


# ---------------------------------------------------------------------------
# Date standardization  (NEW)
# ---------------------------------------------------------------------------

def standardize_dates(
    df: pd.DataFrame,
    col: str,
    log: list | None = None,
) -> pd.DataFrame:
    """
    Parse a column that contains mixed date formats and normalise every value
    to ISO-8601 strings (YYYY-MM-DD).  Unparseable values become NaT / NaN.
    """
    if col not in df.columns:
        return df

    df = df.copy()
    before_nulls = df[col].isnull().sum()

    df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
    after_nulls  = df[col].isnull().sum()
    coerced      = int(after_nulls - before_nulls)

    # Store as plain date strings so exports stay human-readable
    df[col] = df[col].dt.strftime("%Y-%m-%d")

    msg = f"📅 Standardized dates in **{col}** to YYYY-MM-DD"
    if coerced:
        msg += f" ({coerced} unparseable value(s) set to NaN)"
    if log is not None:
        log.append(msg)

    return df


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

def detect_outliers(df: pd.DataFrame) -> dict:
    """
    Use the IQR method to identify outliers in numeric columns.
    Returns {column_name: outlier_count} for columns that have outliers.
    """
    report = {}
    for col in df.select_dtypes(include="number").columns:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        mask = (df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)
        if mask.any():
            report[col] = int(mask.sum())
    return report


# ---------------------------------------------------------------------------
# Agentic plan executor
# ---------------------------------------------------------------------------

# Full set of actions the AI is allowed to emit (mirrors cleaning_agent.py)
_ALLOWED_ACTIONS = {
    "drop_column",
    "fill_median",
    "fill_mean",
    "fill_mode",
    "fill_constant",
    "remove_duplicates",
    "fix_dtypes",
    "standardize_categories",
    "standardize_dates",
}


def apply_cleaning_plan(
    df: pd.DataFrame,
    plan_steps: list,
) -> tuple[pd.DataFrame, list]:
    """
    Execute a structured cleaning plan produced by the AI agent.

    Each step is a dict with AT LEAST:
        "action"  – one of _ALLOWED_ACTIONS
        "column"  – target column name, or "all" for row-level ops
        "reason"  – plain-English explanation (logged)

    Optional keys:
        "value"   – used by fill_constant
        "dtype"   – used by fix_dtypes  (e.g. "datetime64", "float", "int")

    Returns (cleaned_df, change_log).
    """
    temp = df.copy()
    log  = []

    for step in plan_steps:
        action = step.get("action", "").strip().lower()
        col    = step.get("column", "")
        reason = step.get("reason", "")

        # Skip anything that slipped past the prompt guard
        if action not in _ALLOWED_ACTIONS:
            log.append(f"⚠️ Skipped unrecognised action '{action}' on '{col}'")
            continue

        try:
            # ── Drop column ────────────────────────────────────────────────
            if action == "drop_column":
                if col in temp.columns:
                    temp = temp.drop(columns=[col])
                    log.append(f"🗑️ Dropped column **{col}** — {reason}")

            # ── Fill with median ───────────────────────────────────────────
            elif action == "fill_median":
                if col in temp.columns:
                    # Coerce to numeric first in case the column is stored as object
                    temp[col] = pd.to_numeric(temp[col], errors="coerce")
                    if pd.api.types.is_numeric_dtype(temp[col]):
                        median_val    = temp[col].median()
                        missing_count = temp[col].isnull().sum()
                        temp[col]     = temp[col].fillna(median_val)
                        log.append(
                            f"📊 Filled {missing_count} missing in **{col}** "
                            f"with median ({median_val:.2f}) — {reason}"
                        )

            # ── Fill with mean ─────────────────────────────────────────────
            elif action == "fill_mean":
                if col in temp.columns:
                    temp[col] = pd.to_numeric(temp[col], errors="coerce")
                    if pd.api.types.is_numeric_dtype(temp[col]):
                        mean_val      = temp[col].mean()
                        missing_count = temp[col].isnull().sum()
                        temp[col]     = temp[col].fillna(mean_val)
                        log.append(
                            f"📊 Filled {missing_count} missing in **{col}** "
                            f"with mean ({mean_val:.2f}) — {reason}"
                        )

            # ── Fill with mode ─────────────────────────────────────────────
            elif action == "fill_mode":
                if col in temp.columns:
                    mode_vals = temp[col].mode()
                    if not mode_vals.empty:
                        missing_count = temp[col].isnull().sum()
                        temp[col]     = temp[col].fillna(mode_vals[0])
                        log.append(
                            f"📊 Filled {missing_count} missing in **{col}** "
                            f"with mode ('{mode_vals[0]}') — {reason}"
                        )

            # ── Fill with constant  (NEW) ──────────────────────────────────
            elif action == "fill_constant":
                if col in temp.columns:
                    # AI can suggest a value; fall back to "Unknown" for strings
                    value         = step.get("value", "Unknown")
                    missing_count = temp[col].isnull().sum()
                    temp[col]     = temp[col].fillna(value)
                    log.append(
                        f"📝 Filled {missing_count} missing in **{col}** "
                        f"with '{value}' — {reason}"
                    )

            # ── Remove duplicates ──────────────────────────────────────────
            elif action == "remove_duplicates":
                before  = len(temp)
                temp    = temp.drop_duplicates()
                removed = before - len(temp)
                log.append(f"🔁 Removed {removed} duplicate rows — {reason}")

            # ── Fix data types ─────────────────────────────────────────────
            elif action == "fix_dtypes":
                # Pass col + dtype hint when available so we cast precisely
                temp = fix_data_types(
                    temp,
                    col=col if col != "all" else None,
                    dtype=step.get("dtype"),
                    log=log,
                )

            # ── Standardize categories ─────────────────────────────────────
            elif action == "standardize_categories":
                # Now targets ONLY the specified column, not every object col
                temp = standardize_categorical(
                    temp,
                    col=col if col != "all" else None,
                    log=log,
                )

            # ── Standardize dates  (NEW) ───────────────────────────────────
            elif action == "standardize_dates":
                if col and col != "all":
                    temp = standardize_dates(temp, col=col, log=log)

        except Exception as e:
            log.append(f"⚠️ Skipped step '{action}' on '{col}': {e}")

    return temp, log