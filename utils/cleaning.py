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

def standardize_categorical(df: pd.DataFrame, log: list | None = None) -> pd.DataFrame:
    """
    Strip whitespace, lowercase, and normalize common boolean-like strings
    in all object (string) columns.
    """
    df = df.copy()
    changed_cols = []

    for col in df.select_dtypes(include="object").columns:
        before = df[col].copy()
        df[col] = df[col].astype(str).str.strip().str.lower()
        df[col] = df[col].replace(
            {
                "yes": "yes", "y": "yes", "1": "yes", "true": "yes",
                "no": "no",  "n": "no",  "0": "no",  "false": "no",
                "x": pd.NA,  "unknown": pd.NA,
            }
        )
        if not df[col].equals(before):
            changed_cols.append(col)

    if log is not None and changed_cols:
        log.append(
            f"✏️ Standardized categorical casing/values in: {', '.join(changed_cols)}"
        )

    return df


# ---------------------------------------------------------------------------
# Type inference
# ---------------------------------------------------------------------------

def fix_data_types(df: pd.DataFrame, log: list | None = None) -> pd.DataFrame:
    """
    Attempt to infer better dtypes for each column:
    1. Try casting to numeric.
    2. If that fails, try casting to datetime.
    """
    df = df.copy()
    fixed = []

    for col in df.columns:
        original_dtype = str(df[col].dtype)

        # Numeric pass
        try:
            converted = pd.to_numeric(df[col], errors="ignore")
            if str(converted.dtype) != original_dtype:
                df[col] = converted
                fixed.append(f"{col} → numeric")
                continue
        except Exception:
            pass

        # Datetime pass
        try:
            converted = pd.to_datetime(df[col], errors="ignore")
            if str(converted.dtype) != original_dtype:
                df[col] = converted
                fixed.append(f"{col} → datetime")
        except Exception:
            pass

    if log is not None and fixed:
        log.append(f"🔧 Fixed data types: {'; '.join(fixed)}")

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
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        mask = (df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)
        if mask.any():
            report[col] = int(mask.sum())
    return report


# ---------------------------------------------------------------------------
# Agentic plan executor
# ---------------------------------------------------------------------------

def apply_cleaning_plan(df: pd.DataFrame, plan_steps: list) -> tuple[pd.DataFrame, list]:
    """
    Execute a structured cleaning plan produced by the AI agent.

    Each step is a dict:
        {
            "action":  "fill_median" | "fill_mean" | "fill_mode" |
                       "drop_column" | "remove_duplicates" |
                       "fix_dtypes"  | "standardize_categories",
            "column":  "<column_name> or 'all'",
            "reason":  "<plain English explanation>"
        }

    Returns (cleaned_df, change_log).
    """
    temp = df.copy()
    log = []

    for step in plan_steps:
        action = step.get("action", "").lower()
        col    = step.get("column", "")
        reason = step.get("reason", "")

        try:
            if action == "drop_column":
                if col in temp.columns:
                    temp = temp.drop(columns=[col])
                    log.append(f"🗑️ Dropped column **{col}** — {reason}")

            elif action == "fill_median":
                if col in temp.columns and pd.api.types.is_numeric_dtype(temp[col]):
                    median_val    = temp[col].median()
                    missing_count = temp[col].isnull().sum()
                    temp[col]     = temp[col].fillna(median_val)
                    log.append(
                        f"📊 Filled {missing_count} missing in **{col}** "
                        f"with median ({median_val:.2f}) — {reason}"
                    )

            elif action == "fill_mean":
                if col in temp.columns and pd.api.types.is_numeric_dtype(temp[col]):
                    mean_val      = temp[col].mean()
                    missing_count = temp[col].isnull().sum()
                    temp[col]     = temp[col].fillna(mean_val)
                    log.append(
                        f"📊 Filled {missing_count} missing in **{col}** "
                        f"with mean ({mean_val:.2f}) — {reason}"
                    )

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

            elif action == "remove_duplicates":
                before  = len(temp)
                temp    = temp.drop_duplicates()
                removed = before - len(temp)
                log.append(f"🔁 Removed {removed} duplicate rows — {reason}")

            elif action == "fix_dtypes":
                temp = fix_data_types(temp, log=log)

            elif action == "standardize_categories":
                temp = standardize_categorical(temp, log=log)

        except Exception as e:
            log.append(f"⚠️ Skipped step '{action}' on '{col}': {e}")

    return temp, log