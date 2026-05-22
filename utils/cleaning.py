import pandas as pd

# Sanitization
def sanitize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Replace common junk/null-like strings with pd.NA."""
    junk_values = ["", " ", "NULL", "null", "?", "N/A", "n/a", "NA"]
    return df.replace(junk_values, pd.NA)

# Standardization
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
                "nan":     pd.NA,
            }
        )
        if not df[c].equals(before):
            changed_cols.append(c)

    if log is not None and changed_cols:
        log.append(f"✏️ Standardized categorical values in: {', '.join(changed_cols)}")

    return df

# Type inference
def fix_data_types(
    df: pd.DataFrame,
    col: str | None = None,
    dtype: str | None = None,
    log: list | None = None,
) -> pd.DataFrame:
    """
    Attempt to infer / cast better dtypes.
    If `col` + `dtype` are provided, cast that column explicitly.
    Otherwise fall back to auto-inference across all columns.
    """
    df = df.copy()
    fixed = []

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
                log.append(f"Could not cast {col} to {dtype}: {e}")
        if log is not None and fixed:
            log.append(f"🔧 Fixed data types: {'; '.join(fixed)}")
        return df

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


# Date standardization
def standardize_dates(
    df: pd.DataFrame,
    col: str,
    log: list | None = None,
) -> pd.DataFrame:
    """Parse mixed date formats and normalise to YYYY-MM-DD strings."""
    if col not in df.columns:
        return df

    df = df.copy()
    before_nulls = df[col].isnull().sum()
    df[col]      = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
    after_nulls  = df[col].isnull().sum()
    coerced      = int(after_nulls - before_nulls)
    df[col]      = df[col].dt.strftime("%Y-%m-%d")

    msg = f"Standardized dates in **{col}** to YYYY-MM-DD"
    if coerced:
        msg += f" ({coerced} unparseable value(s) set to NaN)"
    if log is not None:
        log.append(msg)

    return df


# Outlier detection
def detect_outliers(df: pd.DataFrame) -> dict:
    """IQR-based outlier detection. Returns {col: count}."""
    report = {}
    for col in df.select_dtypes(include="number").columns:
        Q1   = df[col].quantile(0.25)
        Q3   = df[col].quantile(0.75)
        IQR  = Q3 - Q1
        mask = (df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)
        if mask.any():
            report[col] = int(mask.sum())
    return report


# Code-gen fallback sandbox executor
def _run_codegen_step(
    df: pd.DataFrame,
    code: str,
    step: dict,
    log: list,
) -> pd.DataFrame:
    """
    Safely execute AI-generated pandas code inside a restricted sandbox.

    The sandbox exposes only `df` and `pd`. The code MUST assign its result
    back to `df`. Any exception is caught, logged, and the original df is
    returned unchanged so one bad step never breaks the whole plan.
    """
    action = step.get("action", "custom")
    col    = step.get("column", "")
    reason = step.get("reason", "")

    # Restricted globals  
    sandbox = {
        "__builtins__": {
            # whitelist only safe builtins
            "len": len, "range": range, "int": int, "float": float,
            "str": str, "list": list, "dict": dict, "print": print,
            "isinstance": isinstance, "enumerate": enumerate, "zip": zip,
        },
        "pd": pd,
        "df": df.copy(), # operate on a copy so failures are non-destructive
    }

    try:
        exec(code, sandbox) # run the generated code
        result = sandbox.get("df") # retrieve the (possibly modified) df

        if not isinstance(result, pd.DataFrame):
            raise ValueError("AI code did not produce a DataFrame named `df`.")

        log.append(
            f"Custom code applied for `{action}`"
            + (f" on **{col}**" if col and col != "all" else "")
            + f" — {reason}"
        )
        return result

    except Exception as e:
        log.append(
            f"Code-gen fallback failed for `{action}`"
            + (f" on '{col}'" if col else "")
            + f": {e}"
        )
        return df   # return original df untouched


# Allowed built-in actions
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

_EXECUTION_ORDER = [
    "fix_dtypes",
    "standardize_dates",
    "standardize_categories",
    "remove_duplicates",
    "fill_median",
    "fill_mean",
    "fill_mode",
    "fill_constant",
    "drop_column",
    # anything not in this list (custom/codegen) runs at the end
]


def _sort_steps(steps: list) -> list:
    """
    Sort plan steps into the safe execution order defined above.
    Steps with unrecognised actions (codegen) are appended at the end.
    """
    def sort_key(step):
        action = step.get("action", "").strip().lower()
        try:
            return _EXECUTION_ORDER.index(action)
        except ValueError:
            return len(_EXECUTION_ORDER)   # custom steps go last

    return sorted(steps, key=sort_key)


def _execute_step(temp: pd.DataFrame, step: dict, log: list) -> pd.DataFrame:
    """Execute a single built-in cleaning step. Returns updated df."""
    action = step.get("action", "").strip().lower()
    col    = step.get("column", "")
    reason = step.get("reason", "")

    if action == "drop_column":
        if col in temp.columns:
            temp = temp.drop(columns=[col])
            log.append(f"Dropped column **{col}** — {reason}")

    elif action == "fill_median":
        if col in temp.columns:
            temp[col] = pd.to_numeric(temp[col], errors="coerce")
            if pd.api.types.is_numeric_dtype(temp[col]):
                median_val    = temp[col].median()
                missing_count = temp[col].isnull().sum()
                temp[col]     = temp[col].fillna(median_val)
                log.append(
                    f"Filled {missing_count} missing in **{col}** "
                    f"with median ({median_val:.2f}) — {reason}"
                )

    elif action == "fill_mean":
        if col in temp.columns:
            temp[col] = pd.to_numeric(temp[col], errors="coerce")
            if pd.api.types.is_numeric_dtype(temp[col]):
                mean_val      = temp[col].mean()
                missing_count = temp[col].isnull().sum()
                temp[col]     = temp[col].fillna(mean_val)
                log.append(
                    f"Filled {missing_count} missing in **{col}** "
                    f"with mean ({mean_val:.2f}) — {reason}"
                )

    elif action == "fill_mode":
        if col in temp.columns:
            mode_vals = temp[col].mode()
            if not mode_vals.empty:
                missing_count = temp[col].isnull().sum()
                temp[col]     = temp[col].fillna(mode_vals[0])
                log.append(
                    f"Filled {missing_count} missing in **{col}** "
                    f"with mode ('{mode_vals[0]}') — {reason}"
                )

    elif action == "fill_constant":
        if col in temp.columns:
            value         = step.get("value", "Unknown")
            missing_count = temp[col].isnull().sum()
            temp[col]     = temp[col].fillna(value)
            log.append(
                f"Filled {missing_count} missing in **{col}** "
                f"with '{value}' — {reason}"
            )

    elif action == "remove_duplicates":
        before  = len(temp)
        temp    = temp.drop_duplicates()
        removed = before - len(temp)
        log.append(f"Removed {removed} duplicate rows — {reason}")

    elif action == "fix_dtypes":
        temp = fix_data_types(
            temp,
            col=col if col != "all" else None,
            dtype=step.get("dtype"),
            log=log,
        )

    elif action == "standardize_categories":
        temp = standardize_categorical(
            temp,
            col=col if col != "all" else None,
            log=log,
        )

    elif action == "standardize_dates":
        if col and col != "all":
            temp = standardize_dates(temp, col=col, log=log)

    return temp


def apply_cleaning_plan(
    df: pd.DataFrame,
    plan_steps: list,
    codegen_fn=None,
) -> tuple[pd.DataFrame, list]:
    """
    Execute a structured cleaning plan produced by the AI agent.

    Steps are automatically reordered into a safe execution sequence
    regardless of the order the AI returned them in, preventing issues
    like filling nulls before fixing dtypes, or standardizing categories
    after filling with unstandardized values.

    Each step dict must have at least:
        "action"  – one of _ALLOWED_ACTIONS, or anything else → codegen fallback
        "column"  – target column name, or "all" for row-level ops
        "reason"  – plain-English explanation (logged)

    Optional keys:
        "value"   – used by fill_constant
        "dtype"   – used by fix_dtypes

    Returns (cleaned_df, change_log).
    """
    temp = df.copy()
    log  = []

    # Reorder steps into safe execution sequence
    ordered_steps = _sort_steps(plan_steps)

    # Log the reordered sequence so the UI can show it
    log.append(
        "Execution order: "
        + " → ".join(s.get("action", "?") for s in ordered_steps)
    )

    for step in ordered_steps:
        action = step.get("action", "").strip().lower()
        col    = step.get("column", "")

        try:
            if action not in _ALLOWED_ACTIONS:
                # Unknown action → codegen fallback
                if codegen_fn is not None:
                    code = codegen_fn(step, temp)
                    if code:
                        temp = _run_codegen_step(temp, code, step, log)
                    else:
                        log.append(f"Code-gen returned nothing for `{action}` on '{col}' — skipped")
                else:
                    log.append(f"Skipped unrecognised action `{action}` on '{col}' (no codegen available)")
            else:
                temp = _execute_step(temp, step, log)

        except Exception as e:
            log.append(f"Skipped step `{action}` on '{col}': {e}")

    return temp, log