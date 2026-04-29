import pandas as pd


def build_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Return a structured dict describing the dataset.

    Includes shape, column names, dtypes, missing value counts/percentages,
    duplicate count, numeric + categorical stats, and a 5-row sample.
    """
    return {
        "shape": {
            "rows": len(df),
            "columns": len(df.columns),
        },
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "numeric_stats": df.describe(include="number").fillna("").to_dict(),
        "categorical_stats": (
            df.describe(include="object").fillna("").to_dict()
            if df.select_dtypes(include="object").shape[1] > 0
            else {}
        ),
        "sample": df.head(5).fillna("").to_dict(orient="records"),
    }