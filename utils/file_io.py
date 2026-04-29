import pandas as pd
from utils.cleaning import sanitize_data


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    Parse a Streamlit UploadedFile (CSV or Excel) into a DataFrame
    and apply initial sanitization (junk-string → pd.NA).
    """
    name = uploaded_file.name
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    return sanitize_data(df)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Encode a DataFrame as UTF-8 CSV bytes, ready for st.download_button."""
    return df.to_csv(index=False).encode("utf-8")


def make_export_filename(original_filename: str) -> str:
    """Return a 'cleaned_<name>.csv' string for the download button label."""
    base = original_filename or "data.csv"
    if not base.endswith(".csv"):
        base += ".csv"
    return f"cleaned_{base}"