"""Reusable ingestion and validation pipeline for SmartStock sales data."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Union

import pandas as pd

REQUIRED_COLUMNS = [
    "date",
    "product_id",
    "warehouse_id",
    "units_sold",
    "stock_level",
    "price",
    "promotion",
    "temperature",
    "event",
]

NUMERIC_COLUMNS = ["units_sold", "stock_level", "temperature", "price"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "sample" / "sample_sales_data.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "clean_sales_data.csv"


def load_raw_dataset(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load a raw CSV dataset into a pandas DataFrame."""
    input_path = Path(file_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")
    return pd.read_csv(input_path)


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Sequence[str] = REQUIRED_COLUMNS,
) -> None:
    """Validate that all required columns exist in the dataset."""
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing_columns)
        )


def parse_date_column(
    df: pd.DataFrame,
    date_column: str = "date",
    date_format: str | None = None,
) -> pd.DataFrame:
    """Parse a date column to pandas datetime format."""
    parsed_df = df.copy()
    parsed_df[date_column] = pd.to_datetime(
        parsed_df[date_column], format=date_format, errors="coerce"
    )
    return parsed_df


def sort_data_chronologically(
    df: pd.DataFrame,
    date_column: str = "date",
) -> pd.DataFrame:
    """Sort the DataFrame chronologically by the date column."""
    sorted_df = df.copy()
    return sorted_df.sort_values(by=date_column, kind="mergesort").reset_index(drop=True)


def detect_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows that are duplicates of an earlier row in the dataset."""
    return df.loc[df.duplicated(keep="first")].copy().reset_index(drop=True)


def _normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Treat blank strings as missing values for validation checks."""
    normalized = df.copy()
    for column in normalized.columns:
        if pd.api.types.is_object_dtype(normalized[column]) or pd.api.types.is_string_dtype(normalized[column]):
            normalized[column] = normalized[column].replace(r"^\s*$", pd.NA, regex=True)
    return normalized


def detect_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize missing values by column."""
    normalized_df = _normalize_missing_values(df)
    missing_summary = normalized_df.isna().sum().reset_index()
    missing_summary.columns = ["column", "missing_count"]
    return missing_summary[missing_summary["missing_count"] > 0].reset_index(drop=True)


def detect_invalid_numeric_values(
    df: pd.DataFrame,
    numeric_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Identify non-numeric values in the provided numeric columns."""
    columns = list(numeric_columns) if numeric_columns is not None else list(NUMERIC_COLUMNS)
    rows = []
    for column in columns:
        if column not in df.columns:
            continue
        converted = pd.to_numeric(df[column], errors="coerce")
        invalid_mask = df[column].notna() & converted.isna()
        if invalid_mask.any():
            invalid_rows = df.loc[invalid_mask, ["date", "product_id", "warehouse_id"]].copy()
            invalid_rows["column"] = column
            invalid_rows["value"] = df.loc[invalid_mask, column].tolist()
            invalid_rows["issue_type"] = f"invalid_{column}"
            rows.append(invalid_rows)
    if not rows:
        return pd.DataFrame(columns=["date", "product_id", "warehouse_id", "column", "value", "issue_type"])
    return pd.concat(rows, ignore_index=True)


def detect_negative_units_sold(
    df: pd.DataFrame,
    units_column: str = "units_sold",
) -> pd.DataFrame:
    """Return rows where units_sold is negative."""
    if units_column not in df.columns:
        return pd.DataFrame(columns=["date", "product_id", "warehouse_id", units_column])
    numeric_values = pd.to_numeric(df[units_column], errors="coerce")
    negative_mask = numeric_values < 0
    return df.loc[negative_mask, ["date", "product_id", "warehouse_id", units_column]].copy().reset_index(drop=True)


def detect_invalid_prices(
    df: pd.DataFrame,
    price_column: str = "price",
) -> pd.DataFrame:
    """Return rows where the price is missing, zero, or negative."""
    if price_column not in df.columns:
        return pd.DataFrame(columns=["date", "product_id", "warehouse_id", price_column])
    numeric_values = pd.to_numeric(df[price_column], errors="coerce")
    invalid_mask = df[price_column].isna() | (numeric_values <= 0)
    return df.loc[invalid_mask, ["date", "product_id", "warehouse_id", price_column]].copy().reset_index(drop=True)


def generate_validation_report(
    df: pd.DataFrame,
    cleaned_df: pd.DataFrame | None = None,
    required_columns: Sequence[str] = REQUIRED_COLUMNS,
) -> dict:
    """Build a summary report of detected data quality issues."""
    duplicate_rows_found = int(df.duplicated(subset=list(df.columns)).sum())
    missing_values = detect_missing_values(df)
    invalid_numeric_rows = detect_invalid_numeric_values(df)
    negative_units_rows = detect_negative_units_sold(df)
    invalid_price_rows = detect_invalid_prices(df)
    invalid_date_rows = int(df["date"].isna().sum()) if "date" in df.columns else 0

    report = {
        "required_columns": list(required_columns),
        "rows_loaded": int(len(df)),
        "rows_after_cleaning": int(len(cleaned_df)) if cleaned_df is not None else None,
        "duplicate_rows_found": duplicate_rows_found,
        "missing_values_by_column": {
            row["column"]: int(row["missing_count"]) for _, row in missing_values.iterrows()
        },
        "invalid_numeric_values_by_column": {
            column: int(
                (df[column].notna() & pd.to_numeric(df[column], errors="coerce").isna()).sum()
            )
            for column in ["units_sold", "stock_level", "temperature", "price"]
            if column in df.columns
        },
        "negative_units_sold_rows": int(len(negative_units_rows)),
        "invalid_price_rows": int(len(invalid_price_rows)),
        "invalid_date_rows": invalid_date_rows,
        "issues_detected": bool(
            duplicate_rows_found
            or not missing_values.empty
            or not invalid_numeric_rows.empty
            or len(negative_units_rows) > 0
            or len(invalid_price_rows) > 0
            or invalid_date_rows > 0
        ),
    }
    report["status"] = "pass" if not report["issues_detected"] else "fail"
    return report


def ingest_sales_data(
    input_path: Union[str, Path] = DEFAULT_INPUT_PATH,
    output_path: Union[str, Path] = DEFAULT_OUTPUT_PATH,
    required_columns: Sequence[str] = REQUIRED_COLUMNS,
) -> tuple[pd.DataFrame, dict]:
    """Execute the ingestion pipeline, validate the dataset, and save the cleaned CSV."""
    raw_df = load_raw_dataset(input_path)
    validate_required_columns(raw_df, required_columns)

    parsed_df = parse_date_column(raw_df.copy(), date_column="date")
    sorted_df = sort_data_chronologically(parsed_df, date_column="date")

    duplicate_rows = detect_duplicate_rows(sorted_df)
    deduplicated_df = sorted_df.drop_duplicates(ignore_index=True)

    missing_values = detect_missing_values(deduplicated_df)
    invalid_numeric_values = detect_invalid_numeric_values(deduplicated_df)
    negative_units_rows = detect_negative_units_sold(deduplicated_df)
    invalid_price_rows = detect_invalid_prices(deduplicated_df)

    if "date" in deduplicated_df.columns:
        invalid_dates_mask = deduplicated_df["date"].isna()
    else:
        invalid_dates_mask = pd.Series(False, index=deduplicated_df.index)

    normalized_df = _normalize_missing_values(deduplicated_df)

    numeric_units = pd.to_numeric(normalized_df["units_sold"], errors="coerce") if "units_sold" in normalized_df.columns else pd.Series(pd.NA, index=normalized_df.index)
    numeric_stock = pd.to_numeric(normalized_df["stock_level"], errors="coerce") if "stock_level" in normalized_df.columns else pd.Series(pd.NA, index=normalized_df.index)
    numeric_temperature = pd.to_numeric(normalized_df["temperature"], errors="coerce") if "temperature" in normalized_df.columns else pd.Series(pd.NA, index=normalized_df.index)
    numeric_price = pd.to_numeric(normalized_df["price"], errors="coerce") if "price" in normalized_df.columns else pd.Series(pd.NA, index=normalized_df.index)

    valid_mask = (
        ~(normalized_df[required_columns].isna().any(axis=1))
        & ~invalid_dates_mask
        & ~((numeric_units.isna()) | (numeric_units < 0))
        & ~((numeric_stock.isna()))
        & ~((numeric_temperature.isna()))
        & ~((numeric_price.isna()) | (numeric_price <= 0))
    )

    cleaned_df = deduplicated_df.loc[valid_mask].copy().reset_index(drop=True)
    cleaned_df = cleaned_df.sort_values(by="date", kind="mergesort").reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)

    report = generate_validation_report(
        sorted_df,
        cleaned_df=cleaned_df,
        required_columns=required_columns,
    )
    report["duplicate_rows_removed"] = int(len(duplicate_rows))
    report["missing_rows_removed"] = int(len(deduplicated_df) - len(cleaned_df))
    report["invalid_numeric_rows_removed"] = int(len(invalid_numeric_values))
    report["negative_units_sold_rows_removed"] = int(len(negative_units_rows))
    report["invalid_price_rows_removed"] = int(len(invalid_price_rows))
    report["output_path"] = str(output_path)
    return cleaned_df, report


if __name__ == "__main__":
    cleaned_df, report = ingest_sales_data()
    print(f"Saved {len(cleaned_df)} cleaned rows to {DEFAULT_OUTPUT_PATH}")
    print(report)
