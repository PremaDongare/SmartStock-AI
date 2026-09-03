"""Data ingestion utilities for SmartStock AI."""

from importlib import import_module

__all__ = [
    "REQUIRED_COLUMNS",
    "DEFAULT_INPUT_PATH",
    "DEFAULT_OUTPUT_PATH",
    "detect_duplicate_rows",
    "detect_invalid_numeric_values",
    "detect_invalid_prices",
    "detect_missing_values",
    "detect_negative_units_sold",
    "generate_validation_report",
    "ingest_sales_data",
    "load_raw_dataset",
    "parse_date_column",
    "sort_data_chronologically",
    "validate_required_columns",
]


def __getattr__(name):
    if name in __all__:
        module = import_module(".ingest", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
