from pathlib import Path

import pandas as pd
import pytest

from ml.ingestion.ingest import (
    DEFAULT_INPUT_PATH,
    REQUIRED_COLUMNS,
    detect_duplicate_rows,
    detect_invalid_numeric_values,
    detect_invalid_prices,
    detect_missing_values,
    detect_negative_units_sold,
    generate_validation_report,
    ingest_sales_data,
    load_raw_dataset,
    parse_date_column,
    sort_data_chronologically,
    validate_required_columns,
)


@pytest.fixture
def valid_raw_dataframe():
    return pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "product_id": "SKU-0001",
                "warehouse_id": "WH-01",
                "units_sold": 42,
                "stock_level": 500,
                "price": 29.99,
                "promotion": False,
                "temperature": 18.5,
                "event": "Holiday",
            },
            {
                "date": "2024-01-01",
                "product_id": "SKU-0001",
                "warehouse_id": "WH-01",
                "units_sold": 38,
                "stock_level": 480,
                "price": 25.5,
                "promotion": True,
                "temperature": 20.2,
                "event": "SpecialEvent",
            },
        ]
    )


@pytest.fixture
def invalid_dataframe():
    return pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "product_id": "SKU-0001",
                "warehouse_id": "WH-01",
                "units_sold": 42,
                "stock_level": 500,
                "price": 29.99,
                "promotion": False,
                "temperature": 18.5,
                "event": "Holiday",
            },
            {
                "date": "2024-01-02",
                "product_id": "SKU-0001",
                "warehouse_id": "WH-01",
                "units_sold": 42,
                "stock_level": 500,
                "price": 29.99,
                "promotion": False,
                "temperature": 18.5,
                "event": "Holiday",
            },
            {
                "date": "2024-01-03",
                "product_id": "SKU-0002",
                "warehouse_id": "WH-02",
                "units_sold": -5,
                "stock_level": 300,
                "price": 0,
                "promotion": True,
                "temperature": "bad",
                "event": None,
            },
            {
                "date": "2024-01-04",
                "product_id": "SKU-0003",
                "warehouse_id": "WH-03",
                "units_sold": "abc",
                "stock_level": None,
                "price": -10,
                "promotion": True,
                "temperature": 21.0,
                "event": "",
            },
        ]
    )


def test_load_raw_dataset(tmp_path):
    file_path = tmp_path / "raw.csv"
    expected = pd.DataFrame({"date": ["2024-01-01"], "product_id": ["SKU-0001"]})
    expected.to_csv(file_path, index=False)

    loaded = load_raw_dataset(file_path)

    assert loaded.equals(expected)


def test_validate_required_columns_raises_for_missing_dataframe():
    df = pd.DataFrame({"date": ["2024-01-01"], "product_id": ["SKU-0001"]})

    with pytest.raises(ValueError, match="missing required columns"):
        validate_required_columns(df, REQUIRED_COLUMNS)


def test_sample_dataset_loads_with_required_columns():
    sample = load_raw_dataset(DEFAULT_INPUT_PATH)

    assert list(sample.columns) == REQUIRED_COLUMNS
    assert len(sample) > 0


def test_missing_required_columns_are_reported():
    df = pd.DataFrame({"date": ["2024-01-01"], "product_id": ["SKU-0001"]})

    with pytest.raises(ValueError, match="warehouse_id.*units_sold.*stock_level"):
        validate_required_columns(df)


def test_missing_values_are_reported_by_column(valid_raw_dataframe):
    dataframe = valid_raw_dataframe.copy()
    dataframe.loc[0, "event"] = " "
    dataframe.loc[1, "stock_level"] = None

    missing = detect_missing_values(dataframe)

    assert missing.set_index("column").loc["event", "missing_count"] == 1
    assert missing.set_index("column").loc["stock_level", "missing_count"] == 1


def test_duplicate_records_are_detected(valid_raw_dataframe):
    dataframe = pd.concat([valid_raw_dataframe, valid_raw_dataframe.iloc[[0]]], ignore_index=True)

    duplicates = detect_duplicate_rows(dataframe)

    assert len(duplicates) == 1
    assert duplicates.iloc[0]["product_id"] == "SKU-0001"


def test_negative_units_sold_are_detected(valid_raw_dataframe):
    dataframe = valid_raw_dataframe.copy()
    dataframe.loc[0, "units_sold"] = -1

    negative_units = detect_negative_units_sold(dataframe)

    assert len(negative_units) == 1
    assert negative_units.iloc[0]["units_sold"] == -1


@pytest.mark.parametrize("price", [0, -1, None])
def test_invalid_prices_are_detected(valid_raw_dataframe, price):
    dataframe = valid_raw_dataframe.copy()
    dataframe.loc[0, "price"] = price

    invalid_prices = detect_invalid_prices(dataframe)

    assert len(invalid_prices) == 1


def test_invalid_dates_are_coerced_and_reported(valid_raw_dataframe):
    dataframe = valid_raw_dataframe.copy()
    dataframe.loc[0, "date"] = "not-a-date"

    parsed = parse_date_column(dataframe, date_format="%Y-%m-%d")
    report = generate_validation_report(parsed)

    assert pd.isna(parsed.loc[0, "date"])
    assert report["invalid_date_rows"] == 1
    assert report["status"] == "fail"


def test_incorrect_numeric_data_types_are_detected(valid_raw_dataframe):
    dataframe = valid_raw_dataframe.copy()
    dataframe["units_sold"] = dataframe["units_sold"].astype(object)
    dataframe["temperature"] = dataframe["temperature"].astype(object)
    dataframe.loc[0, "units_sold"] = "many"
    dataframe.loc[1, "temperature"] = "warm"

    invalid_numeric = detect_invalid_numeric_values(dataframe)

    assert set(invalid_numeric["column"]) == {"units_sold", "temperature"}
    assert len(invalid_numeric) == 2


def test_parse_date_column_and_sort_data_chronologically(valid_raw_dataframe):
    parsed = parse_date_column(valid_raw_dataframe.copy())
    sorted_df = sort_data_chronologically(parsed)

    assert pd.api.types.is_datetime64_any_dtype(parsed["date"])
    assert sorted_df["date"].tolist() == sorted(parsed["date"].tolist())


def test_sort_data_chronologically_orders_dates_stably(valid_raw_dataframe):
    parsed = parse_date_column(valid_raw_dataframe)

    sorted_df = sort_data_chronologically(parsed)

    assert sorted_df["date"].is_monotonic_increasing
    assert sorted_df["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-01",
        "2024-01-02",
    ]


def test_valid_dataset_passes_validation(valid_raw_dataframe):
    parsed = parse_date_column(valid_raw_dataframe)

    report = generate_validation_report(parsed)

    assert report["status"] == "pass"
    assert report["issues_detected"] is False
    assert report["rows_loaded"] == 2


def test_detect_duplicate_rows_and_missing_values(invalid_dataframe):
    duplicates = detect_duplicate_rows(invalid_dataframe)
    missing = detect_missing_values(invalid_dataframe)

    assert len(duplicates) == 1
    assert "event" in missing["column"].values
    assert "temperature" in missing["column"].values or "stock_level" in missing["column"].values


def test_detect_invalid_numeric_values_and_price_and_units_issues(invalid_dataframe):
    invalid_numeric = detect_invalid_numeric_values(invalid_dataframe)
    negative_units = detect_negative_units_sold(invalid_dataframe)
    invalid_price = detect_invalid_prices(invalid_dataframe)

    assert not invalid_numeric.empty
    assert len(negative_units) == 1
    assert len(invalid_price) >= 2


def test_generate_validation_report(invalid_dataframe):
    report = generate_validation_report(invalid_dataframe)

    assert report["rows_loaded"] == len(invalid_dataframe)
    assert report["status"] == "fail"
    assert report["duplicate_rows_found"] == 1
    assert report["negative_units_sold_rows"] >= 1


def test_ingest_sales_data_creates_clean_dataset_and_report(tmp_path):
    raw_path = tmp_path / "raw.csv"
    output_path = tmp_path / "processed" / "clean_sales_data.csv"

    valid_df = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "product_id": "SKU-0001",
                "warehouse_id": "WH-01",
                "units_sold": 10,
                "stock_level": 100,
                "price": 29.99,
                "promotion": False,
                "temperature": 18.5,
                "event": "Holiday",
            },
            {
                "date": "2024-01-02",
                "product_id": "SKU-0001",
                "warehouse_id": "WH-01",
                "units_sold": 12,
                "stock_level": 120,
                "price": 31.5,
                "promotion": True,
                "temperature": 20.0,
                "event": "BlackFriday",
            },
            {
                "date": "2024-01-02",
                "product_id": "SKU-0001",
                "warehouse_id": "WH-01",
                "units_sold": 12,
                "stock_level": 120,
                "price": 31.5,
                "promotion": True,
                "temperature": 20.0,
                "event": "BlackFriday",
            },
            {
                "date": "2024-01-03",
                "product_id": "SKU-0002",
                "warehouse_id": "WH-02",
                "units_sold": 8,
                "stock_level": 200,
                "price": 41.0,
                "promotion": False,
                "temperature": 17.7,
                "event": "",
            },
        ]
    )
    valid_df.to_csv(raw_path, index=False)

    cleaned_df, report = ingest_sales_data(raw_path, output_path)

    assert output_path.exists()
    assert len(cleaned_df) == 2
    assert report["output_path"] == str(output_path)
    assert list(cleaned_df.columns) == REQUIRED_COLUMNS
    assert cleaned_df["date"].isna().sum() == 0
