import pandas as pd
import pytest

from ml.training.data_split import split_dataset, split_time_series


@pytest.fixture
def time_series_data():
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    rows = []
    for product_id, offset in [("SKU-0001", 0), ("SKU-0002", 100)]:
        for warehouse_id in ["WH-01", "WH-02"]:
            for index, date in enumerate(dates):
                rows.append(
                    {
                        "date": date,
                        "product_id": product_id,
                        "warehouse_id": warehouse_id,
                        "units_sold": index + offset,
                    }
                )
    return pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)


def test_split_is_chronological_and_has_expected_proportions(time_series_data):
    train, validation, test = split_dataset(time_series_data)

    assert len(train) == 56
    assert len(validation) == 12
    assert len(test) == 12
    assert train["date"].max() < validation["date"].min()
    assert validation["date"].max() < test["date"].min()


def test_each_product_warehouse_series_is_split_independently(time_series_data):
    train, validation, test = split_dataset(time_series_data)

    for _, series_train in train.groupby(["product_id", "warehouse_id"]):
        key = tuple(series_train[["product_id", "warehouse_id"]].iloc[0])
        series_validation = validation[
            (validation["product_id"] == key[0])
            & (validation["warehouse_id"] == key[1])
        ]
        series_test = test[
            (test["product_id"] == key[0]) & (test["warehouse_id"] == key[1])
        ]
        assert series_train["date"].max() < series_validation["date"].min()
        assert series_validation["date"].max() < series_test["date"].min()


def test_input_is_not_mutated_or_randomly_shuffled(time_series_data):
    original = time_series_data.copy(deep=True)

    train, validation, test = split_time_series(time_series_data)

    pd.testing.assert_frame_equal(time_series_data, original)
    combined = pd.concat([train, validation, test], ignore_index=True)
    assert set(map(tuple, combined.to_numpy())) == set(map(tuple, original.to_numpy()))


def test_invalid_ratios_raise(time_series_data):
    with pytest.raises(ValueError, match="sum to 1"):
        split_dataset(time_series_data, train_ratio=0.8, validation_ratio=0.1, test_ratio=0.2)


def test_missing_split_columns_raise(time_series_data):
    with pytest.raises(ValueError, match="missing required columns"):
        split_dataset(time_series_data.drop(columns=["warehouse_id"]))
