import pandas as pd
import pytest

from ml.features.feature_engineering import (
    FEATURE_COLUMNS,
    create_features,
    create_target,
    handle_missing_values,
)


@pytest.fixture
def demand_data():
    dates = pd.date_range("2024-01-01", periods=35, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "product_id": "SKU-0001",
            "warehouse_id": "WH-01",
            "units_sold": range(1, 36),
            "price": [10.0] * 35,
            "promotion": [False] * 35,
            "stock_level": [1000] * 35,
            "temperature": [20.0] * 35,
            "event": ["No event"] * 35,
        }
    )


def test_create_features_contains_requested_columns(demand_data):
    features = create_features(demand_data)

    assert set(FEATURE_COLUMNS).issubset(features.columns)
    assert features["day_of_week"].tolist()[:2] == [0, 1]
    assert features["is_weekend"].dtype == bool


def test_lags_and_rollings_use_only_past_demand(demand_data):
    features = create_features(demand_data)

    assert pd.isna(features.loc[0, "lag_1"])
    assert features.loc[7, "lag_1"] == 7
    assert features.loc[7, "lag_7"] == 1
    assert features.loc[7, "rolling_mean_7"] == pytest.approx(4)
    assert features.loc[8, "rolling_mean_7"] == pytest.approx(5)
    assert features.loc[28, "lag_28"] == 1


def test_features_are_sorted_per_series_before_calculation(demand_data):
    shuffled = demand_data.sample(frac=1, random_state=7)

    features = create_features(shuffled)

    assert features["date"].is_monotonic_increasing
    assert features.loc[7, "lag_7"] == 1


def test_features_are_calculated_independently_per_product_and_warehouse(demand_data):
    second_series = demand_data.assign(
        product_id="SKU-0002",
        units_sold=demand_data["units_sold"] + 100,
    )

    features = create_features(pd.concat([demand_data, second_series]))

    assert features.groupby(["product_id", "warehouse_id"]).size().eq(35).all()
    second_first_lag = features.loc[
        (features["product_id"] == "SKU-0002") & features["lag_1"].notna(),
        "lag_1",
    ].iloc[0]
    assert second_first_lag == 101


def test_create_target_returns_units_sold(demand_data):
    target = create_target(demand_data)

    assert target.name == "units_sold"
    pd.testing.assert_series_equal(target, demand_data["units_sold"], check_names=True)


def test_handle_missing_values_drops_history_rows_and_fills_remaining_gaps(demand_data):
    features = create_features(demand_data)
    features.loc[30, "temperature"] = None
    features.loc[30, "event"] = None

    cleaned = handle_missing_values(features)

    assert len(cleaned) == len(demand_data) - 28
    assert cleaned[["lag_1", "lag_7", "lag_14", "lag_28"]].notna().all().all()
    assert cleaned.loc[cleaned["date"] == "2024-01-31", "temperature"].notna().all()
    assert cleaned.loc[cleaned["date"] == "2024-01-31", "event"].item() == "Unknown"


def test_missing_required_input_columns_raise(demand_data):
    with pytest.raises(ValueError, match="missing required columns"):
        create_features(demand_data.drop(columns=["event"]))
