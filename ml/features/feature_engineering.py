"""Leakage-safe feature engineering for SmartStock demand forecasting."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

GROUP_COLUMNS = ["product_id", "warehouse_id"]
REQUIRED_COLUMNS = [
    "date",
    "product_id",
    "warehouse_id",
    "units_sold",
    "price",
    "promotion",
    "stock_level",
    "temperature",
    "event",
]
LAG_FEATURES = {
    "lag_1": 1,
    "lag_7": 7,
    "lag_14": 14,
    "lag_28": 28,
}
ROLLING_WINDOWS = (7, 14, 28)


def _validate_input(data: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError("Data is missing required columns: " + ", ".join(missing))


def _prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    _validate_input(data)
    prepared = data.copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="raise")
    return prepared.sort_values(GROUP_COLUMNS + ["date"], kind="mergesort").reset_index(drop=True)


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create forecasting features using only observations available before each row.

    The lag and rolling features are calculated independently per product/warehouse
    series. Rolling windows shift by one row, so the current target is excluded.
    """
    features = _prepare_data(data)
    grouped_units = features.groupby(GROUP_COLUMNS, sort=False)["units_sold"]
    grouped_price = features.groupby(GROUP_COLUMNS, sort=False)["price"]

    for name, periods in LAG_FEATURES.items():
        features[name] = grouped_units.shift(periods)

    prior_units = grouped_units.shift(1)
    prior_group = prior_units.groupby(
        [features[column] for column in GROUP_COLUMNS],
        sort=False,
    )
    for window in ROLLING_WINDOWS:
        features[f"rolling_mean_{window}"] = prior_group.transform(
            lambda values: values.rolling(window, min_periods=1).mean()
        )
        if window in (7, 28):
            features[f"rolling_std_{window}"] = prior_group.transform(
                lambda values: values.rolling(window, min_periods=2).std()
            )

    features["day_of_week"] = features["date"].dt.dayofweek
    features["day_of_month"] = features["date"].dt.day
    features["month"] = features["date"].dt.month
    features["week_of_year"] = features["date"].dt.isocalendar().week.astype("int64")
    features["is_weekend"] = features["day_of_week"] >= 5
    features["price_change"] = grouped_price.diff()

    return features


def create_target(data: pd.DataFrame) -> pd.Series:
    """Return the demand target without shifting or otherwise altering it."""
    if "units_sold" not in data.columns:
        raise ValueError("Data must contain the target column: units_sold")
    return data["units_sold"].copy().rename("units_sold")


def handle_missing_values(
    data: pd.DataFrame,
    *,
    drop_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Prepare feature rows with a transparent missing-value policy.

    Initial lag/rolling rows do not have enough history and are dropped by default.
    Remaining numeric gaps use the training-safe column median, while categorical
    gaps are labeled explicitly. ``drop_columns`` can require additional columns
    to be complete instead of imputing them.
    """
    cleaned = data.copy()
    default_drop = [
        column
        for column in [
            "lag_1",
            "lag_7",
            "lag_14",
            "lag_28",
            "rolling_mean_7",
            "rolling_mean_14",
            "rolling_mean_28",
        ]
        if column in cleaned.columns
    ]
    columns_to_drop = list(drop_columns) if drop_columns is not None else default_drop
    cleaned = cleaned.dropna(subset=columns_to_drop).copy()

    numeric_columns = cleaned.select_dtypes(include="number").columns
    for column in numeric_columns:
        if cleaned[column].isna().any():
            cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    categorical_columns = cleaned.select_dtypes(include=["object", "string", "category"]).columns
    for column in categorical_columns:
        if cleaned[column].isna().any():
            cleaned[column] = cleaned[column].fillna("Unknown")
    return cleaned.reset_index(drop=True)


FEATURE_COLUMNS = [
    *LAG_FEATURES,
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_28",
    "day_of_week",
    "day_of_month",
    "month",
    "week_of_year",
    "is_weekend",
    "price_change",
    "promotion",
    "stock_level",
    "temperature",
    "event",
]
