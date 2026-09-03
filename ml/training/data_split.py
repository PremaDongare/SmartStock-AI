"""Chronological, leakage-safe train/validation/test splitting."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def split_dataset(
    data: pd.DataFrame,
    *,
    date_column: str = "date",
    group_columns: Sequence[str] = ("product_id", "warehouse_id"),
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data chronologically, independently for each product/warehouse series.

    Rows are sorted by date using a stable sort and are never shuffled. Within
    each series, the first 70% goes to training, the next 15% to validation,
    and the remaining rows to testing. Integer boundaries use ``floor`` for
    train and validation; the test set receives the remainder.
    """
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(ratio < 0 or ratio > 1 for ratio in ratios) or not sum(ratios) == 1:
        raise ValueError("train_ratio, validation_ratio, and test_ratio must sum to 1")
    required = [date_column, *group_columns]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError("Data is missing required columns: " + ", ".join(missing))

    prepared = data.copy()
    prepared[date_column] = pd.to_datetime(prepared[date_column], errors="raise")
    prepared["_split_order"] = range(len(prepared))
    prepared = prepared.sort_values(
        [*group_columns, date_column, "_split_order"], kind="mergesort"
    )

    train_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for _, series in prepared.groupby(list(group_columns), sort=False, dropna=False):
        row_count = len(series)
        train_end = int(row_count * train_ratio)
        validation_end = train_end + int(row_count * validation_ratio)
        train_parts.append(series.iloc[:train_end])
        validation_parts.append(series.iloc[train_end:validation_end])
        test_parts.append(series.iloc[validation_end:])

    def finalize(parts: list[pd.DataFrame]) -> pd.DataFrame:
        if not parts:
            result = prepared.iloc[0:0].copy()
        else:
            result = pd.concat(parts, ignore_index=True)
        return result.drop(columns="_split_order").reset_index(drop=True)

    return finalize(train_parts), finalize(validation_parts), finalize(test_parts)


def split_time_series(
    data: pd.DataFrame,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Alias for :func:`split_dataset` with an explicit time-series name."""
    return split_dataset(data, **kwargs)
