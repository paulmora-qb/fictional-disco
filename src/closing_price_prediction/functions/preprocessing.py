"""Functions for preprocessing the data."""

import pandas as pd


def subtract_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Subtract two DataFrames.

    Args:
    ----
        df1 (pd.DataFrame): DataFrame that will be subtracted from.
        df2 (pd.DataFrame): DataFrame that will be subtracted.

    Returns:
    -------
        pd.DataFrame: DataFrame resulting from the subtraction of df2 from df1.

    """
    return df1 - df2


def create_auto_aggregation(
    stock_prices: pd.DataFrame, aggregation_params: dict[str, str]
) -> pd.DataFrame:
    """Create the aggregation columns.

    Args:
        stock_prices (pd.DataFrame): DataFrame with the stock prices.
        aggregation_params (dict[str, str]): Dictionary with the aggregation parameters.

    Returns:
        pd.DataFrame: DataFrame with the date and the aggregation columns
    """
    agg_data = pd.DataFrame(index=stock_prices.index)
    for agg in aggregation_params:
        for window in agg["aggregation_lengths"]:
            agg_type = agg["aggregation_type"]
            agg_data_temp = stock_prices.rolling(window=window).agg(agg_type)
            agg_data_temp = agg_data_temp.rename(
                columns={
                    col: f"{col}_{agg_type}_{window}"
                    for col in agg_data_temp.columns
                    if col != "Date"
                },
            )
            agg_data = pd.concat([agg_data, agg_data_temp], axis=1)

    return agg_data


def _create_feature_shift(
    feature_table: pd.DataFrame, shift_period: int
) -> pd.DataFrame:
    """Shift the feature tables.

    Args:
        feature_table (pd.DataFrame): Tables that need to be shifted.
        shift_period (int): Number of periods to shift the tables.

    Returns:
        pd.DataFrame: Shifted tables.
    """
    feature_table = feature_table.shift(shift_period)
    feature_table.columns = [
        f"{col}_shift_{shift_period}" for col in feature_table.columns
    ]
    return feature_table


def create_master_table(
    closing_prices: pd.DataFrame, *feature_tables: list[pd.DataFrame]
) -> pd.DataFrame:
    """Create the master table.

    The target table is the closing prices. The feature tables are shifted by one day.

    Args:
        closing_prices (pd.DataFrame): DataFrame with the closing prices.

    Returns:
        pd.DataFrame: DataFrame with the closing prices and the shifted feature tables.
    """

    shifted_feature_tables = [
        _create_feature_shift(feature_table, 1) for feature_table in feature_tables
    ]

    return pd.concat([closing_prices] + shifted_feature_tables, axis=1)
