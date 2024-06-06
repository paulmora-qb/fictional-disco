"""Functions for preprocessing the data."""

import pandas as pd
from functools import reduce


def subtract_dataframes(
    df1: pd.DataFrame, df2: pd.DataFrame, name: str
) -> pd.DataFrame:
    """Subtract two DataFrames.

    Args:
    ----
        df1 (pd.DataFrame): DataFrame that will be subtracted from.
        df2 (pd.DataFrame): DataFrame that will be subtracted.
        name (str): Name of the operation.

    Returns:
    -------
        pd.DataFrame: DataFrame resulting from the subtraction of df2 from df1.

    """
    date_column = "Date"
    merged_df = pd.merge(df1, df2, on=date_column, suffixes=("_df1", "_df2"))

    # Subtract the columns from df2 from df1 and rename the columns
    result_df = pd.DataFrame()
    result_df[date_column] = merged_df[date_column]
    for column in df1.columns:
        if column != date_column:
            result_df[f"{column}_{name}"] = (
                merged_df[column + "_df1"] - merged_df[column + "_df2"]
            )
    return result_df


def create_auto_aggregation(
    stock_prices: pd.DataFrame, aggregation_params: dict[str, str]
) -> pd.DataFrame:
    """Create the aggregation columns.

    Args:
    ----
        stock_prices (pd.DataFrame): DataFrame with the stock prices.
        aggregation_params (dict[str, str]): Dictionary with the aggregation parameters.

    Returns:
    -------
        pd.DataFrame: DataFrame with the date and the aggregation columns

    """
    agg_data = pd.DataFrame()
    for agg in aggregation_params:
        for window in agg["aggregation_lengths"]:
            agg_type = agg["aggregation_type"]
            agg_data_temp = (
                stock_prices.drop(columns=["Date"]).rolling(window=window).agg(agg_type)
            )
            agg_data_temp = agg_data_temp.rename(
                columns={
                    col: f"{col}_{agg_type}_{window}" for col in agg_data_temp.columns
                },
            )
            agg_data = pd.concat([agg_data, agg_data_temp], axis=1)
    agg_data["Date"] = stock_prices["Date"]
    return agg_data


def _create_feature_shift(
    feature_table: pd.DataFrame, date_column: str, shift_period: int
) -> pd.DataFrame:
    """Shift the feature tables.

    Args:
    ----
        feature_table (pd.DataFrame): Tables that need to be shifted.
        date_column (str): Name of the date column.
        shift_period (int): Number of periods to shift the tables.

    Returns:
    -------
        pd.DataFrame: Shifted tables.

    """
    feature_table_shift = feature_table.drop(columns=[date_column]).shift(shift_period)
    feature_table_shift.columns = [
        f"{col}_shift_{shift_period}" if col != date_column else col
        for col in feature_table_shift.columns
    ]
    feature_table_shift[date_column] = feature_table[date_column]
    return feature_table


def _create_common_columns_dataframe(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create a dictionary of DataFrames with common columns.

    Args:
    ----
        df (pd.DataFrame): DataFrame with columns to be split.

    Returns:
    -------
        dict[str, pd.DataFrame]: Dictionary with the common columns as keys and the
            DataFrames as values.

    """
    df_dict = {}
    # Iterate over columns
    for col in df.columns:
        # Extract common part of column name
        common_part = col.split("_")[0]
        # Check if common part already exists in dictionary
        if common_part in df_dict:
            # Append column to existing DataFrame
            df_dict[common_part][col] = df[col]
        else:
            # Create new DataFrame with the column
            df_dict[common_part] = df[[col]].copy()
    return df_dict


def _remove_missing_cell_rows(
    df_dict: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Remove rows with missing cells from the DataFrames.

    Args:
    ----
        df_dict (dict[str, pd.DataFrame]): Dictionary with the DataFrames.

    Returns:
    -------
        dict[str, pd.DataFrame]: Dictionary with the DataFrames without missing cells.

    """
    return {key: df.dropna() for key, df in df_dict.items()}


def create_master_dict(
    closing_prices: pd.DataFrame,
    master_table_params: dict[str, str],
    *feature_tables: list[pd.DataFrame],
) -> pd.DataFrame:
    """Create the master table.

    This function is doing multiple things:

    1. Shift the feature tables.
    2. Concatenate the closing prices and the shifted feature tables.
    3. Create a dictionary with the common columns.
    4. Remove rows with missing cells.

    Args:
    ----
        closing_prices (pd.DataFrame): DataFrame with the closing prices.
        master_table_params (dict[str, str]): Dictionary with the parameters for the
            master table creation.
        *feature_tables (list[pd.DataFrame]): List of DataFrames with the features.

    Returns:
    -------
        pd.DataFrame: DataFrame with the closing prices and the shifted feature tables.

    """
    # Create the shifted feature tables
    date_column = master_table_params["date_column"]
    shift_period = master_table_params["shift_period"]
    stock_tables = [
        _create_feature_shift(
            feature_table=feature_table,
            date_column=date_column,
            shift_period=shift_period,
        )
        for feature_table in feature_tables
    ]
    stock_tables += [closing_prices]
    # Merge all the data frames
    merged_df = reduce(
        lambda left, right: pd.merge(left, right, on=date_column), stock_tables
    )
    # Create a dictionary with the common columns
    master_dict = _create_common_columns_dataframe(merged_df)
    # Remove rows with missing cells
    return _remove_missing_cell_rows(master_dict)
