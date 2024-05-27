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
    date_column = "Date"
    df1 = df1.set_index(date_column)
    df2 = df2.set_index(date_column)

    return df1 - df2


def create_auto_aggregation(df, column_name, aggregation_period):
    """Create a new column with the auto-aggregated data.

    Args:
    ----
        df (pd.DataFrame): DataFrame with the data.
        column_name (str): Name of the column to be auto-aggregated.
        aggregation_period (int): Number of periods to aggregate.

    Returns:
    -------
        pd.DataFrame: DataFrame with the new column.

    """
    return df.assign(
        **{
            f"{column_name}_auto_aggregated_{aggregation_period}": df[column_name]
            .rolling(aggregation_period)
            .mean()
        }
    )


def create_feature_shift(df, column_name, shift_period):
    """Create a new column with the shifted data.

    Args:
    ----
        df (pd.DataFrame): DataFrame with the data.
        column_name (str): Name of the column to be shifted.
        shift_period (int): Number of periods to shift.

    Returns:
    -------
        pd.DataFrame: DataFrame with the new column.

    """
    return df.assign(
        **{f"{column_name}_shifted_{shift_period}": df[column_name].shift(shift_period)}
    )


def create_master_table(data_dict, relevant_columns):
    """Create a master table with the relevant columns.

    Args:
    ----
        data_dict (dict[str, pd.DataFrame]): Dictionary with the stock information for each symbol.
        relevant_columns (list[str]): List of relevant columns.

    Returns:
    -------
        dict[str, pd.DataFrame]: Keys are the column names and values are the
            concatenated DataFrames. In the concatenated DataFrames, the columns are
            the stock symbols.

    """
    return {
        col: pd.concat(
            [df[[col]].rename(columns={col: name}) for name, df in data_dict.items()],
            axis=1,
        )
        for col in relevant_columns
    }
