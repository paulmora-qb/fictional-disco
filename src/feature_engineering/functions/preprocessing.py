"""Functions for preprocessing the data."""

import pandas as pd
from functools import reduce
from typing import Any


def basic_arithmetic(
    price_data: pd.DataFrame, arithmetic_params: dict[str, str]
) -> pd.DataFrame:
    """Perform basic arithmetic operations on the price data.

    Args:
        price_data (pd.DataFrame): DataFrame with the price data.
        arithmetic_params (dict[str, str]): Dictionary with the arithmetic parameters.

    Returns:
        pd.DataFrame: DataFrame with the new columns.
    """
    for operation in arithmetic_params:
        new_column = operation["new_column"]
        formula = operation["formula"]

        if "-" in formula:
            columns = formula.split(" - ")
            price_data[new_column] = price_data[columns[0]] - price_data[columns[1]]
    return price_data


def calculate_rolling_aggregations(
    price_data: pd.DataFrame, aggregation_params: dict[str, str]
) -> pd.DataFrame:
    """Calculate rolling aggregations on the price data.

    Args:
        price_data (pd.DataFrame): DataFrame with the price data.
        aggregation_params (dict[str, str]): Dictionary with the aggregation parameters.

    Returns:
        pd.DataFrame: DataFrame with the new columns.
    """

    def apply_rolling(group, aggregation_params):
        for param in aggregation_params:
            columns = param["aggregation_columns"]
            aggregation_type = param["aggregation_type"]
            aggregation_lengths = param["aggregation_lengths"]

            for column in columns:
                for length in aggregation_lengths:
                    column_name = f"{column}_{aggregation_type}_{length}"
                    if aggregation_type == "mean":
                        group[column_name] = group[column].rolling(window=length).mean()
                    elif aggregation_type == "std":
                        group[column_name] = group[column].rolling(window=length).std()
        return group

    # Group by stock ticker and apply the rolling calculations
    price_data = price_data.groupby("stock_ticker").apply(
        apply_rolling, aggregation_params
    )

    return price_data.reset_index(drop=True)


def shift_features(
    price_data: pd.DataFrame, shift_params: dict[str, Any]
) -> pd.DataFrame:
    """_summary_

    Args:
        price_data (pd.DataFrame): _description_
        shift_params (dict[str, Any]): _description_

    Returns:
        pd.DataFrame: _description_
    """
    feature_table_shift = feature_table.drop(columns=[date_column]).shift(shift_period)
    feature_table_shift.columns = [
        f"{col}_shift_{shift_period}" if col != date_column else col
        for col in feature_table_shift.columns
    ]
    feature_table_shift[date_column] = feature_table[date_column]
    return feature_table
