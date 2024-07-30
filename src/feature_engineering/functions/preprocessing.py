"""Functions for preprocessing the data."""

import re
from typing import Any

import numpy as np
import pandas as pd


def basic_arithmetic(
    price_data: pd.DataFrame, arithmetic_params: dict[str, str]
) -> pd.DataFrame:
    """Perform basic arithmetic operations on the price data.

    Args:
    ----
        price_data (pd.DataFrame): DataFrame with the price data.
        arithmetic_params (dict[str, str]): Dictionary with the arithmetic parameters.

    Returns:
    -------
        pd.DataFrame: DataFrame with the new columns.

    """
    for operation in arithmetic_params:
        new_column = f"ftr_{operation['new_column']}"
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
    ----
        price_data (pd.DataFrame): DataFrame with the price data.
        aggregation_params (dict[str, str]): Dictionary with the aggregation parameters.

    Returns:
    -------
        pd.DataFrame: DataFrame with the new columns.

    """

    def apply_rolling(group, aggregation_params):
        for param in aggregation_params:
            columns = param["aggregation_columns"]
            aggregation_type = param["aggregation_type"]
            aggregation_lengths = param["aggregation_lengths"]

            for column in columns:
                for length in aggregation_lengths:
                    column_name = f"ftr_{column}_{aggregation_type}_{length}"
                    if aggregation_type == "mean":
                        group[column_name] = group[column].rolling(window=length).mean()
                    elif aggregation_type == "std":
                        group[column_name] = group[column].rolling(window=length).std()
        return group

    price_data = price_data.groupby("stock_ticker").apply(
        apply_rolling, aggregation_params
    )

    return price_data.reset_index(drop=True)


def shift_features(
    price_data: pd.DataFrame, shift_params: dict[str, Any]
) -> pd.DataFrame:
    """Shift the features in the price data.

    Args:
    ----
        price_data (pd.DataFrame): DataFrame containing the features that should be
            shifted
        shift_params (dict[str, Any]): Dictionary containing the parameters for the
            shift.

    Returns:
    -------
        pd.DataFrame: DataFrame with the shifted features.

    """

    def apply_shifts(
        group: pd.DataFrame, shift_params: dict[str, str], columns: list[str]
    ):
        shift_period = shift_params["shift_period"]
        for column in columns:
            group[f"{column}_shifted_{shift_period}"] = group[column].shift(
                shift_period
            )
        return group

    feature_columns = _filter_strings(price_data.columns, "ftr.*")
    return (
        price_data.groupby("stock_ticker")
        .apply(apply_shifts, shift_params, feature_columns)
        .reset_index(drop=True)
    )


def log_returns(
    price_data: pd.DataFrame, log_return_params: dict[str, str]
) -> pd.DataFrame:
    """Calculate the log returns for the price data.

    Args:
    ----
        price_data (pd.DataFrame): Price DataFrame containing the stock prices.
        log_return_params (dict[str, str]): Parameters for the log returns.

    Returns:
    -------
        pd.DataFrame: DataFrame with the log returns.

    """

    def log_return(series):
        return np.log(series / series.shift(1))

    columns = log_return_params["columns"]
    for column in columns:
        log_return_column = f"log_return_{column}"
        price_data[log_return_column] = price_data.groupby("stock_ticker")[
            column
        ].transform(log_return)

    return price_data


def _filter_strings(lst_with_strings: list, pattern: str) -> list[str]:
    """Filter strings from a list based on a pattern using regex.

    Args:
    ----
        lst_with_strings (list): The list of strings to filter.
        pattern (str): The pattern to match against the strings.

            - The pattern can include wildcards represented by '*' at
                the start or end of the pattern.
            - If the pattern starts with '*', it matches strings ending
                with the remaining pattern.
            - If the pattern ends with '*', it matches strings starting
                with the remaining pattern.

    Returns:
    -------
        list: A new list containing the filtered strings.

    Example:
    -------
        strings = ["discount_max_10m", "discount_max_3m", "country", "gender"]
        result = _filter_strings(strings, pattern='*e')

        print(result)
        # Output: ["discount_max_10m", "discount_max_3m"]

    Note:
    ----
        - The function uses RegEx expressions to match the pattern against the strings.
        - Matching is case-sensitive.
        - The function Returns new list with filtered strings.

    """
    if pattern.startswith("*."):
        pattern = pattern[1:] + "$"
    elif pattern.endswith(".*"):
        pattern = "^" + pattern[:-1]
    else:
        pattern = "^" + pattern + "$"

    compiled_pattern = re.compile(pattern)
    return [string for string in lst_with_strings if compiled_pattern.search(string)]
