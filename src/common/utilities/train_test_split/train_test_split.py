"""Train-test split function."""

import pandas as pd
from typing import Any


def train_test_split(
    stock_price_table: pd.DataFrame, modeling_params: dict[str, Any]
) -> pd.DataFrame:
    """Split the stock price table into training and testing sets.

    This function splits the stock price table into training and testing sets based on
        the time window specified in the modeling parameters.

    Args:
    ----
        stock_price_table (pd.DataFrame): The stock price table.
        modeling_params (dict[str, Any]): The parameters for the train-test split.

    Returns:
    -------
        pd.DataFrame: The stock price table with a 'train_test' column indicating the
            split.

    """
    train_test_params = modeling_params["train_test_split"]
    time_window = train_test_params["time_window"]
    train_test_column = train_test_params["train_test_column"]
    stock_price_table.index = pd.to_datetime(stock_price_table.index)

    # Parse the time window to get the timedelta
    if time_window.endswith("y"):
        n_years = int(time_window[:-1])
        timedelta = pd.DateOffset(years=n_years)
    elif time_window.endswith("m"):
        n_months = int(time_window[:-1])
        timedelta = pd.DateOffset(months=n_months)
    elif time_window.endswith("d"):
        n_days = int(time_window[:-1])
        timedelta = pd.DateOffset(days=n_days)
    else:
        raise ValueError(
            "Unsupported time window format. Use 'y' for years, 'm' for months, or 'd'"
            "for days."
        )

    # Find the latest date in the index
    latest_date = stock_price_table.index.max()

    # Calculate the cutoff date for the test set
    cutoff_date = latest_date - timedelta

    # Create the 'train_test' column
    stock_price_table.loc[:, train_test_column] = [
        "TEST" if date > cutoff_date else "TRAIN" for date in stock_price_table.index
    ]

    return stock_price_table


def filter_train_test_data(
    stock_price_table: pd.DataFrame,
    train_test_split_params: dict[str, str],
    filter_value: str,
    drop_column: bool = False,
) -> pd.DataFrame:
    """Filter the training data based on the train-test split parameters.

    Args:
    ----
        stock_price_table (pd.DataFrame): The stock price table.
        train_test_split_params (dict[str, str]): The parameters for the train-test
            split. Contains the name of the column indicating the split.
        filter_value (str): The value used to filter the training data. Only rows
            where the train_test_column matches this value will be included in the
            filtered stock price table.
        drop_column (bool): Whether to drop the train_test_column from the filtered
            stock price table. Default is False.

    Returns:
    -------
        pd.DataFrame: The filtered stock price table.

    """
    train_test_column = train_test_split_params["train_test_column"]
    filtered_stock_price_data = stock_price_table.query(
        f"{train_test_column} == '{filter_value}'"
    )
    return (
        filtered_stock_price_data.drop(columns=[train_test_column])
        if drop_column
        else filtered_stock_price_data
    )
