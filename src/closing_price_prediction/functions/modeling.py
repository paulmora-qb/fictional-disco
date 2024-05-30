"""Modeling for closing price prediction."""

from typing import Any, TypeVar
import pandas as pd
from pycaret.regression import RegressionExperiment
import matplotlib.pyplot as plt

T = TypeVar("T")


def train_model(
    stock_price_table_split: pd.DataFrame,
    modeling_params: dict[str, str],
):
    """Trains a model for closing price prediction.

    Args:
        stock_price_table_split (pd.DataFrame): Dataframe containing the stock price
            table, which also contains a column which indicates what should be train
            and what is test.
        modeling_params (dict[str, str]): Parameters for the modeling.
            Contains the subsections:

            `train_test_split` (dict[str, str]): Parameters for the train-test split.

            `setup_params` (dict[str, str]): Parameters for the experiment setup.

            `train_params` (dict[str, str]): Parameters for training the model.

    Returns:
        RegressionExperiment, T: The experiment object and the trained model.
    """
    target_variable_name = _extract_target_variable_name(
        stock_price_table_split.columns
    )

    train_stock_price_table = _filter_train_test_data(
        stock_price_table=stock_price_table_split,
        train_test_split_params=modeling_params["train_test_split"],
        filter_value="TRAIN",
    )

    experiment = _experiment_setup(
        data=train_stock_price_table,
        target_variable_name=target_variable_name,
        setup_params=modeling_params["setup_params"],
    )

    base_model = experiment.compare_models(**modeling_params["train_params"])
    tuned_model = experiment.tune_model(base_model, **modeling_params["tuned_params"])
    return experiment, experiment.finalize_model(tuned_model)


def inference(
    stock_price_table_split: pd.DataFrame,
    experiment: RegressionExperiment,
    model: T,
) -> pd.DataFrame:
    """Makes predictions using the trained model.

    Args:
        stock_price_table_split (pd.DataFrame): Dataframe containing the stock price
            table, which also contains a column which indicates what should be train
            and what is test.
        experiment (RegressionExperiment): The experiment object.
        model (T): The trained model.

    Returns:
        pd.DataFrame: The predictions made by the model.
    """
    return experiment.predict_model(model, data=stock_price_table_split)


def train_test_split(
    stock_price_table: pd.DataFrame, modeling_params: dict[str, Any]
) -> pd.DataFrame:
    """Splits the stock price table into training and testing sets.

    This function splits the stock price table into training and testing sets based on
        the time window specified in the modeling parameters.

    Args:
        stock_price_table (pd.DataFrame): The stock price table.
        modeling_params (dict[str, Any]): The parameters for the train-test split.

    Returns:
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


def _experiment_setup(
    data: pd.DataFrame, target_variable_name: str, setup_params: dict[str, str]
):
    """Sets up a regression experiment.

    This function initializes a RegressionExperiment object and configures it with the
        provided data and target variable.

    Parameters:
        data (pd.DataFrame): The input data for the experiment.
        target_variable_name (str): The name of the target variable.
        setup_params (dict[str, str]): Additional parameters for training.

    Returns:
        RegressionExperiment: The configured RegressionExperiment object.
    """
    regression_experiment = RegressionExperiment()
    return regression_experiment.setup(
        data=data, target=target_variable_name, **setup_params
    )


def _extract_target_variable_name(column_names: list[str]) -> str:
    """Extracts the target variable name from a list of column names.

    Args:
        column_names (list[str]): A list of column names.

    Returns:
        str: The name of the target variable.

    Raises:
        ValueError: If multiple target variables are found without an underscore.
        ValueError: If no target variable is found without an underscore.
    """
    target_variable = None
    for name in column_names:
        if "_" not in name:
            if target_variable is not None:
                raise ValueError("Multiple target variables found without underscore.")
            target_variable = name
    if target_variable is None:
        raise ValueError("No target variable found without underscore.")
    return target_variable


def _filter_train_test_data(
    stock_price_table: pd.DataFrame,
    train_test_split_params: dict[str, str],
    filter_value: str,
) -> pd.DataFrame:
    """Filters the training data based on the train-test split parameters.

    Args:
        stock_price_table (pd.DataFrame): The stock price table.
        train_test_split_params (dict[str, str]): The parameters for the train-test
            split. Contains the name of the column indicating the split.
        filter_value (str): The value used to filter the training data. Only rows
            where the train_test_column matches this value will be included in the
            filtered stock price table.

    Returns:
        pd.DataFrame: The filtered stock price table.
    """
    train_test_column = train_test_split_params["train_test_column"]
    return stock_price_table.query(f"{train_test_column} == '{filter_value}'")
