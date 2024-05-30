"""Modeling for closing price prediction."""

import pandas as pd
from pycaret.regression import RegressionExperiment


def train_model(
    stock_price_table: pd.DataFrame,
    train_params: dict[str, str],
):
    target_variable_name = _extract_target_variable_name(stock_price_table.columns)

    experiment = _experiment_setup(
        data=stock_price_table,
        target_variable_name=target_variable_name,
        train_params=train_params,
    )

    experiment.compare_models()


def inference():
    pass


def post_eda():
    pass


def _experiment_setup(
    data: pd.DataFrame, target_variable_name: str, train_params: dict[str, str]
):
    """Sets up a regression experiment.

    This function initializes a RegressionExperiment object and configures it with the
        provided data and target variable.

    Parameters:
        data (pd.DataFrame): The input data for the experiment.
        target_variable_name (str): The name of the target variable.
        train_params (dict[str, str]): Additional parameters for training.

    Returns:
        RegressionExperiment: The configured RegressionExperiment object.
    """
    regression_experiment = RegressionExperiment()
    return regression_experiment.setup(
        data=data, target=target_variable_name, **train_params
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
