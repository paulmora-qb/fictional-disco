"""Modeling for closing price prediction."""

from typing import TypeVar

import pandas as pd
from pycaret.regression import RegressionExperiment

T = TypeVar("T")


def experiment_setup(
    stock_price_data: pd.DataFrame,
    target_variable_name: str,
    setup_params: dict[str, str],
) -> RegressionExperiment:
    """Experiment setup for pycaret regression.

    Args:
    ----
        stock_price_data (pd.DataFrame): Stock Price Data.
        target_variable_name (str): Name of the target variable.
        setup_params (dict[str, str]): Parameters for the setup.

    Returns:
    -------
        regression_experiment: The regression experiment object.

    """
    regression_experiment = RegressionExperiment()
    return regression_experiment.setup(
        data=stock_price_data, target=target_variable_name, **setup_params
    )
