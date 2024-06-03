"""Modeling for closing price prediction."""

from typing import TypeVar

import pandas as pd
from pycaret.regression import RegressionExperiment

T = TypeVar("T")


def experiment_setup(
    stock_price_data: pd.DataFrame,
    setup_params: dict[str, str],
) -> RegressionExperiment:
    """_summary_

    Args:
        stock_price_data (pd.DataFrame): _description_
        target_variable_name (str): _description_
        setup_params (dict[str, str]): _description_

    Returns:
        regression_experiment: _description_
    """
    regression_experiment = RegressionExperiment()
    return regression_experiment.setup(
        data=stock_price_data, target=target_variable_name, **setup_params
    )
