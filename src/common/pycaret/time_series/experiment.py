"""Modeling for closing price prediction."""

from typing import TypeVar

import pandas as pd
from pycaret.time_series import TSForecastingExperiment

T = TypeVar("T")


def experiment_setup(
    stock_price_data: pd.DataFrame,
    target_variable_name: str,
    setup_params: dict[str, str],
) -> TSForecastingExperiment:
    """_summary_

    Args:
        stock_price_data (pd.DataFrame): _description_
        target_variable_name (str): _description_
        setup_params (dict[str, str]): _description_

    Returns:
        TSForecastingExperiment: _description_
    """
    ts_experiment = TSForecastingExperiment()
    return ts_experiment.setup(
        data=stock_price_data, target=target_variable_name, **setup_params
    )
