"""Modeling for closing price prediction."""

from typing import TypeVar

import pandas as pd
from pycaret.time_series import TSForecastingExperiment
from common.utilities.extract_target_variable_name import extract_target_variable_name
from common.utilities.ensure_positive_values import ensure_positive_values

T = TypeVar("T")


def experiment_setup(
    stock_price_data: pd.DataFrame,
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
    target_variable_name = extract_target_variable_name(stock_price_data.columns)

    ts_experiment = TSForecastingExperiment()
    stock_price_data = ensure_positive_values(df=stock_price_data)
    stock_price_data = stock_price_data.asfreq("B")

    return ts_experiment.setup(
        data=stock_price_data, target=target_variable_name, **setup_params
    )
