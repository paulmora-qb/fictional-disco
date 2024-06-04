"""Modeling for closing price prediction."""

from typing import TypeVar

import pandas as pd
from pycaret.time_series import TSForecastingExperiment

from common.utilities.add_attribute_to_experiment import \
    add_attribute_to_experiment
from common.utilities.ensure_positive_values import ensure_positive_values
from common.utilities.extract_target_variable_name import \
    extract_target_variable_name

T = TypeVar("T")


def experiment_setup(
    stock_price_data: pd.DataFrame,
    setup_params: dict[str, str],
) -> TSForecastingExperiment:
    """_summary_

    Args:
    ----
        stock_price_data (pd.DataFrame): _description_
        target_variable_name (str): _description_
        setup_params (dict[str, str]): _description_

    Returns:
    -------
        TSForecastingExperiment: _description_

    """
    target_variable_name = extract_target_variable_name(stock_price_data.columns)

    ts_experiment = TSForecastingExperiment()
    stock_price_data = ensure_positive_values(df=stock_price_data)
    stock_price_data = stock_price_data.asfreq("B")

    ts_experiment = ts_experiment.setup(
        data=stock_price_data, target=target_variable_name, **setup_params
    )

    stock_price_data.index = stock_price_data.index.to_period("B")
    attr_dict = {
        "master_table_columns": list(stock_price_data.columns),
        "data": stock_price_data,
        "target_vbl_col": target_variable_name,
    }
    for attr_name, attr in attr_dict.items():
        add_attribute_to_experiment(ts_experiment, attr_name, attr)
    return ts_experiment
