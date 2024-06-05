"""Modeling for closing price prediction."""

from typing import TypeVar

import pandas as pd
from pycaret.time_series import TSForecastingExperiment

from common.utilities.add_attribute_to_experiment import add_attribute_to_experiment
from common.utilities.ensure_positive_values import ensure_positive_values
from common.utilities.extract_target_variable_name import extract_target_variable_name
from common.utilities.set_date_index import set_date_index

T = TypeVar("T")


def experiment_setup(
    stock_price_data: pd.DataFrame, modeling_params: dict[str, str]
) -> TSForecastingExperiment:
    """_summary_

    Args:
        stock_price_data (pd.DataFrame): _description_
        modeling_params (dict[str, str]): _description_

    Returns:
        TSForecastingExperiment: _description_
    """
    target_variable_name = extract_target_variable_name(stock_price_data.columns)

    ts_experiment = TSForecastingExperiment()
    stock_price_data = ensure_positive_values(df=stock_price_data)
    stock_price_data = set_date_index(
        date_index_df=stock_price_data,
        frequency_params=modeling_params["frequency_params"],
    )

    ts_experiment = ts_experiment.setup(
        data=stock_price_data,
        target=target_variable_name,
        **modeling_params["setup_params"]
    )

    attr_dict = {
        "feature_columns": stock_price_data.columns.drop(target_variable_name),
        "train_data": stock_price_data,
        "target_vbl_col": target_variable_name,
    }
    add_attribute_to_experiment(ts_experiment, attr_dict)
    return ts_experiment
