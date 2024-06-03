"""Modeling for closing price prediction."""

from typing import TypeVar

import pandas as pd
from common.utilities.extract_target_variable_name import extract_target_variable_name
from common.utilities.train_test_split import filter_train_test_data
from pycaret.time_series import TSForecastingExperiment
from common.pycaret.time_series.experiment import experiment_setup

T = TypeVar("T")


def train_model(
    stock_price_table_split: pd.DataFrame,
    modeling_params: dict[str, str],
):
    """_summary_

    Args:
        stock_price_table_split (pd.DataFrame): _description_
        modeling_params (dict[str, str]): _description_

    Returns:
        _type_: _description_
    """
    target_variable_name = extract_target_variable_name(stock_price_table_split.columns)

    train_stock_price_table = filter_train_test_data(
        stock_price_table=stock_price_table_split,
        train_test_split_params=modeling_params["train_test_split"],
        filter_value="TRAIN",
    )

    experiment = experiment_setup(
        stock_price_data=train_stock_price_table,
        target_variable_name=target_variable_name,
        setup_params=modeling_params["setup_params"],
    )

    base_model = experiment.compare_models(**modeling_params["train_params"])
    tuned_model = experiment.tune_model(base_model, **modeling_params["tuned_params"])
    return experiment, experiment.finalize_model(tuned_model)
