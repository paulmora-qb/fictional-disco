"""Modeling for closing price prediction."""

from typing import TypeVar

import pandas as pd
from pycaret.time_series import TSForecastingExperiment

from common.pycaret.time_series.experiment import experiment_setup
from common.utilities.train_test_split import filter_train_test_data

T = TypeVar("T")


def train_model(
    stock_price_table_split: pd.DataFrame,
    modeling_params: dict[str, str],
) -> tuple[TSForecastingExperiment, T]:
    """Train the model for closing price prediction.

    Args:
    ----
        stock_price_table_split (pd.DataFrame): DataFrame containing the stock price
            table, which also contains a column which indicates what should be train
            and what is test.
        modeling_params (dict[str, str]): Parameters for the modeling.

    Returns:
    -------
        tuple[TSForecastingExperiment, T]: The experiment object and the trained model.

    """
    train_stock_price_table = filter_train_test_data(
        stock_price_table=stock_price_table_split,
        train_test_split_params=modeling_params["train_test_split"],
        filter_value="TRAIN",
        drop_column=True,
    )

    experiment = experiment_setup(
        stock_price_data=train_stock_price_table,
        modeling_params=modeling_params,
    )

    # all_models = list(experiment.models().index)
    # unwanted_models = ["auto_arima", "knn_cds_dt"]
    # tested_models = [model for model in all_models if model not in unwanted_models]
    tested_models = ["bats", "theta", "naive", "croston"]

    train_params = modeling_params["train_params"]
    train_params["include"] = tested_models

    base_model = experiment.compare_models(**train_params)
    tuned_model = experiment.tune_model(base_model, **modeling_params["tuned_params"])
    return experiment, experiment.finalize_model(tuned_model)
