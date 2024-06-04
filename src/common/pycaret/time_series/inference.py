"""Modeling for closing price prediction."""

from typing import Any, TypeVar

import pandas as pd
from pycaret.time_series import TSForecastingExperiment

from common.utilities.train_test_split import filter_train_test_data

T = TypeVar("T")


def inference(
    stock_price_table_split: pd.DataFrame,
    experiment: TSForecastingExperiment,
    model: T,
    modeling_params: dict[str, Any],
) -> pd.DataFrame:
    """Make predictions using the trained model.

    Args:
    ----
        stock_price_table_split (pd.DataFrame): DataFrame containing the stock price
            table, which also contains a column which indicates what should be train
            and what is test.
        experiment (RegressionExperiment): The experiment object.
        model (T): The trained model.
        modeling_params (dict[str, Any]): Parameters for the modeling.

    Returns:
    -------
        pd.DataFrame: The predictions made by the model.

    """
    test_data = filter_train_test_data(
        stock_price_table=stock_price_table_split,
        train_test_split_params=modeling_params["train_test_split"],
        filter_value="TEST",
        drop_column=True,
    )
    train_data = filter_train_test_data(
        stock_price_table=stock_price_table_split,
        train_test_split_params=modeling_params["train_test_split"],
        filter_value="TRAIN",
        drop_column=True,
    )

    train_data = train_data.asfreq("B")
    train_data.index = train_data.index.to_period("B")
    test_data = test_data.asfreq("B")
    test_data.index = test_data.index.to_period("B")
    experiment.data = train_data

    return (
        experiment.predict_model(model, X=train_data, fh=100),
        experiment.pull(),
    )
