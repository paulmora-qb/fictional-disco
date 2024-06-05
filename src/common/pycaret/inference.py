"""Modeling for closing price prediction."""

from typing import Any, TypeVar

import pandas as pd
from pycaret.regression import RegressionExperiment

from common.utilities.train_test_split import filter_train_test_data

T = TypeVar("T")


def inference(
    stock_price_table_split: pd.DataFrame,
    experiment: RegressionExperiment,
    model: T,
) -> pd.DataFrame:
    """Make predictions using the trained model.

    Args:
    ----
        stock_price_table_split (pd.DataFrame): Dataframe containing the stock price
            table, which also contains a column which indicates what should be train
            and what is test.
        experiment (RegressionExperiment): The experiment object.
        model (T): The trained model.
        modeling_params (dict[str, Any]): Parameters for the modeling.

    Returns:
    -------
        pd.DataFrame: The predictions made by the model.

    """
    return experiment.predict_model(model, data=stock_price_table_split)
