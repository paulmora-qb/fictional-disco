"""Modeling for closing price prediction."""

from typing import TypeVar

import pandas as pd
from common.utilities.train_test_split import filter_train_test_data
from common.utilities.extract_target_variable_name import extract_target_variable_name
from common.pycaret.experiment import experiment_setup

T = TypeVar("T")


def train_model(
    stock_price_table_split: pd.DataFrame,
    modeling_params: dict[str, str],
):
    """Trains a model for closing price prediction.

    Args:
    ----
        stock_price_table_split (pd.DataFrame): Dataframe containing the stock price
            table, which also contains a column which indicates what should be train
            and what is test.
        modeling_params (dict[str, str]): Parameters for the modeling.
            Contains the subsections:

            `train_test_split` (dict[str, str]): Parameters for the train-test split.

            `setup_params` (dict[str, str]): Parameters for the experiment setup.

            `train_params` (dict[str, str]): Parameters for training the model.

    Returns:
    -------
        RegressionExperiment, T: The experiment object and the trained model.

    """
    target_variable_name = extract_target_variable_name(stock_price_table_split.columns)

    train_stock_price_table = filter_train_test_data(
        stock_price_table=stock_price_table_split,
        train_test_split_params=modeling_params["train_test_split"],
        filter_value="TRAIN",
        drop_column=True,
    )

    experiment = experiment_setup(
        stock_price_data=train_stock_price_table,
        target_variable_name=target_variable_name,
        setup_params=modeling_params["setup_params"],
    )

    base_model = experiment.compare_models(**modeling_params["train_params"])
    model = experiment.tune_model(base_model, **modeling_params["tuned_params"])
    performance_data = _extract_performance_information(
        experiment, target_variable_name
    )
    return experiment, experiment.finalize_model(model), performance_data


def _extract_performance_information(
    experiment: T, target_variable_name
) -> pd.DataFrame:
    """Extract performance information from the experiment.

    Args:
        experiment (T): The experiment object.
        target_variable_name (_type_): The name of the target variable column.

    Returns:
        pd.DataFrame: The performance data.
    """

    performance_data = experiment.pull()
    performance_data = performance_data.loc[["Mean"], :]
    performance_data.index = [target_variable_name]
    return performance_data
