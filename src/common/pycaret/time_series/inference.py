"""Modeling for closing price prediction."""

from typing import Any, TypeVar

import pandas as pd
from pycaret.time_series import TSForecastingExperiment

from common.utilities.train_test_split import filter_train_test_data
from common.utilities.set_date_index import set_date_index
from common.utilities.add_attribute_to_experiment import add_attribute_to_experiment

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
    test_data = _prepare_exogenous_features(
        stock_price_table=stock_price_table_split,
        modeling_params=modeling_params,
    )
    add_attribute_to_experiment(experiment, {"data": experiment.train_data})
    return _get_prediction_dataframe(
        model=model, test_data=test_data, experiment=experiment
    )


def _get_prediction_dataframe(
    model: T, test_data: pd.DataFrame, experiment: TSForecastingExperiment
) -> pd.DataFrame:
    """_summary_

    Args:
        model (T): _description_
        test_data (pd.DataFrame): _description_
        experiment (TSForecastingExperiment): _description_

    Returns:
        pd.DataFrame: _description_
    """
    exogenous_features = test_data.loc[:, experiment.feature_columns]
    prediction_dataframe = test_data.loc[:, [experiment.target_vbl_col]]

    prediction_dataframe.loc[:, "prediction_label"] = experiment.predict_model(
        model, X=exogenous_features, fh=len(exogenous_features)
    )
    return prediction_dataframe


def _prepare_exogenous_features(
    stock_price_table: pd.DataFrame,
    modeling_params: dict[str, str],
) -> pd.DataFrame:
    """_summary_

    Args:
        experiment (T):
        inference_data (pd.DataFrame): _description_
        modeling_params (dict[str, str]): _description_

    Returns:
        pd.DataFrame: _description_
    """
    test_data = filter_train_test_data(
        stock_price_table=stock_price_table,
        train_test_split_params=modeling_params["train_test_split"],
        filter_value="TEST",
        drop_column=True,
    )
    return set_date_index(
        date_index_df=test_data,
        frequency_params=modeling_params["frequency_params"],
    )
