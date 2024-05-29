"""Modeling for closing price prediction."""

import pandas as pd
from pycaret.regression import RegressionExperiment


def _experiment_setup(
    data: pd.DataFrame, setup_params: dict[str, str], target_variable: str
):
    a = 1


def train_model(
    stock_price_table: pd.DataFrame,
    train_params: dict[str, str],
):
    experiment = _experiment_setup()


def inference():
    pass


def post_eda():
    pass
