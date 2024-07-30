"""Integration test for pipeline"""

import logging

import pandas as pd
from kedro.io import DataCatalog
from kedro.runner import SequentialRunner

from common.backtesting.pipelines import create_pipeline


def test_create_pipeline(
    caplog,
    price_data: pd.DataFrame,
    signals: pd.DataFrame,
    weights: pd.DataFrame,
    params_trading_costs: dict[str, str],
    params_performance_metrics: dict[str, str],
    params_plot_performance_params: dict[str, str],
):
    pipeline = (
        create_pipeline()
        .from_nodes("create_portfolio_returns")
        .to_nodes("plot_performance_metrics")
    )
    catalog = DataCatalog()
    catalog.add_feed_dict(
        {
            "price_data": price_data,
            "weights": weights,
            "signals": signals,
            "params:trading_costs": params_trading_costs,
            "params:performance_metrics": params_performance_metrics,
            "params:plot_performance_params": params_plot_performance_params,
        }
    )

    caplog.set_level(logging.DEBUG, logger="kedro")
    successful_run_msg = "Pipeline execution completed successfully."

    SequentialRunner().run(pipeline, catalog)

    assert successful_run_msg in caplog.text
