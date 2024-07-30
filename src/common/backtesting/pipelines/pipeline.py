"""Pipeline for price prediction."""

from common.backtesting.functions.evaluation import calculate_performance_metrics
from common.backtesting.functions.returns import (
    adjust_returns_for_trading_costs,
    create_portfolio_returns,
)
from common.backtesting.functions.viz import plot_performance_metrics
from kedro.pipeline import Pipeline, node, pipeline


def create_pipeline(top_level_namespace: str = "") -> Pipeline:
    """Pipeline for machine learning techniques modeling.

    Parameters
    ----------
    top_level_namespace : str
        The namespace for the pipeline.

    Returns
    -------
    Pipeline
        The ML modeling pipeline.

    """
    nodes = [
        node(
            func=create_portfolio_returns,
            inputs={
                "prices": "price_data",
                "weights": "weights",
            },
            outputs="portfolio_returns",
            name="create_portfolio_returns",
            tags=["backtesting"],
        ),
        node(
            func=adjust_returns_for_trading_costs,
            inputs={
                "portfolio_returns": "portfolio_returns",
                "signals": "signals",
                "trading_cost_params": "params:trading_costs",
            },
            outputs="adjusted_portfolio_returns",
            name="adjust_portfolio_returns",
            tags=["backtesting"],
        ),
        node(
            func=calculate_performance_metrics,
            inputs={
                "portfolio_returns": "adjusted_portfolio_returns",
                "performance_metric_params": "params:performance_metrics",
            },
            outputs="performance_metrics",
            name="calculate_performance_metrics",
            tags=["backtesting"],
        ),
        node(
            func=plot_performance_metrics,
            inputs={
                "portfolio_returns": "adjusted_portfolio_returns",
                "plot_performance_params": "params:plot_performance_params",
            },
            outputs=None,
            name="plot_performance_metrics",
            tags=["backtesting"],
        ),
    ]
    return pipeline(nodes, namespace=top_level_namespace)
