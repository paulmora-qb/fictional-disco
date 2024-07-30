import numpy as np
import pandas as pd
import math
from common.backtesting.functions.evaluation import (
    mean_return,
    std_deviation,
    cagr,
    max_drawdown,
    sharpe_ratio,
    calculate_performance_metrics,
)


def test_mean_return(adjusted_portfolio_returns, evaluation_results):
    result = mean_return(adjusted_portfolio_returns)
    expected = evaluation_results["mean_return"]
    assert math.isclose(result, expected, rel_tol=1e-3)


def test_std_deviation(adjusted_portfolio_returns, evaluation_results):
    result = std_deviation(adjusted_portfolio_returns)
    expected = evaluation_results["std_deviation"]
    assert math.isclose(result, expected, rel_tol=1e-3)


def test_cagr(adjusted_portfolio_returns, evaluation_results):
    result = cagr(adjusted_portfolio_returns)
    expected = evaluation_results["cagr"]
    assert math.isclose(result, expected, rel_tol=1e-3)


def test_max_drawdown(adjusted_portfolio_returns, evaluation_results):
    result = max_drawdown(adjusted_portfolio_returns)
    expected = evaluation_results["max_drawdown"]
    assert math.isclose(result, expected, rel_tol=1e-3)


def test_sharpe_ratio(adjusted_portfolio_returns, evaluation_results):
    result = sharpe_ratio(adjusted_portfolio_returns)
    expected = evaluation_results["sharpe_ratio"]
    assert math.isclose(result, expected, rel_tol=1e-3)


def test_calculate_performance_metrics(
    adjusted_portfolio_returns: pd.DataFrame,
    evaluation_results: pd.DataFrame,
    params_performance_metrics: dict[str, str],
):
    result = calculate_performance_metrics(
        portfolio_returns=adjusted_portfolio_returns,
        performance_metric_params=params_performance_metrics,
    )
    expected_metrics = {
        "mean_return": evaluation_results["mean_return"],
        "variance_return": evaluation_results["std_deviation"],
        "sharpe_ratio": evaluation_results["sharpe_ratio"],
        "cagr": evaluation_results["cagr"],
        "max_drawdown": evaluation_results["max_drawdown"],
        "sharpe_ratio": evaluation_results["sharpe_ratio"],
    }

    for key, _ in expected_metrics.items():
        assert math.isclose(result[key], expected_metrics[key], rel_tol=1e-3)
