"""Functions to evaluate the performance of a trading strategy."""

import numpy as np
import pandas as pd


def mean_return(returns: pd.Series) -> float:
    """Calculate the mean return of returns."""
    return returns.mean().iloc[0]


def std_deviation(returns: pd.Series) -> float:
    """Calculate the standard deviation of returns."""
    return returns.std().iloc[0]


def cagr(returns: pd.Series) -> float:
    """Calculate the Compound Annual Growth Rate (CAGR) of returns."""
    returns.index = pd.to_datetime(returns.index)
    total_period = (returns.index[-1] - returns.index[0]).days / 365.25
    cumulative_return = (1 + returns).prod() - 1
    cagr_value = (1 + cumulative_return) ** (1 / total_period) - 1
    return cagr_value.iloc[0]


def max_drawdown(returns: pd.Series) -> float:
    """Calculate the maximum drawdown of returns."""
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_dd = drawdown.min()
    return max_dd.iloc[0]


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate the Sharpe ratio of returns."""
    excess_returns = returns - risk_free_rate / 252
    return (excess_returns.mean() / returns.std() * np.sqrt(252)).iloc[0]


def calculate_performance_metrics(
    portfolio_returns: pd.DataFrame, performance_metric_params: str
) -> pd.DataFrame:
    columns = performance_metric_params["columns"]
    portfolio_returns = portfolio_returns.loc[:, columns]
    return {
        "mean_return": mean_return(portfolio_returns),
        "variance_return": std_deviation(portfolio_returns),
        "sharpe_ratio": sharpe_ratio(portfolio_returns),
        "cagr": cagr(portfolio_returns),
        "max_drawdown": max_drawdown(portfolio_returns),
        "sharpe_ratio": sharpe_ratio(portfolio_returns),
    }
