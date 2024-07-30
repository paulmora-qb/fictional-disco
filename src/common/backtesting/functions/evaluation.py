"""Functions to evaluate the performance of a trading strategy."""

import numpy as np
import pandas as pd


def mean_return(returns: pd.Series) -> float:
    """Calculate the mean return of a series.

    Args:
    ----
        returns (pd.Series): Return series.

    Returns:
    -------
        float: Mean return of the series.

    """
    return returns.mean().iloc[0]


def std_deviation(returns: pd.Series) -> float:
    """Calculate the standard deviation of returns.

    Args:
    ----
        returns (pd.Series): Return series.

    Returns:
    -------
        float: Standard deviation of the returns.

    """
    return returns.std().iloc[0]


def cagr(returns: pd.Series) -> float:
    """Calculate the compound annual growth rate (CAGR) of returns.

    Args:
    ----
        returns (pd.Series): Return series.

    Returns:
    -------
        float: CAGR of the returns.

    """
    returns.index = pd.to_datetime(returns.index)
    total_period = (returns.index[-1] - returns.index[0]).days / 365.25
    cumulative_return = (1 + returns).prod() - 1
    cagr_value = (1 + cumulative_return) ** (1 / total_period) - 1
    return cagr_value.iloc[0]


def max_drawdown(returns: pd.Series) -> float:
    """Calculate the maximum drawdown of returns.

    Args:
    ----
        returns (pd.Series): Return series.

    Returns:
    -------
        float: Maximum drawdown of the returns.

    """
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_dd = drawdown.min()
    return max_dd.iloc[0]


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate the Sharpe ratio of returns.

    Args:
    ----
        returns (pd.Series): Return series.
        risk_free_rate (float, optional): The risk free rate that is subtracted from
            the average return. Defaults to 0.0.

    Returns:
    -------
        float: Sharpe ratio of the returns.

    """
    excess_returns = returns - risk_free_rate / 252
    return (excess_returns.mean() / returns.std() * np.sqrt(252)).iloc[0]


def calculate_performance_metrics(
    portfolio_returns: pd.DataFrame, performance_metric_params: str
) -> pd.DataFrame:
    """Calculate performance metrics of a trading strategy.

    Args:
    ----
        portfolio_returns (pd.DataFrame): DataFrame containing the returns of the
            portfolio that should be evaluated.
        performance_metric_params (str): Dictionary containing the parameters for the
            performance metrics calculation

    Returns:
    -------
        pd.DataFrame: DataFrame containing the calculated performance metrics.

    """
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
