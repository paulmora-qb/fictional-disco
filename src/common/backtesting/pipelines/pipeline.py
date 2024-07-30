"""Pipeline for price prediction."""

from kedro.pipeline import Pipeline, node, pipeline
import pandas as pd


def create_portfolio_returns(
    prices: pd.DataFrame, weights: pd.DataFrame
) -> pd.DataFrame:
    returns = prices.pct_change(fill_method=None)
    portfolio_returns = (returns * weights).sum(axis=1)
    return portfolio_returns.to_frame(name="Portfolio Returns")


def adjust_returns_for_trading_costs(
    portfolio_returns: pd.DataFrame,
    signals: pd.DataFrame,
    trading_cost_params: dict[str, float],
) -> pd.DataFrame:

    bp_trading_cost = trading_cost_params["bp_trading_cost"] / 10000

    portfolio_returns.loc[:, "Cumulative Returns"] = (
        1 + portfolio_returns.loc[:, "Portfolio Returns"]
    ).cumprod()

    # Calculate the changes in trade signals (buy/sell)
    trade_signals = signals.diff().fillna(
        signals
    )  # TODO: Not sure whether that is correct to fill the values with the original signals since I would usually buy them before the signal is given on the closing price yday.
    # Calculate the position sizes at each time step based on weights and cumulative returns
    position_sizes = portfolio_returns.loc[:, "Cumulative Returns"].shift(1).fillna(1)
    investment_changes = trade_signals.multiply(position_sizes, axis=0).abs()
    # Calculate trading costs
    trading_costs = investment_changes * bp_trading_cost
    portfolio_returns.loc[:, "Total Trading Costs"] = trading_costs.sum(axis=1)
    portfolio_returns.loc[:, "Normalized Trading Costs"] = (
        portfolio_returns.loc[:, "Total Trading Costs"]
        / portfolio_returns.loc[:, "Cumulative Returns"]
    )
    portfolio_returns.loc[:, "Adjusted Portfolio Returns"] = (
        portfolio_returns.loc[:, "Portfolio Returns"]
        - portfolio_returns.loc[:, "Normalized Trading Costs"]
    )
    return portfolio_returns


def mean_return(returns: pd.Series) -> float:
    """Calculate the mean return of returns."""
    return returns.mean()


def std_deviation(returns: pd.Series) -> float:
    """Calculate the standard deviation of returns."""
    return returns.std()


def cagr(returns: pd.Series) -> float:
    """Calculate the Compound Annual Growth Rate (CAGR) of returns."""
    returns.index = pd.to_datetime(returns.index)
    total_period = (returns.index[-1] - returns.index[0]).days / 365.25
    cumulative_return = (1 + returns).prod() - 1
    cagr_value = (1 + cumulative_return) ** (1 / total_period) - 1
    return cagr_value


def max_drawdown(returns: pd.Series) -> float:
    """Calculate the maximum drawdown of returns."""
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_dd = drawdown.min()
    return max_dd


def calmar_ratio(returns: pd.Series) -> float:
    """Calculate the Calmar ratio of returns."""
    return cagr(returns) / abs(max_drawdown(returns))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate the Sharpe ratio of returns."""
    excess_returns = returns - risk_free_rate / 252
    return excess_returns.mean() / returns.std() * np.sqrt(252)


def calculate_performance_metrics(portfolio_returns: pd.DataFrame) -> pd.DataFrame:
    portfolio_returns = portfolio_returns.loc[:, "Adjusted Portfolio Returns"]
    return {
        "mean_return": mean_return(portfolio_returns),
        "variance_return": std_deviation(portfolio_returns),
        "sharpe_ratio": sharpe_ratio(portfolio_returns),
        "cagr": cagr(portfolio_returns),
        "max_drawdown": max_drawdown(portfolio_returns),
        "calmar_ratio": calmar_ratio(portfolio_returns),
        "sharpe_ratio": sharpe_ratio(portfolio_returns),
    }


def plot_performance_metrics(performance_metrics: pd.DataFrame) -> None:
    pass


def create_pipeline(top_level_namespace: str) -> Pipeline:
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
                "trading_costs": "params:trading_costs",
            },
            outputs="adjusted_portfolio_returns",
            name="adjust_portfolio_returns",
            tags=["backtesting"],
        ),
        node(
            func=calculate_performance_metrics,
            inputs={
                "portfolio_returns": "adjusted_portfolio_returns",
            },
            outputs="performance_metrics",
            name="calculate_performance_metrics",
            tags=["backtesting"],
        ),
        node(
            func=plot_performance_metrics,
            inputs={
                "performance_metrics": "performance_metrics",
            },
            outputs=None,
            name="plot_performance_metrics",
            tags=["backtesting"],
        ),
        node(
            func=create_performance_summary,
            inputs={
                "performance_metrics": "performance_metrics",
                "performance_plot": "performance_plot",
            },
            outputs="performance_summary",
            name="create_performance_summary",
            tags=["backtesting"],
        ),
    ]
    return pipeline(nodes)
