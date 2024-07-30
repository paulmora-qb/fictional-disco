"""Functions for return creation."""

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
