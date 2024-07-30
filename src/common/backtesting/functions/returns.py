"""Functions for return creation."""

import pandas as pd


def create_portfolio_returns(
    prices: pd.DataFrame, weights: pd.DataFrame
) -> pd.DataFrame:
    """Creation of portfolio returns.

    Args:
    ----
        prices (pd.DataFrame): Price dataframe with date index and stocks as the
            columns. The prices are in the cells.
        weights (pd.DataFrame): Weight dataframe with date index and stocks as the
            columns. The weights are in the cells.

    Returns:
    -------
        pd.DataFrame: Portfolio returns dataframe with date index and one column
            "Portfolio Returns".

    """
    returns = prices.pct_change(fill_method=None)
    portfolio_returns = (returns * weights).sum(axis=1)
    return portfolio_returns.to_frame(name="Portfolio Returns")


def adjust_returns_for_trading_costs(
    portfolio_returns: pd.DataFrame,
    signals: pd.DataFrame,
    trading_cost_params: dict[str, float],
) -> pd.DataFrame:
    """Adjust the portfolio returns for trading costs.

    Args:
    ----
        portfolio_returns (pd.DataFrame): Portfolio returns. DataFrame with one
            column.
        signals (pd.DataFrame): Signals for the trades. This DataFrame contains the
            date as the index and the stocks on the columns. The signals are in the
            cells.
        trading_cost_params (dict[str, float]): Dictionary with the trading costs.
            The key is "bp_trading_cost" and the value is the trading cost in basis
            points.

    Returns:
    -------
        pd.DataFrame: Adjusted portfolio returns with the following columns:
            - Cumulative Returns
            - Total Trading Costs
            - Normalized Trading Costs
            - Portfolio Returns
            - Adjusted Portfolio Returns

    """
    bp_trading_cost = trading_cost_params["bp_trading_cost"] / 10000

    portfolio_returns.loc[:, "Cumulative Returns"] = (
        1 + portfolio_returns.loc[:, "Portfolio Returns"]
    ).cumprod()

    # Calculate the changes in trade signals (buy/sell)
    trade_signals = signals.diff().fillna(
        signals
    )  # TODO: Not sure whether that is correct to fill the values with the original
    # TODO: signals since I would usually buy them before the signal is given on the
    # TODO: closing price yday.
    # Calculate the position sizes at each time step
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
