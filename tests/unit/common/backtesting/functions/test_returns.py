"""Test for the return functions."""

import pandas as pd
from common.backtesting.functions.returns import (
    create_portfolio_returns,
    adjust_returns_for_trading_costs,
)


def test_create_portfolio_returns(
    price_data: pd.DataFrame, weights: pd.DataFrame, portfolio_returns: pd.DataFrame
) -> pd.DataFrame:
    """Pytest"""
    actual_portfolio_returns = create_portfolio_returns(price_data, weights)
    pd.testing.assert_frame_equal(
        actual_portfolio_returns, portfolio_returns, atol=1e-3
    )


def test_adjust_returns_for_trading_costs(
    portfolio_returns: pd.DataFrame,
    signals: pd.DataFrame,
    params_trading_costs: dict[str, float],
    adjusted_portfolio_returns: pd.DataFrame,
    adjusted_returns_columns: list[str],
) -> pd.DataFrame:
    """Pytest"""
    complete_df = adjust_returns_for_trading_costs(
        portfolio_returns, signals, params_trading_costs
    )
    columns = list(complete_df.columns)
    actual_adjusted_portfolio_returns = complete_df.loc[
        :, ["Adjusted Portfolio Returns"]
    ]

    assert columns == adjusted_returns_columns
    pd.testing.assert_frame_equal(
        actual_adjusted_portfolio_returns, adjusted_portfolio_returns, atol=1e-3
    )
