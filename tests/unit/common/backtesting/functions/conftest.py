"""Conftest"""

import pytest
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Set Matplotlib to use a non-interactive backend
matplotlib.use("Agg")


@pytest.fixture
def price_data() -> pd.DataFrame:
    dates = pd.date_range(start="2024-06-01", periods=5, freq="D")
    data = {
        "AAPL": [130, 135, 140, 145, 150],
        "GOOGL": [2750, 2765, 2780, 2795, 2810],
        "MSFT": [310, 320, 330, 340, 350],
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def signals() -> pd.DataFrame:
    dates = pd.date_range(start="2024-06-01", periods=5, freq="D")
    signals = {
        "AAPL": [1, 0, -1, 1, 0],
        "GOOGL": [0, 1, 1, 0, -1],
        "MSFT": [-1, 0, 1, 0, 1],
    }
    return pd.DataFrame(signals, index=dates)


@pytest.fixture
def weights() -> pd.DataFrame:
    dates = pd.date_range(start="2024-06-01", periods=5, freq="D")
    weights = {
        "AAPL": [0.5, 0, 0, 0.4, 0],
        "GOOGL": [0, 0.6, 0.5, 0, 0],
        "MSFT": [0, 0, 0.5, 0, 0.6],
    }
    return pd.DataFrame(weights, index=dates)


@pytest.fixture
def params_trading_costs() -> dict[str, str]:
    return {"bp_trading_cost": 10}


@pytest.fixture
def portfolio_returns() -> pd.DataFrame:
    data = {
        "Portfolio Returns": [0.000000, 0.003273, 0.018337, 0.014286, 0.017647],
    }
    dates = pd.date_range(start="2024-06-01", periods=5, freq="D")
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def adjusted_portfolio_returns() -> pd.DataFrame:
    data = {
        "Adjusted Portfolio Returns": [
            -0.002000,
            0.000283,
            0.016373,
            0.010342,
            0.014699,
        ],
    }
    dates = pd.date_range(start="2024-06-01", periods=5, freq="D")
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def adjusted_returns_columns() -> list[str]:
    return [
        "Portfolio Returns",
        "Cumulative Returns",
        "Total Trading Costs",
        "Normalized Trading Costs",
        "Adjusted Portfolio Returns",
    ]


@pytest.fixture
def evaluation_results():
    return {
        "mean_return": 0.0079394,
        "std_deviation": 0.008367,
        "cagr": 35.522179,
        "max_drawdown": 0.0,
        "sharpe_ratio": 15.06388,
    }


@pytest.fixture
def return_df(
    adjusted_portfolio_returns: pd.DataFrame, portfolio_returns: pd.DataFrame
) -> pd.DataFrame:
    return pd.concat([adjusted_portfolio_returns, portfolio_returns], axis=1)


@pytest.fixture
def figure():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6])
    return fig


@pytest.fixture
def params_performance_metrics() -> dict[str, str]:
    return {"columns": ["Adjusted Portfolio Returns"]}


@pytest.fixture
def params_plot_performance_params() -> dict[str, str]:
    return {"columns": ["Adjusted Portfolio Returns", "Portfolio Returns"]}
