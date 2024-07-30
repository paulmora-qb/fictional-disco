"""Conftest"""

import pandas as pd
import pytest


@pytest.fixture
def price_data() -> pd.DataFrame:
    dates = pd.date_range(start="2024-06-01", periods=5, freq="D")
    data = {
        "AAPL": [130.41, 131.40, 130.85, 133.56, 133.94],
        "GOOGL": [2750.00, 2745.50, 2760.40, 2770.30, 2780.10],
        "MSFT": [310.50, 311.60, 309.90, 312.45, 313.70],
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
def params_performance_metrics() -> dict[str, str]:
    return {"columns": "Adjusted Portfolio Returns"}


@pytest.fixture
def params_plot_performance_params() -> dict[str, str]:
    return {"columns": ["Adjusted Portfolio Returns", "Portfolio Returns"]}
