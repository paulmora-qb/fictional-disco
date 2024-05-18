"""Functions for data collection."""

import pandas as pd
from tqdm import tqdm

import yfinance as yf

data = yf.download("AAPL", period="10y")


def non_incremental_price_creation(
    sp500_data: pd.DataFrame, data_loader_params: dict[str, str]
) -> pd.DataFrame:
    """Create a DataFrame with the price data for the S&P 500 companies.

    Args:
    ----
        sp500_data (pd.DataFrame):
        data_loader_params (dict[str, str]): _description_

    Returns:
    -------
        pd.DataFrame: _description_

    """
    list_symbols = list(sp500_data.loc[:, "Symbol"])
    list_of_price_data = [
        _create_price_dataframe(symbol=symbol, params=data_loader_params)
        for symbol in tqdm(list_symbols)
    ]
    return pd.concat(list_of_price_data, axis=1)


def _create_price_dataframe(
    symbol: str,
    params: dict[str, str],
) -> pd.DataFrame:
    """_summary_

    Args:
    ----
        symbol (str): _description_
        params (dict[str, str]): _description_

    Returns:
    -------
        pd.DataFrame: _description_

    """
    period = params["period"]
    price_column = params["price_column"]

    data = yf.download(symbol, period=period)
    price_data = data.loc[:, price_column]
    price_data.name = symbol
    return price_data
