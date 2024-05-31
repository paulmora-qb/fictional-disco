"""Functions for data collection."""

import pandas as pd
import yfinance as yf
from tqdm import tqdm


def non_incremental_price_creation(
    sp500_data: pd.DataFrame, data_loader_params: dict[str, str]
) -> dict[str, pd.DataFrame]:
    """Create a DataFrame with the price data for the S&P 500 companies.

    Args:
    ----
        sp500_data (pd.DataFrame): DataFrame containing the symbols of the S&P 500
            companies.
        data_loader_params (dict[str, str]): Parameters relevant for the data loading.
            They contain information such as which columns should be retrieved and for
            which time period.

    Returns:
    -------
        dict[str, pd.DataFrame]: Dictionary containing the retrieved columns for all S&P
            500 companies. The keys are the column names and the values are the
            DataFrames with the stock information.

    """
    relevant_columns = data_loader_params["relevant_columns"]
    list_symbols = list(sp500_data.loc[:, "Symbol"])

    data_dict, valid_stock_symbols = _download_all_stock_information(
        list_symbols=list_symbols, data_loader_params=data_loader_params
    )

    return (
        _create_dataframe_per_column(
            data_dict=data_dict, relevant_columns=relevant_columns
        ),
        valid_stock_symbols,
    )


def _create_dataframe_per_column(
    data_dict: dict[str, pd.DataFrame], relevant_columns: list[str]
) -> dict[str, pd.DataFrame]:
    """Gather the data from the different DataFrames into a single DataFrame per column.

    This function takes the same column from each DataFrame in the dictionary and
    concatenates them.

    Args:
    ----
        data_dict (dict[str, pd.DataFrame]): Keys are the stock symbols and values are
            the DataFrames with the stock information.
        relevant_columns (list[str]): Column names from the DataFrames that we want to
            concatenate.

    Returns:
    -------
        dict[str, pd.DataFrame]: Keys are the column names and values are the
            concatenated DataFrames. In the concatenated DataFrames, the columns are
            the stock symbols.

    """
    return {
        col: pd.concat(
            [df[[col]].rename(columns={col: name}) for name, df in data_dict.items()],
            axis=1,
        )
        for col in relevant_columns
    }


def _download_all_stock_information(
    list_symbols: list[str], data_loader_params: dict[str, str]
) -> dict[str, pd.DataFrame]:
    """Download stock information for all the symbols in the list.

    Args:
    ----
        list_symbols (list[str]): List of stock symbols.
        data_loader_params (dict[str, str]): Parameters for the data loader.

    Returns:
    -------
        dict[str, pd.DataFrame]: Dictionary with the stock information for each symbol.

    """
    data_dict = {}
    valid_stock_symbols = []
    for symbol in tqdm(list_symbols):
        data = _download_price_dataframe(symbol=symbol, params=data_loader_params)
        if not data.empty:
            data_dict[symbol] = data
            valid_stock_symbols.append(symbol)
    return data_dict, valid_stock_symbols


def _download_price_dataframe(
    symbol: str,
    params: dict[str, str],
) -> pd.DataFrame:
    """Download the price data for a given symbol and period.

    Args:
    ----
        symbol (str): Stock symbol.
        params (dict[str, str]): Parameters for the data loader.

    Returns:
    -------
        pd.DataFrame: Stock information.

    """
    period = params["period"]
    return yf.download(symbol, period=period)
