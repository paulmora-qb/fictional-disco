"""Functions for data collection."""

import pandas as pd
import yfinance as yf


def data_collection(
    sp500_stock_ticker: list[str], data_loader_params: dict[str, str]
) -> pd.DataFrame:
    """Collect stock information for the S&P 500 stock tickers.

    Args:
    ----
        sp500_stock_ticker (list[str]): List of S&P 500 stock tickers.
        data_loader_params (dict[str, str]): Parameters for the data loader.

    Returns:
    -------
        pd.DataFrame: DataFrame with the stock information.

    """
    return _download_all_stock_information(
        list_symbols=sp500_stock_ticker, data_loader_params=data_loader_params
    )


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
    period = data_loader_params["period"]

    data = yf.download(list_symbols, period=period)
    data.columns = ["_".join(col).strip() for col in data.columns.values]

    long_df = pd.melt(
        data.reset_index(), id_vars="Date", var_name="column", value_name="value"
    )
    long_df[["table", "stock_ticker"]] = long_df["column"].str.split("_", expand=True)
    long_df_cols = long_df.pivot_table(
        index=["Date", "stock_ticker"], columns="table", values="value"
    ).reset_index()
    return _standardize_columns(long_df_cols)


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize the column names of the DataFrame.

    Args:
    ----
        df (pd.DataFrame): DataFrame to standardize.

    """

    def _standardize_column_name(col):
        return col.strip().lower().replace(" ", "_")

    # Apply the function to all column names
    df.columns = [_standardize_column_name(col) for col in df.columns]
    return df
