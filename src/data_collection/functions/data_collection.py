"""Functions for data collection."""

import requests
import os
import pandas as pd


def load_prices(
    sp500_data: pd.DataFrame, data_loader_params: dict[str, str]
) -> pd.DataFrame:
    """Create a DataFrame with the price data for the S&P 500 companies.

    Args:
        sp500_data (pd.DataFrame):
        data_loader_params (dict[str, str]): _description_

    Returns:
        pd.DataFrame: _description_
    """
    basic_link = _create_basic_link(data_loader_params)

    list_of_dataframes = [
        _create_pandas_data(
            symbol=symbol, basic_link=basic_link, data_loader_params=data_loader_params
        )
        for symbol in sp500_data["Symbol"]
    ]

    full_df = pd.concat(list_of_dataframes, axis=1)


def _create_pandas_data(
    symbol: str,
    basic_link: str,
    data_loader_params: dict[str, str],
) -> None:
    value_key_name = data_loader_params["time_series_name_key"]
    price_key = data_loader_params["price_key"]

    download_link = basic_link.format(symbol=symbol)
    data = _load_price_data(download_link=download_link, value_key_name=value_key_name)

    date_value_dict = {
        key: float(value[price_key]) for key, value in data[value_key_name].items()
    }
    return pd.DataFrame(data=list(date_value_dict.items()), columns=["Date", symbol])


def _load_price_data(download_link: str, value_key_name: str) -> dict:
    """Load the price data from the Alpha Vantage API.

    Args:
        download_link (str): The download link for the API.

    Returns:
        dict: The price data.
    """
    r = requests.get(download_link)
    if r.status_code != 200:
        raise ValueError(f"Error: {r.status_code}")

    price_dictionary = r.json()

    # if data

    return r.json()


def _create_basic_link(data_loader_params: dict[str, str]) -> str:
    """Create the basic link for the Alpha Vantage API.

    This base link contains all necessary information except the symbol information.

    Args:
        data_loader_params (dict[str, str]): The data loader parameters.

    Returns:
        str: The filled basic link.
    """
    basic_link = data_loader_params["basic_download_link"]
    function = data_loader_params["function"]
    output_size = data_loader_params["output_size"]
    api_key = _retrieve_api_key()

    filled_basic = (
        basic_link + f"function={function}&outputsize={output_size}&apikey={api_key}"
    )

    return filled_basic + "&symbol={symbol}"


def _retrieve_api_key() -> str:
    """Retrieve the Alpha Vantage API key from the environment.

    Raises:
        ValueError: If the API key is not found in the environment.

    Returns:
        str: The API key.
    """
    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
    if api_key is None:
        raise ValueError("API key not found. Please set it as an environment variable.")
    return api_key
