"""Functions for set date index."""

import pandas as pd


def set_date_index(
    date_index_df: pd.DataFrame, freq_params: dict[str, str]
) -> pd.DataFrame:
    """_summary_

    Args:
        date_index_df (pd.DataFrame): _description_
        frequency (str, optional): _description_. Defaults to "B".

    Returns:
        pd.DataFrame: _description_
    """
    frequency = freq_params["frequency"]
    date_index_df = date_index_df.asfreq(frequency)
    date_index_df.index = date_index_df.index.to_period(frequency)
    return date_index_df
