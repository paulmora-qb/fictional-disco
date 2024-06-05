"""Functions for set date index."""

import pandas as pd


def set_date_index(
    date_index_df: pd.DataFrame, frequency_params: dict[str, str]
) -> pd.DataFrame:
    """Set the correct date index.

    Args:
        date_index_df (pd.DataFrame): DataFrame for which to set the time index.
        frequency_params (dict[str, str]): Frequency parameters file which contains the
            information of what frequency the index should be set to.

    Returns:
        pd.DataFrame: DataFrame with correct time index.
    """
    frequency = frequency_params["frequency"]
    date_index_df = date_index_df.asfreq(frequency)
    date_index_df.index = date_index_df.index.to_period(frequency)
    return date_index_df
