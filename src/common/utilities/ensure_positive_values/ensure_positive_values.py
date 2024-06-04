"""Function to ensure positive values."""

import pandas as pd


def ensure_positive_values(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure to convert negative values to positive values.

    This function goes through all columns and checks whether there is any negative
    value. If there is a negative value, then the minimum negative is added plus
    one. This is only done of course for columns which are not of type string.

    Args:
    ----
        df (pd.DataFrame): DataFrame to ensure positive values.

    Returns:
    -------
        pd.DataFrame: DataFrame with positive values.

    """
    for column in df.columns:
        if df[column].dtype != "object":
            min_value = df[column].min()
            if min_value < 0:
                df[column] = df[column] + abs(min_value) + 1
    return df
