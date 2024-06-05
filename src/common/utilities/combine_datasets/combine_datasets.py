"""Functions for combining datasets."""

import functools

import pandas as pd


def concatenate_datasets(*experiment_predictions: tuple[pd.DataFrame]) -> pd.DataFrame:
    """Concatenation of datasets of various versions.

    Concatenating list of dataframes. This is done on the common keys of all dataframes.

    Returns
    -------
        pd.DataFrame: Concatenated dataframe.

    """
    common_columns = list(
        set.intersection(*[set(x.columns) for x in experiment_predictions])
    )

    return pd.concat(
        [x.loc[:, common_columns] for x in experiment_predictions]
    ).reset_index()


def merge_datasets(**kwargs) -> pd.DataFrame:
    """Merge datasets based on the common keys.

    Returns
    -------
        pd.DataFrame: Merged dataframe.

    """
    merge_keys = kwargs["params"]
    dfs_list = []
    for key, df in kwargs.items():
        if key != "params":
            df = df.reset_index()
            rem_cols = set(df.columns).difference(merge_keys)
            df.rename(columns={col: f"{key}_{col}" for col in rem_cols}, inplace=True)
            dfs_list.append(df)

    return functools.reduce(
        lambda df1, df2: pd.merge(df1, df2, on=merge_keys), dfs_list
    )
