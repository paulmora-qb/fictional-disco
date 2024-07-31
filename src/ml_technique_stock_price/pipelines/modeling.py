"""Pipeline for price prediction."""

from functools import partial

import pandas as pd
from common.utilities.multi_variant.pipelines import (
    create_experiment_predictions_variant_concat_pipeline,
)
from kedro.pipeline import Pipeline, node, pipeline


def filter_data(unfiltered_df: pd.DataFrame, cutoff_date: str) -> pd.DataFrame:
    """Filter the data based on the cutoff date.

    Args:
    ----
        unfiltered_df (pd.DataFrame): The unfiltered DataFrame.
        cutoff_date (str): The cutoff date.

    Returns:
    -------
        pd.DataFrame: The filtered DataFrame.

    """
    cutoff_date = pd.to_datetime(cutoff_date)
    unfiltered_df["date"] = pd.to_datetime(unfiltered_df["date"])

    latest_dates = unfiltered_df.groupby("stock_ticker")["date"].max()
    valid_stocks = latest_dates[latest_dates >= cutoff_date].index

    return unfiltered_df[
        unfiltered_df["stock_ticker"].isin(valid_stocks)
        & (unfiltered_df["date"] <= cutoff_date)
    ]


def train_model(stock_prices: pd.DataFrame, modeling_params: dict) -> dict:
    """Train the model.

    Args:
    ----
        stock_prices (pd.DataFrame): The stock prices.
        modeling_params (dict): The modeling parameters.

    Returns:
    -------
        dict: The trained model.

    """
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

    temp_location = "./temp"
    ts_df = TimeSeriesDataFrame.from_data_frame(
        stock_prices, **modeling_params["ts_dataframe"]
    )
    temp_dict = modeling_params["autogluon_init"]
    temp_dict["path"] = temp_location
    predictor = TimeSeriesPredictor(**temp_dict)
    predictor.fit(ts_df, **modeling_params["model_fit"])
    return ts_df, predictor


def inference(stock_prices: pd.DataFrame, predictor: dict) -> pd.DataFrame:
    """Make predictions using the trained model.

    Args:
    ----
        stock_prices (pd.DataFrame): The stock prices.
        predictor (dict): The trained model.

    Returns:
    -------
        pd.DataFrame: The predictions made by the model.

    """
    # if os.path.exists(temp_location):
    #     shutil.rmtree(temp_location)
    predictions = predictor.predict(stock_prices).reset_index()
    return predictions.loc[:, ["item_id", "mean", "timestamp"]]


def stock_selection(
    predictions: pd.DataFrame, stock_price_params: dict[str, str]
) -> pd.DataFrame:
    """Select the stocks based on the predictions.

    Args:
    ----
        predictions (pd.DataFrame): Predictions of the returns.
        stock_price_params (dict[str, str]): Parameters for the stock prices.

    Returns:
    -------
        pd.DataFrame: The selected stocks.

    """
    cutoff_percentile = stock_price_params["cutoff_percentile"]
    log_return_sums = predictions.groupby("item_id")["mean"].sum()

    top_performer = log_return_sums.nlargest(
        int(len(log_return_sums) * cutoff_percentile)
    ).index
    lowest_performer = log_return_sums.nsmallest(
        int(len(log_return_sums) * cutoff_percentile)
    ).index

    predictions["indicator"] = 0
    predictions.loc[predictions["item_id"].isin(list(top_performer)), "indicator"] = 1
    predictions.loc[predictions["item_id"].isin(lowest_performer), "indicator"] = -1

    return pd.DataFrame(predictions).pivot_table(
        index="timestamp", columns="item_id", values="indicator", fill_value=0
    )


def _create_modeling_pipeline(top_level_namespace: str, variant: str) -> Pipeline:
    """Pipeline for machine learning techniques modeling.

    Parameters
    ----------
    top_level_namespace : str
        The namespace for the pipeline.
    variant : str
        The variant for the pipeline.

    Returns
    -------
    Pipeline
        The ML modeling pipeline.

    """
    nodes = [
        node(
            func=partial(filter_data, cutoff_date=variant),
            inputs="price_w_features",
            outputs="filtered_price_w_features",
            name="",
            tags=["modeling"],
        ),
        node(
            func=train_model,
            inputs={
                "stock_prices": "filtered_price_w_features",
                "modeling_params": "params:modeling_params",
            },
            outputs=["ts_df", "predictor"],
            name="train_model",
            tags=["modeling"],
        ),
        node(
            func=inference,
            inputs={
                "stock_prices": "ts_df",
                "predictor": "predictor",
            },
            outputs="predictions",
            name="inference",
            tags=["modeling"],
        ),
        node(
            func=stock_selection,
            inputs={
                "predictions": "predictions",
                "stock_price_params": "params:stock_price_params",
            },
            outputs="signals",
            name="signal_creation",
            tags=["modeling"],
        ),
    ]
    namespace = f"{top_level_namespace}.{variant}"
    return pipeline(
        nodes,
        namespace=namespace,
        inputs={"price_w_features"},
        parameters={
            "modeling_params": f"{top_level_namespace}.modeling_params",
            "stock_price_params": f"{top_level_namespace}.stock_price_params",
        },
    )


def _create_all_relevant_cutoffs(start_year: int, end_year: int) -> list[str]:
    dates = pd.date_range(
        start=f"{start_year}-01-01", end=f"{end_year}-12-31", freq="W-FRI"
    )
    business_dates = dates[dates.isin(pd.bdate_range(dates.min(), dates.max()))]
    return business_dates.strftime("%Y-%m-%d").tolist()


def create_modeling_pipeline(top_level_namespace: str) -> Pipeline:
    """Create the pipeline for the closing price prediction.

    Args:
    ----
        top_level_namespace (str): The top level namespace.
        variants (list[str]): The list of variants to include in the pipeline.

    Returns:
    -------
        Pipeline: The closing price prediction pipeline.

    """
    # variants = _create_all_relevant_cutoffs(start_year=2021, end_year=2023)
    variants = ["2023-01-06", "2023-01-13"]

    return sum(
        _create_modeling_pipeline(
            top_level_namespace=top_level_namespace, variant=variant
        )
        for variant in variants
    ) + create_experiment_predictions_variant_concat_pipeline(
        top_level_namespace=top_level_namespace,
        variants=variants,
        experiment_name="signals",
    )
