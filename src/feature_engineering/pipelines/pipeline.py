"""Pipeline for feature engineering."""

from kedro.pipeline import Pipeline, node, pipeline
from feature_engineering.functions.preprocessing import (
    create_auto_aggregation,
    create_master_dict,
    subtract_dataframes,
)


def _create_feature_pipeline() -> Pipeline:
    """Pipeline for machine learning techniques features.

    Parameters
    ----------
    top_level_namespace : str
        The namespace for the pipeline.

    Returns
    -------
    Pipeline
        The ML feature engineering pipeline.

    """
    nodes = [
        node(
            func=subtract_dataframes,
            inputs={
                "df1": "high",
                "df2": "low",
                "name": "params:subtraction.high_minus_low",
            },
            outputs="high_minus_low",
            name="subtraction_high_minus_low",
            tags=["feature_engineering"],
        ),
        node(
            func=subtract_dataframes,
            inputs={
                "df1": "close",
                "df2": "open",
                "name": "params:subtraction.close_minus_open",
            },
            outputs="close_minus_open",
            name="subtraction_close_minus_open",
            tags=["feature_engineering"],
        ),
        node(
            func=create_auto_aggregation,
            inputs={
                "stock_prices": "adj_close",
                "aggregation_params": "params:aggregation",
            },
            outputs="price_aggregation",
            name="auto_aggregation",
            tags=["feature_engineering"],
        ),
        node(
            func=create_master_dict,
            inputs=[
                "close",
                "high_minus_low",
                "close_minus_open",
                "price_aggregation",
            ],
            outputs="master_tables",
            name="create_master_table",
            tags=["feature_engineering"],
        ),
    ]

    return pipeline(nodes)


def create_pipeline() -> Pipeline:
    """Create the feature engineering pipeline.

    Returns:
    -------
        Pipeline: The feature engineering pipeline.

    """
    return _create_feature_pipeline()
