"""Pipeline for price prediction."""

from kedro.pipeline import Pipeline, node, pipeline
from closing_price_prediction.functions.preprocessing import (
    subtract_dataframes,
    create_auto_aggregation,
    create_master_table,
)


def create_feature_pipeline() -> Pipeline:
    """Pipeline for machine learning techniques features.

    Returns
    -------
        Pipeline: The ML feature engineering pipeline.

    """
    nodes = [
        node(
            func=subtract_dataframes,
            inputs={"df1": "high", "df2": "low"},
            outputs="high_minus_low",
            name="subtraction_high_minus_low",
            tags=["feature_engineering"],
        ),
        node(
            func=subtract_dataframes,
            inputs={"df1": "close", "df2": "open"},
            outputs="close_minus_open",
            name="subtraction_close_minus_open",
            tags=["feature_engineering"],
        ),
        node(
            func=create_auto_aggregation,
            inputs={
                "stock_prices": "adj_close",
                "aggregation_params": "params:closing_price_prediction.aggregation",
            },
            outputs="price_aggregation",
            name="auto_aggregation",
            tags=["feature_engineering"],
        ),
        node(
            func=create_master_table,
            inputs=[
                "close",
                "high_minus_low",
                "close_minus_open",
                "price_aggregation",
            ],
            outputs="master_table",
            name="create_master_table",
            tags=["feature_engineering"],
        ),
    ]

    return pipeline(nodes)
