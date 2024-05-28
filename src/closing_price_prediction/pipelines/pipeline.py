"""Pipeline for price prediction."""

from kedro.pipeline import Pipeline, node, pipeline
from closing_price_prediction.functions.preprocessing import (
    subtract_dataframes,
    create_auto_aggregation,
    create_master_dict,
)
from closing_price_prediction.functions.modeling import train_model, inference, post_eda


def _create_feature_pipeline() -> Pipeline:
    """Pipeline for machine learning techniques features.

    Returns
    -------
        Pipeline: The ML feature engineering pipeline.

    """
    nodes = [
        node(
            func=subtract_dataframes,
            inputs={
                "df1": "high",
                "df2": "low",
                "name": "params:closing_price_prediction.subtraction.high_minus_low",
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
                "name": "params:closing_price_prediction.subtraction.close_minus_open",
            },
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


# def _create_modeling_pipeline() -> Pipeline:
#     """Pipeline for machine learning techniques modeling.

#     Returns
#     -------
#         Pipeline: The ML modeling pipeline.

#     """
#     nodes = [
#         node(
#             func=train_model,
#             inputs={}
#             )
#     ]

#     return pipeline(nodes)


def create_pipeline() -> Pipeline:
    """Create the pipeline for the closing price prediction.

    Args:
    ----
        top_level_namespace (str): The top level namespace.
        variants (list[str]): The list of variants to include in the pipeline.

    Returns:
    -------
        Pipeline: The closing price prediction pipeline.

    """
    return _create_feature_pipeline() + _create_modeling_pipeline()
