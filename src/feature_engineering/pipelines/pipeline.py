"""Pipeline for feature engineering."""

from kedro.pipeline import Pipeline, node, pipeline

from feature_engineering.functions import (basic_arithmetic,
                                           calculate_rolling_aggregations,
                                           shift_features)


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
            func=basic_arithmetic,
            inputs={
                "price_data": "price_data",
                "arithmetic_params": "params:arithmetic",
            },
            outputs="price_data_temp1",
            name="price_data_temp1",
            tags=["feature_engineering"],
        ),
        node(
            func=calculate_rolling_aggregations,
            inputs={
                "price_data": "price_data_temp1",
                "aggregation_params": "params:aggregation",
            },
            outputs="price_data_temp2",
            name="auto_aggregation",
            tags=["feature_engineering"],
        ),
        node(
            func=shift_features,
            inputs={
                "price_data": "price_data_temp2",
                "shift_params": "params:shift",
            },
            outputs="price_data_final",
            name="price_data_final",
            tags=["feature_engineering"],
        ),
    ]

    return pipeline(nodes)


def create_pipeline() -> Pipeline:
    """Create the feature engineering pipeline.

    Returns
    -------
        Pipeline: The feature engineering pipeline.

    """
    return _create_feature_pipeline()
