"""Pipeline for price prediction."""

from kedro.pipeline import Pipeline, node, pipeline
from closing_price_prediction.functions.preprocessing import (
    subtract_dataframes,
    create_auto_aggregation,
    create_master_dict,
)
from closing_price_prediction.functions.modeling import train_model, inference, post_eda


def _create_feature_pipeline(top_level_namespace: str) -> Pipeline:
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

    return pipeline(
        nodes,
        inputs={"high", "low", "close", "open", "adj_close"},
        namespace=top_level_namespace,
    )


def _create_modeling_pipeline(top_level_namespace: str, variant: str) -> Pipeline:
    """Pipeline for machine learning techniques modeling.

    Parameters
    ----------
    top_level_namespace : str
        The namespace for the pipeline.

    variant : str
        The variant of the pipeline.

    Returns
    -------
    Pipeline
        The ML modeling pipeline.

    """
    nodes = [
        node(
            func=train_model,
            inputs={
                "stock_price_table": "stock_price_table",
                "train_params": "params:train_params",
            },
            outputs=["experiment", "tuned_model"],
            name="train_model",
            tags=["modeling"],
        )
    ]

    namespace = f"{top_level_namespace}.{variant}"
    return pipeline(
        nodes,
        parameters={
            "params:train_params": f"params:{top_level_namespace}.train_params"
        },
        namespace=namespace,
    )


def create_pipeline(top_level_namespace: str, variants: list[str]) -> Pipeline:
    """Create the pipeline for the closing price prediction.

    Args:
    ----
        top_level_namespace (str): The top level namespace.
        variants (list[str]): The list of variants to include in the pipeline.

    Returns:
    -------
        Pipeline: The closing price prediction pipeline.

    """
    return _create_feature_pipeline(top_level_namespace=top_level_namespace) + sum(
        _create_modeling_pipeline(
            top_level_namespace=top_level_namespace, variant=variant
        )
        for variant in variants
    )
