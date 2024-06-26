"""Pipeline for price prediction."""

from kedro.pipeline import Pipeline, node, pipeline

from common.plotting import line_plot
from common.pycaret import inference, train_model
from common.utilities.multi_variant.pipelines import (
    create_experiment_predictions_variant_concat_pipeline,
    create_experiment_predictions_variant_merge_pipeline)
from common.utilities.train_test_split import train_test_split


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
            func=train_test_split,
            inputs={
                "stock_price_table": "stock_price_table",
                "modeling_params": "params:modeling_params",
            },
            outputs="stock_price_table_split",
            name="train_test_split",
            tags=["modeling"],
        ),
        node(
            func=train_model,
            inputs={
                "stock_price_table_split": "stock_price_table_split",
                "modeling_params": "params:modeling_params",
            },
            outputs=["train_experiment", "finalized_model", "performance"],
            name="train_model",
            tags=["modeling"],
        ),
        node(
            func=inference,
            inputs={
                "stock_price_table_split": "stock_price_table_split",
                "experiment": "train_experiment",
                "model": "finalized_model",
            },
            outputs="predictions",
            name="inference",
            tags=["modeling"],
        ),
        node(
            func=line_plot,
            inputs={
                "predictions": "predictions",
                "modeling_params": "params:modeling_params",
            },
            outputs="line_plot",
            name="post_eda",
            tags=["modeling"],
        ),
    ]

    namespace = f"{top_level_namespace}.{variant}"
    return pipeline(
        nodes,
        parameters={
            "params:modeling_params": f"params:{top_level_namespace}.modeling_params",
        },
        namespace=namespace,
    )


def create_modeling_pipeline(top_level_namespace: str, variants: list[str]) -> Pipeline:
    """Create the pipeline for the closing price prediction.

    Args:
    ----
        top_level_namespace (str): The top level namespace.
        variants (list[str]): The list of variants to include in the pipeline.

    Returns:
    -------
        Pipeline: The closing price prediction pipeline.

    """
    return (
        sum(
            _create_modeling_pipeline(
                top_level_namespace=top_level_namespace, variant=variant
            )
            for variant in variants
        )
        + create_experiment_predictions_variant_merge_pipeline(
            top_level_namespace=top_level_namespace,
            variants=variants,
            experiment_name="predictions",
        )
        + create_experiment_predictions_variant_concat_pipeline(
            top_level_namespace=top_level_namespace,
            variants=variants,
            experiment_name="performance",
        )
    )
