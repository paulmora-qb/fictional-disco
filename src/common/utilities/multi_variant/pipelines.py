"""Pipelines for multi variant output."""

from common.utilities.combine_datasets import concatenate_datasets, merge_datasets
from kedro.pipeline import Pipeline, node, pipeline


def create_experiment_predictions_variant_concat_pipeline(
    top_level_namespace: str,
    variants: list[str],
    experiment_name: str,
) -> Pipeline:
    """Create pipeline that concatenates DataFrames when having multiple variants.

    Args:
    ----
        top_level_namespace (str): Namespace for the pipeline.
        variants (list[str]): List of variants to include in the pipeline.
        experiment_name (str): Name of the experiment.

    Returns:
    -------
        Pipeline: The pipeline for concatenating DataFrames.

    """
    exp_pred_inputs_list = [f"{variant}.{experiment_name}" for variant in variants]
    nodes = [
        node(
            func=concatenate_datasets,
            inputs=exp_pred_inputs_list,
            outputs=f"{experiment_name}_concatenated",
            name=f"{experiment_name}_concatenated",
            tags=["inference"],
        )
    ]

    return pipeline(nodes, namespace=top_level_namespace)


def create_experiment_predictions_variant_merge_pipeline(
    top_level_namespace: str,
    variants: list[str],
    experiment_name: str,
) -> Pipeline:
    """Create pipeline that merges DataFrames when having multiple variants.

    Args:
    ----
        top_level_namespace (str): Namespace for the pipeline.
        variants (list[str]): List of variants to include in the pipeline.
        experiment_name (str): Name of the experiment.

    Returns:
    -------
        Pipeline: The pipeline for merging DataFrames.

    """
    exp_pred_inputs_list = [f"{variant}.{experiment_name}" for variant in variants]
    exp_pred_inputs = dict(zip(variants, exp_pred_inputs_list))
    exp_pred_inputs["params"] = "params:predictions.merge_cols"
    nodes = [
        node(
            func=merge_datasets,
            inputs=exp_pred_inputs,
            outputs=f"{experiment_name}_merged",
            name=f"{experiment_name}_merged",
            tags=["inference"],
        )
    ]

    return pipeline(nodes, namespace=top_level_namespace)
