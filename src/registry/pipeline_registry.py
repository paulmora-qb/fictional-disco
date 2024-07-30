"""Project pipelines."""

from kedro.pipeline import Pipeline

from data_collection.pipelines import \
    create_data_collection_pipeline as data_collection
from feature_engineering.pipelines import \
    create_pipeline as feature_engineering
from ml_technique_stock_price.pipelines import \
    create_modeling_pipeline as ml_technique_modeling


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns
    -------
        A mapping from pipeline names to ``Pipeline`` objects.

    """
    return {
        # Data Collection Pipelines
        "data_collection": data_collection(),
        # Feature Engineering
        "feature_engineering": feature_engineering(),
        # Stock Predictions: ML Technique Pipelines
        "ml_technique_modeling": ml_technique_modeling(
            top_level_namespace="ml_technique_modeling",
        ),
    }
