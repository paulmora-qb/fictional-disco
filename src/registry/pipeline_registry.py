"""Project pipelines."""

from data_collection.pipelines import (
    create_non_incremental_pipeline as non_incremental_data_collection,
)
from closing_price_prediction.pipelines import (
    create_feature_pipeline as ml_technique_pipeline,
)
from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns
    -------
        A mapping from pipeline names to ``Pipeline`` objects.

    """
    return {
        "non_incremental_data_collection": non_incremental_data_collection(),
        "ml_technique_pipeline": ml_technique_pipeline(),
    }
