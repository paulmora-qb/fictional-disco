"""Project pipelines."""

from data_collection.pipelines import create_pipeline as data_collection
from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    return {
        "data_collection": data_collection()
    }
