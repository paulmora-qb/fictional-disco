"""Project pipelines."""

from data_collection.pipelines import (
    create_non_incremental_pipeline as non_incremental_data_collection,
)
from closing_price_prediction.pipelines import (
    create_pipeline as ml_technique_pipeline,
)
from kedro.pipeline import Pipeline
import pickle

with open("data/01_raw/list_stock_symbols.pkl", "rb") as f:
    list_stock_symbols = pickle.load(f)

DYNAMIC_PIPELINES_MAPPING = {"stock_symbols": list_stock_symbols}


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns
    -------
        A mapping from pipeline names to ``Pipeline`` objects.

    """
    return {
        "non_incremental_data_collection": non_incremental_data_collection(),
        "ml_technique_pipeline": ml_technique_pipeline(
            top_level_namespace="closing_price_prediction",
            variants=["MMM"],
        ),
    }
