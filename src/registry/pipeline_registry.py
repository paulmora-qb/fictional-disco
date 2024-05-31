"""Project pipelines."""

import pickle

from kedro.pipeline import Pipeline

from data_collection.pipelines import \
    create_non_incremental_pipeline as non_incremental_data_collection
from ml_technique_stock_price.pipelines import \
    create_feature_pipeline as ml_technique_feature_engineering
from ml_technique_stock_price.pipelines import \
    create_modeling_pipeline as ml_technique_modeling

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
        # Data Collection Pipelines
        "non_incremental_data_collection": non_incremental_data_collection(),
        # Stock Predictions: ML Technique Pipelines
        "ml_technique_feature_engineering": ml_technique_feature_engineering(
            top_level_namespace="closing_price_prediction"
        ),
        "ml_technique_modeling": ml_technique_modeling(
            top_level_namespace="closing_price_prediction",
            variants=["MMM", "AOS", "ABT", "ABBV", "ACN"],
        ),
        # # Stock Predictions: Time Series Pipelines
        # "ts_technique_modeling": ts_technique_modeling(
        #     top_level_namespace="ts_technique_modeling",
        #     variants=["MMM", "AOS", "ABT", "ABBV", "ACN"],
        # ),
    }
