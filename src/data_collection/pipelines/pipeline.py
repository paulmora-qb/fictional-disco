"""Pipeline for data collection."""

from kedro.pipeline import Pipeline, node, pipeline
from data_collection.functions import (
    non_incremental_price_creation,
)


def create_non_incremental_pipeline() -> Pipeline:
    """Pipeline for data collection.

    Returns
    -------
        Pipeline: The data collection pipeline.

    """
    nodes = [
        node(
            func=non_incremental_price_creation,
            inputs={
                "sp500_data": "sp500_data",
                "data_loader_params": "params:data_loader",
            },
            outputs="price_data",
            name="data_collection",
            tags=["data_collection"],
        ),
    ]

    return pipeline(nodes)
