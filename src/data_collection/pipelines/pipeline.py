"""Pipeline for data collection."""

from data_collection.functions import data_collection
from kedro.pipeline import Pipeline, node, pipeline


def create_data_collection_pipeline() -> Pipeline:
    """Pipeline for data collection.

    Returns
    -------
        Pipeline: The data collection pipeline.

    """
    nodes = [
        node(
            func=data_collection,
            inputs={
                "sp500_stock_ticker": "sp500_stock_ticker",
                "data_loader_params": "params:data_loader",
            },
            outputs="price_data",
            name="data_collection",
            tags=["data_collection"],
        ),
    ]

    return pipeline(nodes)
