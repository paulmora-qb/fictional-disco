"""Pipeline for data collection."""

from kedro.pipeline import Pipeline, node, pipeline

from data_collection.functions import data_collection


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
                "sp500_data": "sp500_data",
                "data_loader_params": "params:data_loader",
            },
            outputs=["price_data", "valid_stock_symbols"],
            name="data_collection",
            tags=["data_collection"],
        ),
    ]

    return pipeline(nodes)
