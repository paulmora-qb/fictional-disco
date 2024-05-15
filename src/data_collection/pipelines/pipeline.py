"""Pipeline for data collection."""

from kedro import Pipeline, node, pipeline
from data_collection.functions import data_collection

def create_pipeline() -> Pipeline:
    """Pipeline for data collection.

    Returns:
        Pipeline: The data collection pipeline.
    """
    nodes = [
        node(
            func=data_collection,
            inputs=["sp500_data"],
            outputs="full_df_sp500_data",
            name="data_collection",
            tags=["data_collection"]
        )
    ]

    return pipeline(nodes)
