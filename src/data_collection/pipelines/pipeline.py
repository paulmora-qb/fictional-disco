"""Pipeline for data collection."""

from kedro.pipeline import Pipeline, node, pipeline
from data_collection.functions import load_prices


def create_pipeline() -> Pipeline:
    """Pipeline for data collection.

    Returns
    -------
        Pipeline: The data collection pipeline.

    """
    nodes = [
        node(
            func=load_prices,
            inputs={
                "sp500_data": "sp500_data",
                "data_loader_params": "params:data_loader",
            },
            outputs="full_df_sp500_data",
            name="data_collection",
            tags=["data_collection"],
        )
    ]

    return pipeline(nodes)
