"""Integration test for pipeline"""

from common.backtesting.pipelines import create_pipeline
import logging
from kedro.io import DataCatalog
from kedro.runner import SequentialRunner


def test_create_pipeline(caplog, dummy_data, dummy_parameters):
    pipeline = (
        create_pipeline()
        .from_nodes("create_portfolio_returns")
        .to_nodes("create_performance_summary")
    )
    catalog = DataCatalog()
    catalog.add_feed_dict(
        {
            "model_input_table": dummy_data,
            "params:model_options": dummy_parameters["model_options"],
        }
    )

    caplog.set_level(logging.DEBUG, logger="kedro")
    successful_run_msg = "Pipeline execution completed successfully."

    SequentialRunner().run(pipeline, catalog)

    assert successful_run_msg in caplog.text
