"""Test for viz functions."""

import matplotlib.pyplot as plt

from common.backtesting.functions.viz import plot_performance_metrics


def test_plot_performance_metrics(return_df, params_plot_performance_params):
    """Pytest for plotting function."""
    figure = plot_performance_metrics(return_df, params_plot_performance_params)
    assert isinstance(
        figure, plt.Figure
    ), "The return type should be a matplotlib.figure.Figure"

    assert len(figure.axes) > 0, "The figure should have at least one axis"
