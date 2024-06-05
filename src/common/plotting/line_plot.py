"""Plotting functions for closing price prediction."""

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from common.utilities.extract_target_variable_name import extract_target_variable_name


def line_plot(
    predictions: pd.DataFrame,
    modeling_params: dict[str, Any],
) -> plt.Figure:
    """Plot a line plot of the actual and predicted values.

    Args:
    ----
        predictions (pd.DataFrame): DataFrame containing the actual and predicted
            values.
        modeling_params (dict[str, Any]): Parameters for the modeling.

    Returns:
    -------
        plt.Figure: The line plot.

    """
    # Extract necessary parameters from modeling_params
    target_column_name = extract_target_variable_name(predictions.columns)
    plot_params = modeling_params.get("plotting_parameters", {})

    # Define default plot parameters if not provided
    title = plot_params.get("title", f"Stock Symbol: {target_column_name}")
    xlabel = plot_params.get("xlabel", "Date")
    ylabel = plot_params.get("ylabel", "Stock Price")
    true_color = plot_params.get("true_color", "b")
    pred_color = plot_params.get("pred_color", "r")
    linestyle = plot_params.get("linestyle", "-")
    alpha = plot_params.get("alpha", 0.8)

    # Plotting
    plt.figure(figsize=(10, 6))

    plt.plot(
        predictions.index.to_timestamp(),
        predictions[target_column_name],
        linestyle=linestyle,
        color=true_color,
        alpha=alpha,
        label="Actual",
    )
    plt.plot(
        predictions.index.to_timestamp(),
        predictions["prediction_label"],
        linestyle=linestyle,
        color=pred_color,
        alpha=alpha,
        label="Prediction",
    )
    # Customizing the plot
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    return plt
