"""Functions for visualization of backtesting results."""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def plot_performance_metrics(
    portfolio_returns: pd.DataFrame,
    plot_performance_params: dict[str, str],
) -> plt.Figure:
    """Plot portfolio returns over time.

    Args:
    ----
        portfolio_returns (pd.DataFrame): DataFrame containing the returns of the
            portfolio that should be plotted.
        plot_performance_params (dict[str, str]): List of columns to plot.

    Raises:
    ------
        ValueError: If the column names in 'plotting_columns' are not in the
            'portfolio_returns' DataFrame.

    Returns:
    -------
        plt.Figure: Figure object containing the plot.

    """
    plotting_columns = plot_performance_params["columns"]
    if not plotting_columns or not all(
        col in portfolio_returns.columns for col in plotting_columns
    ):
        raise ValueError("Invalid column names in 'plotting_columns'")
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {"Adjusted Portfolio Returns": "blue", "Portfolio Returns": "green"}
    for column in plotting_columns:
        ax.plot(
            portfolio_returns.index,
            portfolio_returns[column],
            label=column,
            color=colors.get(column, "black"),
        )

    # Formatting the x-axis
    ax.xaxis.set_major_locator(
        mdates.WeekdayLocator(interval=1)
    )  # Set interval for weeks
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%Y-%m-%d")
    )  # Date format on x-axis

    plt.xticks(rotation=45)

    # Set plot labels and title
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("Returns", fontsize=14)
    ax.set_title("Portfolio Returns Over Time", fontsize=16)

    ax.legend(fontsize=12)
    ax.grid(True)
    plt.tight_layout()

    return fig
