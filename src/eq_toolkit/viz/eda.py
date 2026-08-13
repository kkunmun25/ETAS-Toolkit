"""
Combined exploratory data analysis (EDA) visualization.

One function call creates a labeled multi-panel figure
from an earthquake Catalog.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from .fmd import plot_fmd
from .maps import plot_epicenters
from .time import (
    plot_time_magnitude,
    plot_cumulative_count,
)
from .interevent import plot_interevent_times
from .space import depth_cross_section


def plot_eda(
    catalog,
    mc=2.5,
    azimuth=0.0,
    figsize=(16, 14),
):
    """
    Create a multi-panel exploratory earthquake figure.

    Parameters
    ----------
    catalog
        Earthquake Catalog object.

    mc : float, default=2.5
        Magnitude of completeness used by the FMD
        and spatial visualizations.

    azimuth : float, default=0
        Azimuth used for the depth cross-section.

    figsize : tuple, default=(16, 14)
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
        Combined EDA figure.

    Notes
    -----
    One function call creates:

        1. Frequency-magnitude distribution
        2. Epicenter map
        3. Time-magnitude distribution
        4. Cumulative event count
        5. Inter-event time distribution
        6. Depth cross-section
    """

    fig = plt.figure(figsize=figsize)

    grid = fig.add_gridspec(
        3,
        2,
        hspace=0.35,
        wspace=0.25,
    )

    # --------------------------------------------------
    # 1. Frequency-magnitude distribution
    # --------------------------------------------------

    ax_fmd = fig.add_subplot(
        grid[0, 0]
    )

    plot_fmd(
        catalog,
        mc=mc,
        ax=ax_fmd,
    )

    ax_fmd.set_title(
        "Frequency-Magnitude Distribution"
    )

    # --------------------------------------------------
    # 2. Epicenter map
    # --------------------------------------------------

    ax_map = fig.add_subplot(
        grid[0, 1]
    )

    plot_epicenters(
        catalog,
        ax=ax_map,
    )

    ax_map.set_title(
        "Earthquake Epicenters"
    )

    # --------------------------------------------------
    # 3. Time-magnitude
    # --------------------------------------------------

    ax_time = fig.add_subplot(
        grid[1, 0]
    )

    plot_time_magnitude(
        catalog,
        ax=ax_time,
    )

    ax_time.set_title(
        "Magnitude-Time Distribution"
    )

    # --------------------------------------------------
    # 4. Cumulative count
    # --------------------------------------------------

    ax_cumulative = fig.add_subplot(
        grid[1, 1]
    )

    plot_cumulative_count(
        catalog,
        ax=ax_cumulative,
    )

    ax_cumulative.set_title(
        "Cumulative Event Count"
    )

    # --------------------------------------------------
    # 5. Inter-event time
    # --------------------------------------------------

    ax_interevent = fig.add_subplot(
        grid[2, 0]
    )

    plot_interevent_times(
        catalog,
        ax=ax_interevent,
    )

    ax_interevent.set_title(
        "Inter-event Time Distribution"
    )

    # --------------------------------------------------
    # 6. Depth cross-section
    # --------------------------------------------------

    ax_depth = fig.add_subplot(
        grid[2, 1]
    )

    depth_cross_section(
        catalog,
        azimuth=azimuth,
        ax=ax_depth,
    )

    ax_depth.set_title(
        f"Depth Cross-section ({azimuth:.0f}°)"
    )

    # Overall figure title
    fig.suptitle(
        "Earthquake Catalog Exploratory Data Analysis",
        fontsize=16,
        y=0.995,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.98]
    )

    return fig


__all__ = [
    "plot_eda",
]