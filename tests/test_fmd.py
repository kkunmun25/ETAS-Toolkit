import numpy as np
import pandas as pd

from eq_toolkit.viz.fmd import plot_fmd


def test_plot_fmd_runs():
    magnitudes = np.array([
        2.0, 2.1, 2.1, 2.2, 2.2,
        2.3, 2.3, 2.4, 2.4, 2.5,
        2.6, 2.7, 2.8, 2.9, 3.0,
        3.1, 3.2, 3.3, 3.4, 3.5,
    ])

    catalog = pd.DataFrame({
        "magnitude": magnitudes,
    })

    ax = plot_fmd(
        catalog,
        mc=2.2,
    )

    assert ax is not None
    assert ax.get_xlabel() == "Magnitude"
    assert ax.get_ylabel() == "Number of earthquakes"