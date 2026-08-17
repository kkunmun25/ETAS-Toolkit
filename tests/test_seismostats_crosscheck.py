import numpy as np
import pytest

from eq_toolkit.quality.mc import (
    maxc,
    mbs,
)


def make_synthetic_catalog(seed=42):
    """
    Create a reproducible synthetic Gutenberg-Richter-like
    magnitude catalog.
    """

    rng = np.random.default_rng(seed)

    mc_true = 2.0
    b_true = 1.0

    magnitudes = (
        mc_true
        + rng.exponential(
            scale=1.0 / (
                b_true * np.log(10)
            ),
            size=2000,
        )
    )

    magnitudes = np.round(
        magnitudes,
        1,
    )

    magnitudes = magnitudes[
        magnitudes <= 5.0
    ]

    return magnitudes


def test_maxc_against_seismostats():

    from seismostats import Catalog

    magnitudes = make_synthetic_catalog()

    # ---------------------------------------------------------------
    # Our implementation
    # ---------------------------------------------------------------

    our_mc = maxc(
        magnitudes,
        bin_width=0.1,
    )

    # ---------------------------------------------------------------
    # SeismoStats
    # ---------------------------------------------------------------

    catalog = Catalog.from_dict(
        {
            "magnitude": magnitudes
        }
    )

    reference_mc, reference_info = (
        catalog.estimate_mc_maxc(
            fmd_bin=0.1
        )
    )

    # ---------------------------------------------------------------
    # Both should produce finite Mc estimates.
    # ---------------------------------------------------------------

    assert np.isfinite(
        our_mc
    )

    assert np.isfinite(
        reference_mc
    )

    # ---------------------------------------------------------------
    # MAXC implementations should agree within
    # one magnitude bin.
    # ---------------------------------------------------------------

    assert abs(
        our_mc - reference_mc
    ) <= 0.1000001


def test_mbs_against_seismostats():

    from seismostats import Catalog

    magnitudes = make_synthetic_catalog()

    # ---------------------------------------------------------------
    # Our MBS implementation
    # ---------------------------------------------------------------

    our_mc = mbs(
        magnitudes,
        bin_width=0.1,
    )

    # ---------------------------------------------------------------
    # SeismoStats MBS
    # ---------------------------------------------------------------

    catalog = Catalog.from_dict(
        {
            "magnitude": magnitudes
        }
    )

    # SeismoStats requires discretized magnitudes
    # for the b-stability method.
    catalog.delta_m = 0.1

    reference_mc, reference_info = (
        catalog.estimate_mc_b_stability()
    )

    # ---------------------------------------------------------------
    # Both should return finite values.
    # ---------------------------------------------------------------

    assert np.isfinite(
        our_mc
    )

    assert np.isfinite(
        reference_mc
    )

    # ---------------------------------------------------------------
    # Allow one magnitude bin of difference.
    # ---------------------------------------------------------------

    assert abs(
        our_mc - reference_mc
    ) <= 0.1000001