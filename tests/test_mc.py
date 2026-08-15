import numpy as np

from eq_toolkit.quality.mc import (
    maxc,
    b_value,
    b_value_sigma,
    gft,
    mbs,
)


# =====================================================================
# MAXC
# =====================================================================

def test_maxc():

    magnitudes = [
        2.0,
        2.1,
        2.1,
        2.2,
        2.2,
        2.3,
        2.3,
        2.3,
        2.3,
        2.4,
        2.4,
        2.5,
        3.0,
        3.2,
    ]

    mc = maxc(magnitudes)

    assert np.isfinite(mc)
    assert mc >= 2.3


# =====================================================================
# B-VALUE
# =====================================================================

def test_b_value():

    magnitudes = np.array([
        2.5,
        2.6,
        2.7,
        2.8,
        3.0,
        3.1,
        3.2,
        3.3,
        3.5,
        3.6,
    ])

    b = b_value(
        magnitudes,
        mc=2.5,
        bin_width=0.1,
    )

    assert b > 0
    assert np.isfinite(b)


# =====================================================================
# B-VALUE UNCERTAINTY
# =====================================================================

def test_b_value_sigma():

    magnitudes = np.array([
        2.5,
        2.6,
        2.7,
        2.8,
        3.0,
        3.1,
        3.2,
        3.3,
        3.5,
        3.6,
        3.7,
        3.8,
    ])

    sigma = b_value_sigma(
        magnitudes,
        mc=2.5,
        bin_width=0.1,
    )

    assert sigma > 0
    assert np.isfinite(sigma)


# =====================================================================
# GFT
# =====================================================================

def test_gft():

    magnitudes = np.array([
        2.0,
        2.1,
        2.1,
        2.2,
        2.2,
        2.3,
        2.3,
        2.4,
        2.5,
        2.5,
        2.6,
        2.7,
        2.8,
        3.0,
        3.1,
        3.2,
    ])

    try:

        mc = gft(
            magnitudes,
            bin_width=0.1,
            min_events=5,
        )

        assert np.isfinite(mc)

    except ValueError:

        # Small catalog may fail both
        # GFT thresholds.
        pass


# =====================================================================
# SYNTHETIC GR CATALOG
# =====================================================================

def _make_synthetic_gr_catalog(
    true_mc=2.5,
    b=1.0,
    max_magnitude=5.0,
    bin_width=0.1,
):
    """
    Create a synthetic Gutenberg-Richter catalog
    with a known completeness magnitude.
    """

    magnitudes = []

    thresholds = np.arange(
        true_mc,
        max_magnitude + bin_width / 2.0,
        bin_width,
    )

    n_mc = 10000

    cumulative = np.floor(
        n_mc
        * 10.0 ** (
            -b * (thresholds - true_mc)
        )
    ).astype(int)

    for i, magnitude in enumerate(
        thresholds
    ):

        if i + 1 < len(cumulative):
            next_count = cumulative[i + 1]
        else:
            next_count = 0

        number_in_bin = (
            cumulative[i] - next_count
        )

        magnitudes.extend(
            [magnitude] * number_in_bin
        )

    return np.asarray(
        magnitudes
    )


# =====================================================================
# GFT KNOWN-MC TEST
# =====================================================================

def test_gft_recovers_known_mc():

    true_mc = 2.5

    magnitudes = _make_synthetic_gr_catalog(
        true_mc=true_mc,
        b=1.0,
        max_magnitude=5.0,
        bin_width=0.1,
    )

    rng = np.random.default_rng(42)

    incomplete_events = np.round(
        rng.uniform(
            1.0,
            2.4,
            300,
        ),
        1,
    )

    magnitudes = np.concatenate(
        [
            incomplete_events,
            magnitudes,
        ]
    )

    estimated_mc = gft(
        magnitudes,
        bin_width=0.1,
        min_fit=0.95,
        fallback_fit=0.90,
        min_events=50,
    )

    assert np.isclose(
        estimated_mc,
        true_mc,
        atol=0.1,
    )


# =====================================================================
# MBS
# =====================================================================

def test_mbs():

    magnitudes = _make_synthetic_gr_catalog(
        true_mc=2.5,
        b=1.0,
        max_magnitude=5.0,
        bin_width=0.1,
    )

    rng = np.random.default_rng(42)

    incomplete_events = np.round(
        rng.uniform(
            1.0,
            2.4,
            300,
        ),
        1,
    )

    magnitudes = np.concatenate(
        [
            incomplete_events,
            magnitudes,
        ]
    )

    estimated_mc = mbs(
        magnitudes,
        bin_width=0.1,
        window=0.5,
        min_events=50,
    )

    assert np.isfinite(
        estimated_mc
    )

    assert np.isclose(
        estimated_mc,
        2.5,
        atol=0.1,
    )