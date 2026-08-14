import numpy as np
from eq_toolkit.quality.mc import (maxc,b_value,gft)

# MAXC TEST

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


# B-VALUE TEST

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

    # b-value must be positive
    assert b > 0

    # It should be a finite number
    assert np.isfinite(b)


# GFT TEST

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
        # A small catalog may legitimately fail the
        # 95% and 90% criteria.
        pass


# SYNTHETIC CATALOG

def _make_synthetic_gr_catalog(
    true_mc=2.5,
    b=1.0,
    max_magnitude=5.0,
    bin_width=0.1,
):
    """
    Create a synthetic Gutenberg-Richter catalog
    with a known Mc.

    The cumulative number of earthquakes follows:

        N(M) = N(Mc) * 10^[-b(M-Mc)]
    """

    magnitudes = []

    thresholds = np.arange(
        true_mc,
        max_magnitude + bin_width / 2.0,
        bin_width,
    )

    # Large starting number so that the synthetic
    # catalog has a smooth GR distribution.
    n_mc = 10000

    cumulative = np.floor(
        n_mc
        * 10.0 ** (
            -b * (thresholds - true_mc)
        )
    ).astype(int)

    for i, magnitude in enumerate(thresholds):

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

    return np.asarray(magnitudes)


# GFT SYNTHETIC RECOVERY TEST

def test_gft_recovers_known_mc():

    true_mc = 2.5

    magnitudes = _make_synthetic_gr_catalog(
        true_mc=true_mc,
        b=1.0,
        max_magnitude=5.0,
        bin_width=0.1,
    )

    # Add incomplete / poorly represented
    # smaller earthquakes below the true Mc.
    rng = np.random.default_rng(42)

    incomplete_events = np.round(
        rng.uniform(1.0, 2.4, 300),
        1,
    )

    magnitudes = np.concatenate(
        [incomplete_events, magnitudes]
    )

    estimated_mc = gft(
        magnitudes,
        bin_width=0.1,
        min_fit=0.95,
        fallback_fit=0.90,
        min_events=50,
    )

    # The known Mc should be recovered.
    assert np.isclose(
        estimated_mc,
        true_mc,
        atol=0.1,
    )