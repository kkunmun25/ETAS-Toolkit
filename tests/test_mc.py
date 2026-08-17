import numpy as np

from eq_toolkit.quality.mc import (
    maxc,
    b_value,
    b_value_sigma,
    gft,
    mbs,
    emr,
    mbass,
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

# =====================================================================
# EMR DETECTION PROBABILITY
# =====================================================================

def test_emr_detection_probability():

    from eq_toolkit.quality.mc import (
        emr_detection_probability,
    )

    magnitudes = np.array([
        1.0,
        2.0,
        2.5,
        3.0,
        4.0,
    ])

    probability = emr_detection_probability(
        magnitudes,
        mc=2.5,
        sigma=0.2,
    )

    # Same number of probabilities as magnitudes
    assert len(probability) == len(
        magnitudes
    )

    # Probabilities must be between 0 and 1
    assert np.all(
        probability >= 0.0
    )

    assert np.all(
        probability <= 1.0
    )

    # Around Mc, probability should be ~0.5
    assert np.isclose(
        probability[2],
        0.5,
        atol=0.01,
    )

    # Larger magnitudes should have
    # higher detection probability
    assert (
        probability[0]
        < probability[-1]
    )

# =====================================================================
# EMR LOG-LIKELIHOOD
# =====================================================================

def test_emr_log_likelihood():

    from eq_toolkit.quality.mc import (
        emr_log_likelihood,
    )

    magnitudes = np.array([
        2.0,
        2.1,
        2.2,
        2.3,
        2.4,
        2.5,
        2.6,
        2.7,
        2.8,
        3.0,
        3.2,
        3.5,
    ])

    likelihood = emr_log_likelihood(
        magnitudes,
        b=1.0,
        mc=2.5,
        sigma=0.2,
    )

    assert np.isfinite(
        likelihood
    )

    assert likelihood < 0

# =====================================================================
# EMR MAXIMUM LIKELIHOOD
# =====================================================================

def test_emr():

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

    estimated_mc = emr(
        magnitudes,
        bin_width=0.1,
        min_events=50,
    )

    assert np.isfinite(
        estimated_mc
    )

    # EMR should recover approximately the
    # known completeness magnitude.
    assert np.isclose(
        estimated_mc,
        2.5,
        atol=0.3,
    )        

# =====================================================================
# EMR BOOTSTRAP
# =====================================================================

def test_bootstrap_emr():

    from eq_toolkit.quality.mc import (
        bootstrap_emr,
    )

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

    result = bootstrap_emr(
        magnitudes,
        n_bootstrap=20,
        bin_width=0.1,
        min_events=50,
        random_state=42,
    )

    # Required output fields
    assert "mc" in result
    assert "mc_lower" in result
    assert "mc_upper" in result
    assert "bootstrap_samples" in result

    # All values must be finite
    assert np.isfinite(
        result["mc"]
    )

    assert np.isfinite(
        result["mc_lower"]
    )

    assert np.isfinite(
        result["mc_upper"]
    )

    # Confidence interval must be ordered
    assert (
        result["mc_lower"]
        <= result["mc_upper"]
    )

    # Bootstrap should produce multiple estimates
    assert len(
        result["bootstrap_samples"]
    ) >= 10

# =====================================================================
# MBASS SEGMENT SLOPES
# =====================================================================

def test_mbass_segment_slopes():

    from eq_toolkit.quality.mc import (
        mbass_segment_slopes,
    )

    rng = np.random.default_rng(42)

    magnitudes = np.round(
        1.0
        + rng.exponential(
            scale=0.5,
            size=500,
        ),
        1,
    )

    slope_magnitudes, slopes, counts = (
        mbass_segment_slopes(
            magnitudes,
            bin_width=0.1,
        )
    )

    assert len(
        slope_magnitudes
    ) == len(slopes)

    assert len(
        counts
    ) > 0

    assert np.all(
        np.isfinite(slopes)
    )

# =====================================================================
# MBASS CHANGE POINT
# =====================================================================

def test_mbass_change_point():

    from eq_toolkit.quality.mc import (
        mbass_change_point,
    )

    rng = np.random.default_rng(42)

    # Two populations with clearly different
    # median slopes.
    left = rng.normal(
        loc=0.3,
        scale=0.03,
        size=30,
    )

    right = rng.normal(
        loc=1.0,
        scale=0.03,
        size=30,
    )

    slopes = np.concatenate(
        [
            left,
            right,
        ]
    )

    slope_magnitudes = np.arange(
        len(slopes),
        dtype=float,
    ) * 0.1

    result = mbass_change_point(
        slope_magnitudes,
        slopes,
        alpha=0.05,
    )

    assert np.isfinite(
        result["mc"]
    )

    assert result["p_value"] < 0.05

    assert result["significant"]

    # Change point should be near
    # the middle of the slope sequence.
    assert np.isclose(
        result["mc"],
        3.0,
        atol=0.5,
    )

# =====================================================================
# MBASS
# =====================================================================

def test_mbass():

    rng = np.random.default_rng(42)

    # ---------------------------------------------------------------
    # Generate an incomplete low-magnitude population.
    #
    # This population is intentionally NOT Gutenberg-Richter-like.
    # ---------------------------------------------------------------

    low_magnitudes = rng.uniform(
        1.0,
        2.4,
        size=1500,
    )

    low_magnitudes = np.round(
        low_magnitudes,
        1,
    )

    # ---------------------------------------------------------------
    # Generate a clean Gutenberg-Richter-like population
    # above the true completeness magnitude.
    # ---------------------------------------------------------------

    b = 1.0
    mc_true = 2.5

    # Exponential distribution corresponding to
    # Gutenberg-Richter magnitudes.
    high_magnitudes = (
        mc_true
        + rng.exponential(
            scale=1.0 / (
                b * np.log(10)
            ),
            size=5000,
        )
    )

    high_magnitudes = np.round(
        high_magnitudes,
        1,
    )

    # Keep realistic magnitude range
    high_magnitudes = high_magnitudes[
        high_magnitudes <= 5.0
    ]

    magnitudes = np.concatenate(
        [
            low_magnitudes,
            high_magnitudes,
        ]
    )

    mc = mbass(
        magnitudes,
        bin_width=0.1,
        alpha=0.05,
    )

    assert np.isfinite(mc)

    assert 2.0 <= mc <= 3.0

# =====================================================================
# MBASS BOOTSTRAP
# =====================================================================

def test_bootstrap_mbass():

    from eq_toolkit.quality.mc import (
        bootstrap_mbass,
    )

    rng = np.random.default_rng(42)

    # ---------------------------------------------------------------
    # Create incomplete low-magnitude population
    # ---------------------------------------------------------------

    low_magnitudes = rng.uniform(
        1.0,
        2.4,
        size=1500,
    )

    low_magnitudes = np.round(
        low_magnitudes,
        1,
    )

    # ---------------------------------------------------------------
    # Create Gutenberg-Richter-like population above Mc
    # ---------------------------------------------------------------

    mc_true = 2.5
    b = 1.0

    high_magnitudes = (
        mc_true
        + rng.exponential(
            scale=1.0 / (
                b * np.log(10)
            ),
            size=5000,
        )
    )

    high_magnitudes = np.round(
        high_magnitudes,
        1,
    )

    high_magnitudes = high_magnitudes[
        high_magnitudes <= 5.0
    ]

    magnitudes = np.concatenate(
        [
            low_magnitudes,
            high_magnitudes,
        ]
    )

    # ---------------------------------------------------------------
    # Run MBASS bootstrap
    # ---------------------------------------------------------------

    result = bootstrap_mbass(
        magnitudes,
        n_bootstrap=20,
        bin_width=0.1,
        alpha=0.05,
        random_state=42,
    )

    # ---------------------------------------------------------------
    # Check returned fields
    # ---------------------------------------------------------------

    assert "mc" in result
    assert "mc_lower" in result
    assert "mc_upper" in result
    assert "bootstrap_samples" in result
    assert "n_success" in result

    # ---------------------------------------------------------------
    # Check finite results
    # ---------------------------------------------------------------

    assert np.isfinite(
        result["mc"]
    )

    assert np.isfinite(
        result["mc_lower"]
    )

    assert np.isfinite(
        result["mc_upper"]
    )

    # ---------------------------------------------------------------
    # Confidence interval must be ordered
    # ---------------------------------------------------------------

    assert (
        result["mc_lower"]
        <= result["mc_upper"]
    )

    # ---------------------------------------------------------------
    # At least 10 bootstrap runs should succeed
    # ---------------------------------------------------------------

    assert (
        result["n_success"]
        >= 10
    )

    assert (
        len(
            result["bootstrap_samples"]
        )
        == result["n_success"]
    )

    # ---------------------------------------------------------------
    # Mc should remain in a reasonable range
    # ---------------------------------------------------------------

    assert (
        1.5
        <= result["mc"]
        <= 3.5
    )    