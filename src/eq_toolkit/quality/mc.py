import numpy as np

from scipy.special import erf
from scipy.stats import kstest
from scipy.optimize import minimize
from scipy.stats import mannwhitneyu


# MAXC

def maxc(magnitudes, bin_width=0.1):
    """
    Estimate Mc using Maximum Curvature.

    An empirical +0.2 correction is applied following
    Wiemer & Wyss (2000).
    """

    magnitudes = np.asarray(magnitudes, dtype=float)

    magnitudes = magnitudes[np.isfinite(magnitudes)]

    if len(magnitudes) == 0:
        raise ValueError("No valid magnitudes supplied.")

    if bin_width <= 0:
        raise ValueError("bin_width must be positive.")

    min_mag = (
        np.floor(magnitudes.min() / bin_width)
        * bin_width
    )

    max_mag = (
        np.ceil(magnitudes.max() / bin_width)
        * bin_width
    )

    bins = np.arange(
        min_mag,
        max_mag + bin_width,
        bin_width,
    )

    counts, edges = np.histogram(
        magnitudes,
        bins=bins,
    )

    if len(counts) == 0:
        raise ValueError(
            "Unable to construct magnitude bins."
        )

    max_index = np.argmax(counts)

    mc_max_curvature = edges[max_index]

    mc = mc_max_curvature + 0.2

    return float(mc)


# B-VALUE


def b_value(magnitudes, mc, bin_width=0.1):
    """
    Calculate the Gutenberg-Richter maximum-likelihood b-value.

    b = log10(e) /
        [mean(M) - (Mc - delta_M / 2)]
    """

    magnitudes = np.asarray(
        magnitudes,
        dtype=float,
    )

    magnitudes = magnitudes[
        np.isfinite(magnitudes)
    ]

    if len(magnitudes) == 0:
        raise ValueError(
            "No valid magnitudes supplied."
        )

    if bin_width <= 0:
        raise ValueError(
            "bin_width must be positive."
        )

    complete = magnitudes[
        magnitudes >= mc
    ]

    if len(complete) < 2:
        raise ValueError(
            "Not enough earthquakes above Mc "
            "to calculate b-value."
        )

    mean_magnitude = np.mean(complete)

    denominator = (
        mean_magnitude
        - (mc - bin_width / 2.0)
    )

    if denominator <= 0:
        raise ValueError(
            "Invalid denominator while calculating "
            "b-value."
        )

    b = np.log10(np.e) / denominator

    return float(b)


# B-VALUE UNCERTAINTY


def b_value_sigma(
    magnitudes,
    mc,
    bin_width=0.1,
):
    """
    Calculate the standard uncertainty of the b-value.

    Shi & Bolt (1982):

        sigma_b = 2.3 * b^2 * sqrt(sum((Mi - Mmean)^2)/[N * (N - 1)])

    """

    magnitudes = np.asarray(
        magnitudes,
        dtype=float,
    )

    magnitudes = magnitudes[
        np.isfinite(magnitudes)
    ]

    complete = magnitudes[
        magnitudes >= mc
    ]

    n = len(complete)

    if n < 2:
        raise ValueError(
            "At least two earthquakes are required "
            "to calculate b-value uncertainty."
        )

    b = b_value(
        complete,
        mc=mc,
        bin_width=bin_width,
    )

    mean_magnitude = np.mean(complete)

    squared_deviations = np.sum(
        (complete - mean_magnitude) ** 2
    )

    sigma_b = (2.3* b**2* np.sqrt( squared_deviations/(n * (n - 1))))

    return float(sigma_b)



# GFT INTERNAL FUNCTION


def _gft_score(
    magnitudes,
    mc,
    bin_width=0.1,
):
    """
    Calculate Gutenberg-Richter goodness-of-fit percentage.
    """

    magnitudes = np.asarray(
        magnitudes,
        dtype=float,
    )

    complete = magnitudes[
        magnitudes >= mc
    ]

    if len(complete) < 2:
        return np.nan, np.nan

    b = b_value(
        complete,
        mc=mc,
        bin_width=bin_width,
    )

    max_mag = np.max(complete)

    thresholds = np.arange(
        mc,
        max_mag + bin_width / 2.0,
        bin_width,
    )

    thresholds = np.round(
        thresholds,
        10,
    )

    if len(thresholds) == 0:
        return np.nan, np.nan

    observed = np.array([
        np.sum(complete >= m)
        for m in thresholds
    ], dtype=float)

    if observed[0] <= 0:
        return np.nan, np.nan

    predicted = (
        observed[0]
        * 10.0 ** (
            -b * (thresholds - mc)
        )
    )

    misfit = np.sum(
        np.abs(
            observed - predicted
        )
    )

    total_observed = np.sum(
        observed
    )

    if total_observed <= 0:
        return np.nan, np.nan

    goodness = (
        100.0
        * (
            1.0
            - misfit / total_observed
        )
    )

    goodness = max(
        0.0,
        min(100.0, goodness),
    )

    return (
        float(goodness),
        float(b),
    )


# GFT


def gft(
    magnitudes,
    bin_width=0.1,
    min_fit=0.95,
    fallback_fit=0.90,
    min_events=50,
):
    """
    Estimate Mc using the Goodness-of-Fit Test.

    95% goodness-of-fit is attempted first.
    90% is used as fallback.
    """

    magnitudes = np.asarray(
        magnitudes,
        dtype=float,
    )

    magnitudes = magnitudes[
        np.isfinite(magnitudes)
    ]

    if len(magnitudes) == 0:
        raise ValueError(
            "No valid magnitudes supplied."
        )

    if bin_width <= 0:
        raise ValueError(
            "bin_width must be positive."
        )

    if not (
        0 < fallback_fit <= min_fit <= 1
    ):
        raise ValueError(
            "Thresholds must satisfy "
            "0 < fallback_fit <= min_fit <= 1."
        )

    if min_events < 2:
        raise ValueError(
            "min_events must be at least 2."
        )

    min_mag = (
        np.floor(
            magnitudes.min()
            / bin_width
        )
        * bin_width
    )

    max_mag = (
        np.floor(
            magnitudes.max()
            / bin_width
        )
        * bin_width
    )

    candidates = np.arange(
        min_mag,
        max_mag + bin_width / 2.0,
        bin_width,
    )

    candidates = np.round(
        candidates,
        10,
    )

    # ---------------------------------------------------------------
    # 95% criterion
    # ---------------------------------------------------------------

    for candidate in candidates:

        n_events = np.sum(
            magnitudes >= candidate
        )

        if n_events < min_events:
            continue

        goodness, _ = _gft_score(
            magnitudes,
            mc=candidate,
            bin_width=bin_width,
        )

        if np.isfinite(goodness):

            if goodness >= min_fit * 100.0:
                return float(candidate)

  
    # 90% fallback
   

    for candidate in candidates:

        n_events = np.sum(
            magnitudes >= candidate
        )

        if n_events < min_events:
            continue

        goodness, _ = _gft_score(
            magnitudes,
            mc=candidate,
            bin_width=bin_width,
        )

        if np.isfinite(goodness):

            if goodness >= fallback_fit * 100.0:
                return float(candidate)

    raise ValueError(
        "No magnitude of completeness satisfied "
        "the 95% or 90% goodness-of-fit criteria."
    )


# MBS


def mbs(
    magnitudes,
    bin_width=0.1,
    window=0.5,
    min_events=50,
):
    """
    Estimate Mc using the Magnitude of Completeness
    by b-value Stability (MBS).

    This follows the Woessner & Wiemer (2005)
    refinement of the Cao & Gao (2002) method.



    """

    magnitudes = np.asarray(
        magnitudes,
        dtype=float,
    )

    magnitudes = magnitudes[
        np.isfinite(magnitudes)
    ]

    if len(magnitudes) == 0:
        raise ValueError(
            "No valid magnitudes supplied."
        )

    if bin_width <= 0:
        raise ValueError(
            "bin_width must be positive."
        )

    if window <= 0:
        raise ValueError(
            "window must be positive."
        )

    if min_events < 2:
        raise ValueError(
            "min_events must be at least 2."
        )

   
    # Candidate cutoff magnitudes
   

    min_mag = (
        np.floor(
            magnitudes.min()
            / bin_width
        )
        * bin_width
    )

    max_mag = (
        np.floor(
            magnitudes.max()
            / bin_width
        )
        * bin_width
    )

    candidates = np.arange(
        min_mag,
        max_mag + bin_width / 2.0,
        bin_width,
    )

    candidates = np.round(
        candidates,
        10,
    )

    # Calculate b-values for all usable candidates


    b_values = {}
    b_sigmas = {}

    for candidate in candidates:

        n_events = np.sum(
            magnitudes >= candidate
        )

        if n_events < min_events:
            continue

        try:

            b = b_value(
                magnitudes,
                mc=candidate,
                bin_width=bin_width,
            )

            sigma = b_value_sigma(
                magnitudes,
                mc=candidate,
                bin_width=bin_width,
            )

        except ValueError:
            continue

        if (
            np.isfinite(b)
            and np.isfinite(sigma)
        ):

            b_values[
                float(candidate)
            ] = b

            b_sigmas[
                float(candidate)
            ] = sigma

    if len(b_values) < 2:
        raise ValueError(
            "Not enough usable magnitude bins "
            "for MBS estimation."
        )

    candidate_values = sorted(
        b_values.keys()
    )


    # Search for the stable b-value region


    for candidate in candidate_values:

        upper_limit = (
            candidate + window
        )

        window_b_values = [
            b_values[m]
            for m in candidate_values
            if (
                m >= candidate
                and m <= upper_limit + 1e-10
            )
        ]

        
        if len(window_b_values) < 2:
            continue

        b_average = np.mean(
            window_b_values
        )

        difference = abs(
            b_values[candidate]
            - b_average
        )

        sigma_b = b_sigmas[
            candidate
        ]

    
        if difference <= sigma_b:

            return float(candidate)

    raise ValueError(
        "No stable b-value region found "
        "for the supplied catalog."
    )


# EMR DETECTION PROBABILITY

def emr_detection_probability(
    magnitudes,
    mc,
    sigma,
):
    """
    Calculate earthquake detection probability for EMR.

    The detection probability follows a smooth error-function
    transition from low detection probability at small magnitudes
    to high detection probability at large magnitudes.

    """

    magnitudes = np.asarray(
        magnitudes,
        dtype=float,
    )

    if sigma <= 0:
        raise ValueError(
            "sigma must be positive."
        )

    probability = 0.5 * (
        1.0
        + erf(
            (magnitudes - mc)
            /
            (np.sqrt(2.0) * sigma)
        )
    )

    return probability


# EMR LOG-LIKELIHOOD


    """
    Calculate the log-likelihood of an EMR model.

    The observed magnitude distribution is modeled as:

        f(M) ∝ 10^(-b M) * P_detection(M)

    The model is normalized numerically over the observed
    magnitude range.

    """

def emr_log_likelihood(
    magnitudes,
    b,
    mc,
    sigma,
    bin_width=0.1,
):
    
    magnitudes = np.asarray(
        magnitudes,
        dtype=float,
    )

    magnitudes = magnitudes[
        np.isfinite(magnitudes)
    ]

    if len(magnitudes) == 0:
        raise ValueError(
            "No valid magnitudes supplied."
        )

    if b <= 0:
        return -np.inf

    if sigma <= 0:
        return -np.inf

    if bin_width <= 0:
        raise ValueError(
            "bin_width must be positive."
        )

    min_mag = np.min(magnitudes)
    max_mag = np.max(magnitudes)

    # Numerical integration grid
    grid = np.arange(
        min_mag,
        max_mag + bin_width,
        bin_width,
    )

    # Detection probability on grid
    detection = emr_detection_probability(
        grid,
        mc=mc,
        sigma=sigma,
    )

    # Gutenberg-Richter component
    gr = 10.0 ** (
        -b * grid
    )

    # Combined model
    density = gr * detection

    # Numerical normalization
    normalization = np.trapezoid(
        density,
        grid,
    )

    if (
        not np.isfinite(normalization)
        or normalization <= 0
    ):
        return -np.inf

    # Detection probability at observations
    obs_detection = emr_detection_probability(
        magnitudes,
        mc=mc,
        sigma=sigma,
    )

    if np.any(obs_detection <= 0):
        return -np.inf

    # Unnormalized GR density
    obs_gr = 10.0 ** (
        -b * magnitudes
    )

    probability = (
        obs_gr
        * obs_detection
        / normalization
    )

    if np.any(probability <= 0):
        return -np.inf

    log_likelihood = np.sum(
        np.log(probability)
    )

    return float(log_likelihood)

# EMR MAXIMUM-LIKELIHOOD ESTIMATION


def emr(
    magnitudes,
    bin_width=0.1,
    min_events=50,
):
    """
    Estimate Mc using the Entire Magnitude Range (EMR)
    maximum-likelihood method.

    """

    magnitudes = np.asarray(
        magnitudes,
        dtype=float,
    )

    magnitudes = magnitudes[
        np.isfinite(magnitudes)
    ]

    if len(magnitudes) < min_events:
        raise ValueError(
            "Not enough earthquakes for EMR estimation."
        )

    if bin_width <= 0:
        raise ValueError(
            "bin_width must be positive."
        )

    # Initial parameter estimates
    

    initial_mc = np.median(
        magnitudes
    )

    initial_b = 1.0

    initial_sigma = 0.2

    x0 = np.array([
        initial_b,
        initial_mc,
        initial_sigma,
    ])

    # Parameter bounds

    min_mag = np.min(
        magnitudes
    )

    max_mag = np.max(
        magnitudes
    )

    bounds = [
        # b
        (0.1, 3.0),

        # Mc
        (
            min_mag,
            max_mag,
        ),

        # sigma
        (
            0.02,
            max(
                0.5,
                max_mag - min_mag,
            ),
        ),
    ]

    # Make sure initial Mc is inside bounds
    initial_mc = np.clip(
        initial_mc,
        min_mag,
        max_mag,
    )

    x0 = np.array([
        initial_b,
        initial_mc,
        initial_sigma,
    ])

    # Objective function
 

    def objective(parameters):

        b, mc, sigma = parameters

        likelihood = emr_log_likelihood(
            magnitudes,
            b=b,
            mc=mc,
            sigma=sigma,
            bin_width=bin_width,
        )

        if not np.isfinite(
            likelihood
        ):
            return 1e100

        # Minimize negative log-likelihood
        return -likelihood

    # Optimization
 

    result = minimize(
        objective,
        x0=x0,
        bounds=bounds,
        method="L-BFGS-B",
    )

    if not result.success:
        raise ValueError(
            "EMR maximum-likelihood optimization failed: "
            + result.message
        )

    b_estimate = result.x[0]
    mc_estimate = result.x[1]
    sigma_estimate = result.x[2]

    if not np.isfinite(
        mc_estimate
    ):
        raise ValueError(
            "EMR produced an invalid Mc."
        )

    return float(mc_estimate)

# EMR BOOTSTRAP UNCERTAINTY


def bootstrap_emr(
    magnitudes,
    n_bootstrap=100,
    bin_width=0.1,
    min_events=50,
    random_state=None,
):
    """
    Estimate EMR Mc and its bootstrap confidence interval.

 
    """

    magnitudes = np.asarray(
        magnitudes,
        dtype=float,
    )

    magnitudes = magnitudes[
        np.isfinite(magnitudes)
    ]

    if len(magnitudes) < min_events:
        raise ValueError(
            "Not enough earthquakes for EMR bootstrap."
        )

    if n_bootstrap < 10:
        raise ValueError(
            "n_bootstrap must be at least 10."
        )

    
    # Original EMR estimate


    original_mc = emr(
        magnitudes,
        bin_width=bin_width,
        min_events=min_events,
    )

    
    # Random number generator
    

    rng = np.random.default_rng(
        random_state
    )

    bootstrap_samples = []
    # Bootstrap loop
  

    for _ in range(n_bootstrap):

        # Sample earthquakes WITH replacement
        sample = rng.choice(
            magnitudes,
            size=len(magnitudes),
            replace=True,
        )

        try:

            mc_bootstrap = emr(
                sample,
                bin_width=bin_width,
                min_events=min_events,
            )

            if np.isfinite(
                mc_bootstrap
            ):
                bootstrap_samples.append(
                    mc_bootstrap
                )

        except ValueError:
            continue

    bootstrap_samples = np.asarray(
        bootstrap_samples,
        dtype=float,
    )

    if len(bootstrap_samples) < 10:
        raise ValueError(
            "Too few successful bootstrap EMR estimates."
        )

    # 95% percentile confidence interval
   

    lower = np.percentile(
        bootstrap_samples,
        2.5,
    )

    upper = np.percentile(
        bootstrap_samples,
        97.5,
    )

    return {
        "mc": float(original_mc),
        "mc_lower": float(lower),
        "mc_upper": float(upper),
        "bootstrap_samples": bootstrap_samples,
    }

# MBASS - SEGMENT SLOPE


def mbass_segment_slopes(
    magnitudes,
    bin_width=0.1,
):
    """
    Calculate the segment slopes of the non-cumulative
    frequency-magnitude distribution (FMD).

    For consecutive magnitude bins:

        s(M) =
            [log10(N_i) - log10(N_{i+1})]
            / bin_width

    where N_i is the number of earthquakes in a magnitude bin.


    """

    magnitudes = np.asarray(
        magnitudes,
        dtype=float,
    )

    magnitudes = magnitudes[
        np.isfinite(magnitudes)
    ]

    if len(magnitudes) < 3:
        raise ValueError(
            "At least three magnitudes are required."
        )

    if bin_width <= 0:
        raise ValueError(
            "bin_width must be positive."
        )

    # Construct magnitude bins


    min_mag = (
        np.floor(
            magnitudes.min()
            / bin_width
        )
        * bin_width
    )

    max_mag = (
        np.ceil(
            magnitudes.max()
            / bin_width
        )
        * bin_width
    )

    edges = np.arange(
        min_mag,
        max_mag + bin_width,
        bin_width,
    )

    counts, edges = np.histogram(
        magnitudes,
        bins=edges,
    )

    centers = (
        edges[:-1]
        + bin_width / 2.0
    )

    valid = (
        counts[:-1] > 0
    ) & (
        counts[1:] > 0
    )

    if np.sum(valid) < 3:
        raise ValueError(
            "Not enough populated adjacent magnitude bins "
            "to calculate MBASS slopes."
        )

    left_counts = counts[:-1][valid]
    right_counts = counts[1:][valid]

    left_centers = centers[:-1][valid]
    right_centers = centers[1:][valid]

    slopes = (
        np.log10(left_counts)
        - np.log10(right_counts)
    ) / (
        right_centers
        - left_centers
    )

    slope_centers = (
        left_centers
        + right_centers
    ) / 2.0

    return (
        slope_centers.astype(float),
        slopes.astype(float),
        counts.astype(int),
    )

# MBASS - CHANGE POINT


def mbass_change_point(
    slope_magnitudes,
    slopes,
    alpha=0.05,
):
    """
    Detect the main change point in the MBASS slope series.

    The slope sequence is divided at every possible candidate
    point. The two sides are compared using the
    Wilcoxon-Mann-Whitney / Mann-Whitney U test.

    The main discontinuity is the candidate with the smallest
    p-value.

    """

    slope_magnitudes = np.asarray(
        slope_magnitudes,
        dtype=float,
    )

    slopes = np.asarray(
        slopes,
        dtype=float,
    )

    valid = (
        np.isfinite(
            slope_magnitudes
        )
        &
        np.isfinite(slopes)
    )

    slope_magnitudes = (
        slope_magnitudes[valid]
    )

    slopes = slopes[valid]

    if len(slopes) < 5:
        raise ValueError(
            "At least five segment slopes are required "
            "for change-point detection."
        )

    if not 0 < alpha < 1:
        raise ValueError(
            "alpha must be between 0 and 1."
        )

    candidate_magnitudes = []
    p_values = []
    statistics = []

 
    # Try every interior split


    for i in range(
        2,
        len(slopes) - 2,
    ):

        left = slopes[:i]
        right = slopes[i:]

        if len(left) < 2:
            continue

        if len(right) < 2:
            continue

        result = mannwhitneyu(
            left,
            right,
            alternative="two-sided",
            method="auto",
        )

        candidate_magnitudes.append(
            slope_magnitudes[i]
        )

        statistics.append(
            float(result.statistic)
        )

        p_values.append(
            float(result.pvalue)
        )

    if len(p_values) == 0:
        raise ValueError(
            "Unable to identify MBASS change-point candidates."
        )

    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    candidate_magnitudes = np.asarray(
        candidate_magnitudes,
        dtype=float,
    )

    statistics = np.asarray(
        statistics,
        dtype=float,
    )

    # Main discontinuity = smallest p-value
 

    best_index = np.argmin(
        p_values
    )

    mc = candidate_magnitudes[
        best_index
    ]

    best_p = p_values[
        best_index
    ]

    best_statistic = statistics[
        best_index
    ]

    return {
        "mc": float(mc),
        "p_value": float(best_p),
        "statistic": float(
            best_statistic
        ),
        "significant": bool(
            best_p < alpha
        ),
        "candidate_magnitudes":
            candidate_magnitudes,
        "p_values":
            p_values,
    }

# MBASS


def mbass(
    magnitudes,
    bin_width=0.1,
    alpha=0.05,
):
    """
    Estimate magnitude of completeness using MBASS.

    MBASS (Median-Based Analysis of the Segment Slope)
    detects significant changes in the slope of the
    non-cumulative frequency-magnitude distribution.

    The main discontinuity is selected as the magnitude
    corresponding to the smallest probability from the
    Wilcoxon-Mann-Whitney change-point test.

   
    """

    slope_magnitudes, slopes, _ = (
        mbass_segment_slopes(
            magnitudes,
            bin_width=bin_width,
        )
    )

    result = mbass_change_point(
        slope_magnitudes,
        slopes,
        alpha=alpha,
    )

    if not result["significant"]:
        raise ValueError(
            "MBASS did not detect a statistically "
            "significant slope change."
        )

    return float(
        result["mc"]
    )

# MBASS BOOTSTRAP


def bootstrap_mbass(
    magnitudes,
    n_bootstrap=100,
    bin_width=0.1,
    alpha=0.05,
    random_state=None,
):
    """
    Estimate MBASS Mc and bootstrap confidence intervals.

    The catalog is resampled with replacement and MBASS is
    recalculated for every bootstrap sample.

    """
    # Validate input
    

    magnitudes = np.asarray(
        magnitudes,
        dtype=float,
    )

    magnitudes = magnitudes[
        np.isfinite(magnitudes)
    ]

    if len(magnitudes) < 10:
        raise ValueError(
            "Not enough valid magnitudes for MBASS bootstrap."
        )

    if n_bootstrap < 10:
        raise ValueError(
            "n_bootstrap must be at least 10."
        )

    if bin_width <= 0:
        raise ValueError(
            "bin_width must be positive."
        )

    if not 0 < alpha < 1:
        raise ValueError(
            "alpha must be between 0 and 1."
        )


    # Calculate MBASS for the original catalog
    

    original_mc = mbass(
        magnitudes,
        bin_width=bin_width,
        alpha=alpha,
    )

    # Random number generator
    
    rng = np.random.default_rng(
        random_state
    )

    bootstrap_samples = []

    # Bootstrap resampling


    for _ in range(n_bootstrap):

        # Sample earthquakes WITH replacement
        sample = rng.choice(
            magnitudes,
            size=len(magnitudes),
            replace=True,
        )

        try:

            mc_bootstrap = mbass(
                sample,
                bin_width=bin_width,
                alpha=alpha,
            )

            if np.isfinite(
                mc_bootstrap
            ):
                bootstrap_samples.append(
                    mc_bootstrap
                )

        except ValueError:
            continue

    bootstrap_samples = np.asarray(
        bootstrap_samples,
        dtype=float,
    )

    if len(bootstrap_samples) < 10:
        raise ValueError(
            "Too few successful MBASS bootstrap estimates."
        )

    # 95% percentile confidence interval
    

    lower = np.percentile(
        bootstrap_samples,
        2.5,
    )

    upper = np.percentile(
        bootstrap_samples,
        97.5,
    )

    return {
        "mc": float(original_mc),
        "mc_lower": float(lower),
        "mc_upper": float(upper),
        "bootstrap_samples": bootstrap_samples,
        "n_success": int(
            len(bootstrap_samples)
        ),
    }

def lilliefors_exponentiality(
    magnitudes,
    mc,
):
    """
    Lilliefors test for exponentiality of magnitudes
    above the selected magnitude of completeness.

    The Gutenberg-Richter relation implies that
    magnitude excesses (M - Mc) should approximately
    follow an exponential distribution.

    Parameters
    ----------
    magnitudes : array-like
        Earthquake magnitudes.

    mc : float
        Selected magnitude of completeness.

    Returns
    -------
    statistic : float
        Lilliefors KS statistic.

    p_value : float
        Lilliefors p-value.
    """

    import numpy as np
    from statsmodels.stats.diagnostic import lilliefors

    magnitudes = np.asarray(
        magnitudes,
        dtype=float,
    )

    magnitudes = magnitudes[
        np.isfinite(magnitudes)
    ]

    magnitudes = magnitudes[
        magnitudes >= mc
    ]

    if len(magnitudes) < 10:
        raise ValueError(
            "At least 10 magnitudes are required."
        )

    excess = magnitudes - mc

    statistic, p_value = lilliefors(
        excess,
        dist="exp",
    )

    return statistic, p_value