from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize ,minimize_scalar


@dataclass
class ETASParameters:
    """ETAS model parameters."""

    mu: float
    K: float
    alpha: float
    c: float
    p: float

    def validate(self) -> None:
        """Validate ETAS parameter values."""

        if not np.isfinite(self.mu) or self.mu <= 0:
            raise ValueError("mu must be positive.")

        if not np.isfinite(self.K) or self.K < 0:
            raise ValueError("K must be non-negative.")

        if not np.isfinite(self.alpha):
            raise ValueError("alpha must be finite.")

        if not np.isfinite(self.c) or self.c <= 0:
            raise ValueError("c must be positive.")

        if not np.isfinite(self.p) or self.p <= 0:
            raise ValueError("p must be positive.")

def update_mu(bg: np.ndarray,duration: float,) -> float:
    """Update the background ETAS rate."""

    bg = np.asarray(bg, dtype=float)

    if bg.ndim != 1:
        raise ValueError("bg must be a 1-D array.")

    if len(bg) == 0:
        raise ValueError("bg must contain at least one event.")

    if not np.all(np.isfinite(bg)):
        raise ValueError("bg contains non-finite values.")

    if np.any(bg < 0):
        raise ValueError("bg cannot contain negative probabilities.")

    if duration <= 0:
        raise ValueError("duration must be positive.")

    mu = np.sum(bg) / duration

    if not np.isfinite(mu) or mu <= 0:
        raise ValueError("Updated mu must be positive.")

    return float(mu)   

def update_K(
    rho: np.ndarray,
    times: np.ndarray,
    magnitudes: np.ndarray,
    *,
    alpha: float,
    c: float,
    p: float,
    m0: float = 0.0,
) -> float:
    """Update the ETAS productivity parameter K.

    The update uses the expected number of triggered events from
    the E-step and the integrated temporal triggering kernel.
    """

    rho = np.asarray(rho, dtype=float)
    times = np.asarray(times, dtype=float)
    magnitudes = np.asarray(magnitudes, dtype=float)

    if rho.ndim != 2:
        raise ValueError("rho must be a 2-D array.")

    n = len(times)

    if rho.shape != (n, n):
        raise ValueError("rho must have shape (n_events, n_events).")

    if len(magnitudes) != n:
        raise ValueError(
            "times and magnitudes must have the same length."
        )

    if n < 2:
        raise ValueError("At least two events are required.")

    if not np.all(np.isfinite(rho)):
        raise ValueError("rho contains non-finite values.")

    if np.any(rho < 0):
        raise ValueError("rho cannot contain negative values.")

    if alpha < 0:
        raise ValueError("alpha must be non-negative.")

    if c <= 0:
        raise ValueError("c must be positive.")

    if p <= 0:
        raise ValueError("p must be positive.")

    if np.any(np.diff(times) < 0):
        raise ValueError("times must be sorted chronologically.")

    # Expected number of triggered events.
    expected_triggered = np.sum(rho)

    if expected_triggered <= 0:
        return 0.0

    # Productivity of each possible parent earthquake.
    productivity = np.exp(
        alpha * (magnitudes - m0)
    )

    # Observation duration.
    duration = times[-1] - times[0]

    if duration <= 0:
        raise ValueError("Catalog duration must be positive.")

    
    # Integral from t_j to the end of the observation window:
    # ∫ (t - t_j + c)^(-p) dt
    # For p != 1:
    # [(T - t_j + c)^(1-p) - c^(1-p)] / (1-p)
    

    remaining = times[-1] - times + c

    if np.isclose(p, 1.0):
        integral = np.log(remaining / c)
    else:
        integral = (
            remaining ** (1.0 - p)
            - c ** (1.0 - p)
        ) / (1.0 - p)

    denominator = np.sum(
        productivity * integral
    )

    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError(
            "Invalid denominator while updating K."
        )

    K = expected_triggered / denominator

    if not np.isfinite(K) or K < 0:
        raise ValueError("Updated K is invalid.")

    return float(K)

def update_alpha(
    rho: np.ndarray,
    magnitudes: np.ndarray,
    *,
    m0: float = 0.0,
    alpha_bounds: tuple[float, float] = (0.0, 5.0),
) -> float:
    """Update the ETAS magnitude-productivity parameter alpha.

    The update maximizes the expected triggering log-likelihood
    contributed by the parent magnitudes.

    Parameters
    ----------
    rho[i, j] - probability that event j triggered event i.
    m0 - Reference magnitude.
    alpha_bounds - Lower and upper bounds for alpha.

    """

    rho = np.asarray(rho, dtype=float)
    magnitudes = np.asarray(magnitudes, dtype=float)

    if rho.ndim != 2:
        raise ValueError("rho must be a 2-D array.")

    if magnitudes.ndim != 1:
        raise ValueError("magnitudes must be a 1-D array.")

    n = len(magnitudes)

    if rho.shape != (n, n):
        raise ValueError(
            "rho must have shape (n_events, n_events)."
        )

    if n == 0:
        raise ValueError("magnitudes must contain at least one event.")

    if not np.all(np.isfinite(rho)):
        raise ValueError("rho contains non-finite values.")

    if np.any(rho < 0):
        raise ValueError("rho cannot contain negative values.")

    if not np.all(np.isfinite(magnitudes)):
        raise ValueError("magnitudes contain non-finite values.")

    lower, upper = alpha_bounds

    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("alpha bounds must be finite.")

    if lower < 0:
        raise ValueError("alpha lower bound must be non-negative.")

    if lower >= upper:
        raise ValueError(
            "alpha lower bound must be smaller than upper bound."
        )

    # rho[i, j] represents the probability that j triggered i.
    # Therefore, sum over rows to obtain the expected number of
    # times each event acts as a parent.
    parent_weights = rho.sum(axis=0)

    if np.sum(parent_weights) <= 0:
        # No expected triggered events.
        # There is no information with which to estimate alpha.
        return float(lower)

    delta_m = magnitudes - m0

    def negative_objective(alpha: float) -> float:
        """Negative expected log-productivity."""

        # Expected triggering log-likelihood contribution:
        # Σ_j w_j [alpha * (M_j - M0)]
        
        value = np.sum(
            parent_weights * alpha * delta_m
        )

        # We minimize the negative log-likelihood.
        return -float(value)

    result = minimize_scalar(
        negative_objective,
        bounds=(lower, upper),
        method="bounded",
    )

    if not result.success or not np.isfinite(result.x):
        raise ValueError(
            "Numerical optimization failed while updating alpha."
        )

    alpha = float(result.x)

    if not lower <= alpha <= upper:
        raise ValueError("Updated alpha is outside its bounds.")

    return alpha

def update_c(
    rho,
    times,
    p,
    *,
    c_min=1e-5,
    c_max=10.0,
):
    """Update c by maximizing the weighted temporal log-kernel."""

    rho = np.asarray(rho, dtype=float)

    if np.sum(rho) <= 0:
        return 0.0001
    times = np.asarray(times, dtype=float)

    dt = times[:, None] - times[None, :]

    mask = (
        (rho > 0.0)
        & (dt > 0.0)
    )

    weights = rho[mask]
    delays = dt[mask]

    if weights.size == 0:
        return c_min

    def objective(c):
        kernel = (
            np.log(p - 1.0)
            + (p - 1.0) * np.log(c)
            - p * np.log(c + delays)
        )

        return -np.sum(weights * kernel)

    from scipy.optimize import minimize_scalar

    result = minimize_scalar(
        objective,
        bounds=(c_min, c_max),
        method="bounded",
    )

    if not result.success:
        raise RuntimeError(
            "Optimization of c failed."
        )

    return float(result.x)

def update_p(
    rho,
    times,
    c,
    *,
    p_min=1.01,
    p_max=3.0,
):
    """Update p by maximizing the weighted temporal log-kernel."""

    rho = np.asarray(rho, dtype=float)

    if np.sum(rho) <= 0:
        return 0.5
    times = np.asarray(times, dtype=float)

    dt = times[:, None] - times[None, :]

    mask = (
        (rho > 0.0)
        & (dt > 0.0)
    )

    weights = rho[mask]
    delays = dt[mask]

    if weights.size == 0:
        return p_min

    def objective(p):

        if p <= 1.0:
            return np.inf

        kernel = (
            np.log(p - 1.0)
            + (p - 1.0) * np.log(c)
            - p * np.log(c + delays)
        )

        return -np.sum(weights * kernel)

    from scipy.optimize import minimize_scalar

    result = minimize_scalar(
        objective,
        bounds=(p_min, p_max),
        method="bounded",
    )

    if not result.success:
        raise RuntimeError(
            "Optimization of p failed."
        )

    return float(result.x)

    def negative_objective(p: float) -> float:
        """Negative weighted temporal log-likelihood."""

        log_kernel = -p * np.log(dt_valid + c)

        value = np.sum(weights * log_kernel)

        return -float(value)

    result = minimize_scalar(
        negative_objective,
        bounds=(lower, upper),
        method="bounded",
    )

    if not result.success or not np.isfinite(result.x):
        raise ValueError(
            "Numerical optimization failed while updating p."
        )

    p = float(result.x)

    if not lower <= p <= upper:
        raise ValueError("Updated p is outside its bounds.")

    return p

def expected_complete_log_likelihood(
    times: np.ndarray,
    magnitudes: np.ndarray,
    rho: np.ndarray,
    bg: np.ndarray,
    *,
    mu: float,
    K: float,
    alpha: float,
    c: float,
    p: float,
    m0: float = 0.0,
) -> float:
    """Calculate the full ETAS expected complete-data log-likelihood.

    This is the Q-function used by the EM M-step.

    Parameters
    ----------
    rho[i, j] - probability that event j triggered event i.
    bg - Background probabilities.
    mu, K, alpha, c, p - ETAS parameters.
    m0 -  Reference magnitude.
    """

    times = np.asarray(times, dtype=float)
    magnitudes = np.asarray(magnitudes, dtype=float)
    rho = np.asarray(rho, dtype=float)
    bg = np.asarray(bg, dtype=float)

    n = len(times)

    if n == 0:
        raise ValueError("Catalog must contain at least one event.")

    if magnitudes.shape != (n,):
        raise ValueError(
            "magnitudes must have shape (n_events,)."
        )

    if rho.shape != (n, n):
        raise ValueError(
            "rho must have shape (n_events, n_events)."
        )

    if bg.shape != (n,):
        raise ValueError(
            "bg must have shape (n_events,)."
        )

    if not np.all(np.isfinite(times)):
        raise ValueError("times contain non-finite values.")

    if not np.all(np.isfinite(magnitudes)):
        raise ValueError("magnitudes contain non-finite values.")

    if not np.all(np.isfinite(rho)):
        raise ValueError("rho contains non-finite values.")

    if not np.all(np.isfinite(bg)):
        raise ValueError("bg contains non-finite values.")

    if np.any(rho < 0):
        raise ValueError("rho cannot contain negative values.")

    if np.any(bg < 0):
        raise ValueError("bg cannot contain negative values.")

    if mu <= 0:
        raise ValueError("mu must be positive.")

    if K <= 0:
        raise ValueError("K must be positive.")

    if c <= 0:
        raise ValueError("c must be positive.")

    if p <= 0:
        raise ValueError("p must be positive.")

    if np.any(np.diff(times) < 0):
        raise ValueError(
            "times must be sorted chronologically."
        )

    duration = times[-1] - times[0]

    if duration <= 0:
        raise ValueError(
            "Catalog duration must be positive."
        )

    # Background expected log contribution
    
    background_weight = np.sum(bg)

    background_log_term = (
        background_weight * np.log(mu)
    )

    # Triggering expected log contribution
    

    valid = np.tril(
        np.ones((n, n), dtype=bool),
        k=-1,
    )

    dt = times[:, None] - times[None, :]

    dt_valid = dt[valid]
    rho_valid = rho[valid]

    parent_indices = np.where(valid)[1]

    parent_magnitudes = magnitudes[parent_indices]

    positive = rho_valid > 0

    if np.any(positive):
        log_triggering = (
            np.log(K)
            + alpha * (
                parent_magnitudes[positive] - m0
            )
            - p * np.log(
                dt_valid[positive] + c
            )
        )

        triggering_log_term = np.sum(
            rho_valid[positive] * log_triggering
        )
    else:
        triggering_log_term = 0.0


    # Integrated background intensity
    integrated_background = mu * duration

    # Integrated triggering intensity


    productivity = np.exp(
        alpha * (magnitudes - m0)
    )

    remaining = times[-1] - times + c

    if np.isclose(p, 1.0):
        temporal_integral = np.log(
            remaining / c
        )
    else:
        temporal_integral = (
            remaining ** (1.0 - p)
            - c ** (1.0 - p)
        ) / (1.0 - p)

    integrated_triggering = K * np.sum(
        productivity * temporal_integral
    )

    # Full expected complete-data log-likelihood


    value = (
        background_log_term
        + triggering_log_term
        - integrated_background
        - integrated_triggering
    )

    if not np.isfinite(value):
        raise ValueError(
            "Expected complete-data log-likelihood "
            "is not finite."
        )

    return float(value)

def observed_log_likelihood(
    times: np.ndarray,
    magnitudes: np.ndarray,
    *,
    mu: float,
    K: float,
    alpha: float,
    c: float,
    p: float,
    m0: float = 0.0,
) -> float:
    """Observed-data log-likelihood for the temporal ETAS model."""

    times = np.asarray(times, dtype=float)
    magnitudes = np.asarray(magnitudes, dtype=float)

    n = len(times)

    if n < 2:
        raise ValueError("At least two events are required.")

    if magnitudes.shape != (n,):
        raise ValueError(
            "magnitudes must have shape (n_events,)."
        )

    if np.any(np.diff(times) < 0):
        raise ValueError(
            "times must be sorted chronologically."
        )

    if mu <= 0:
        raise ValueError("mu must be positive.")

    if K < 0:
        raise ValueError("K must be non-negative.")

    if c <= 0:
        raise ValueError("c must be positive.")

    if p <= 0:
        raise ValueError("p must be positive.")

    duration = times[-1] - times[0]

    if duration <= 0:
        raise ValueError(
            "Catalog duration must be positive."
        )

    # ---------------------------------------------------------
    # Conditional intensity at every observed event
    # ---------------------------------------------------------

    dt = times[:, None] - times[None, :]

    valid = np.tril(
        np.ones((n, n), dtype=bool),
        k=-1,
    )

    safe_dt = np.where(valid, dt, 1.0)

    productivity = np.exp(
        alpha * (magnitudes - m0)
    )

    triggering = (
        K
        * productivity[None, :]
        / np.power(safe_dt + c, p)
    )

    triggering = np.where(
        valid,
        triggering,
        0.0,
    )

    intensity = (
        mu
        + triggering.sum(axis=1)
    )

    if np.any(intensity <= 0):
        raise ValueError(
            "ETAS intensity must be positive."
        )

    log_event_term = np.sum(
        np.log(intensity)
    )

    # ---------------------------------------------------------
    # Integrated background intensity
    # ---------------------------------------------------------

    integrated_background = (
        mu * duration
    )

    # ---------------------------------------------------------
    # Integrated triggering intensity
    # ---------------------------------------------------------

    remaining = (
        times[-1]
        - times
        + c
    )

    if np.isclose(p, 1.0):

        temporal_integral = np.log(
            remaining / c
        )

    else:

        temporal_integral = (
            remaining ** (1.0 - p)
            - c ** (1.0 - p)
        ) / (1.0 - p)

    integrated_triggering = (
        K
        * np.sum(
            productivity
            * temporal_integral
        )
    )

    log_likelihood = (
        log_event_term
        - integrated_background
        - integrated_triggering
    )

    if not np.isfinite(log_likelihood):
        raise ValueError(
            "Observed log-likelihood is not finite."
        )

    return float(log_likelihood)    