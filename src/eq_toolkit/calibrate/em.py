from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eq_toolkit.calibrate.estep import compute_estep
from eq_toolkit.calibrate.mstep import (
    ETASParameters,
    expected_complete_log_likelihood,
    observed_log_likelihood,
    update_K,
    update_alpha,
    update_c,
    update_mu,
    update_p,
)


@dataclass
class EMResult:
    """Result returned by the ETAS EM calibration."""

    parameters: ETASParameters
    log_likelihood: float
    iterations: int
    converged: bool
    rho: np.ndarray
    bg: np.ndarray

    def validate(self) -> None:
        """Validate the EM result."""

        self.parameters.validate()

        if not np.isfinite(self.log_likelihood):
            raise ValueError(
                "log_likelihood must be finite."
            )

        if self.iterations < 0:
            raise ValueError(
                "iterations cannot be negative."
            )

        if not isinstance(self.converged, (bool, np.bool_)):
            raise ValueError(
                "converged must be boolean."
            )

        rho = np.asarray(self.rho, dtype=float)
        bg = np.asarray(self.bg, dtype=float)

        if rho.ndim != 2:
            raise ValueError("rho must be a 2-D array.")

        if bg.ndim != 1:
            raise ValueError("bg must be a 1-D array.")

        if rho.shape[0] != rho.shape[1]:
            raise ValueError(
                "rho must be square."
            )

        if rho.shape[0] != len(bg):
            raise ValueError(
                "rho and bg must have the same number of events."
            )

        if not np.all(np.isfinite(rho)):
            raise ValueError(
                "rho contains non-finite values."
            )

        if not np.all(np.isfinite(bg)):
            raise ValueError(
                "bg contains non-finite values."
            )

        if np.any(rho < 0):
            raise ValueError(
                "rho cannot contain negative values."
            )

        if np.any(bg < 0):
            raise ValueError(
                "bg cannot contain negative values."
            )

        # Core E-step invariant:
        #
        # bg[i] + sum_j rho[i, j] = 1
        #
        row_sums = bg + rho.sum(axis=1)

        if not np.allclose(
            row_sums,
            1.0,
            atol=1e-10,
            rtol=1e-10,
        ):
            raise ValueError(
                "EM result violates probability invariant."
            )


def check_monotonicity(
    previous_q: float,
    current_q: float,
    *,
    tolerance: float = 1e-10,
) -> None:
    """Ensure the EM Q-function does not decrease."""

    if not np.isfinite(previous_q):
        raise ValueError("previous_q must be finite.")

    if not np.isfinite(current_q):
        raise ValueError("current_q must be finite.")

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")

    allowed_drop = tolerance * max(
        1.0,
        abs(previous_q),
    )

    if current_q < previous_q - allowed_drop:
        raise ValueError(
            "EM Q-function decreased."
        )        

def run_em(
    times: np.ndarray,
    magnitudes: np.ndarray,
    initial_parameters: ETASParameters,
    *,
    m0: float = 0.0,
    max_iterations: int = 100,
    tolerance: float = 1e-5,
) -> EMResult:
    """Run ETAS Expectation-Maximization calibration."""

    times = np.asarray(times, dtype=float)
    magnitudes = np.asarray(magnitudes, dtype=float)

    if times.ndim != 1:
        raise ValueError("times must be a 1-D array.")

    if magnitudes.ndim != 1:
        raise ValueError(
            "magnitudes must be a 1-D array."
        )

    if len(times) != len(magnitudes):
        raise ValueError(
            "times and magnitudes must have the same length."
        )

    if len(times) < 2:
        raise ValueError(
            "At least two events are required."
        )

    if not np.all(np.isfinite(times)):
        raise ValueError(
            "times contain non-finite values."
        )

    if not np.all(np.isfinite(magnitudes)):
        raise ValueError(
            "magnitudes contain non-finite values."
        )

    if np.any(np.diff(times) < 0):
        raise ValueError(
            "times must be sorted chronologically."
        )

    if max_iterations <= 0:
        raise ValueError(
            "max_iterations must be positive."
        )

    if tolerance <= 0:
        raise ValueError(
            "tolerance must be positive."
        )

    parameters = ETASParameters(
        mu=initial_parameters.mu,
        K=initial_parameters.K,
        alpha=initial_parameters.alpha,
        c=initial_parameters.c,
        p=initial_parameters.p,
    )

    parameters.validate()

    duration = times[-1] - times[0]

    if duration <= 0:
        raise ValueError(
            "Catalog duration must be positive."
        )

    
    rho = np.zeros(
        (len(times), len(times)),
        dtype=float,
    )

    bg = np.ones(
        len(times),
        dtype=float,
    )

    converged = False
    iterations = 0
    previous_ll = -np.inf


    for iteration in range(1, max_iterations + 1):

        # =====================================================
        # E-STEP
        # =====================================================

        estep_result = compute_estep(
            times,
            magnitudes,
            mu=parameters.mu,
            K=parameters.K,
            alpha=parameters.alpha,
            c=parameters.c,
            p=parameters.p,
            m0=m0,
        )

        rho = estep_result.rho
        bg = estep_result.bg

        # =====================================================
        # M-STEP
        # =====================================================

        new_mu = update_mu(
            bg,
            duration,
        )

        new_K = update_K(
            rho,
            times,
            magnitudes,
            alpha=parameters.alpha,
            c=parameters.c,
            p=parameters.p,
            m0=m0,
        )

        new_alpha = update_alpha(
            rho,
            magnitudes,
            m0=m0,
        )

        new_c = update_c(
            rho,
            times,
            p=parameters.p,
        )

        new_p = update_p(
            rho,
            times,
            c=new_c,
        )

        new_parameters = ETASParameters(
            mu=new_mu,
            K=new_K,
            alpha=new_alpha,
            c=new_c,
            p=new_p,
        )

        new_parameters.validate()

        
        # =====================================================
        # OBSERVED LOG-LIKELIHOOD
        # =====================================================

        log_likelihood = observed_log_likelihood(
            times,
            magnitudes,
            mu=new_parameters.mu,
            K=new_parameters.K,
            alpha=new_parameters.alpha,
            c=new_parameters.c,
            p=new_parameters.p,
            m0=m0,
        )

        # =====================================================
        # CONVERGENCE CHECK
        # =====================================================

        if iteration > 1:
            

            improvement = log_likelihood - previous_ll
            

            scale = max(
                1.0,
                abs(previous_ll),
            )

            if abs(improvement) <= tolerance * scale:
                parameters = new_parameters
                iterations = iteration
                converged = True
                previous_ll = log_likelihood
                break

        previous_ll = log_likelihood
        parameters = new_parameters
        iterations = iteration 

    result = EMResult(
        parameters=parameters,
        log_likelihood=previous_ll,
        iterations=iterations,
        converged=converged,
        rho=rho,
        bg=bg,
    )

    result.validate()

    return result 

def run_em_restarts(
    times: np.ndarray,
    magnitudes: np.ndarray,
    initial_parameters: list[ETASParameters],
    *,
    m0: float = 0.0,
    max_iterations: int = 100,
    tolerance: float = 1e-5,
) -> EMResult:
    """Run EM from multiple initial parameter sets.

    The solution with the highest final Q-function value is returned.
    """

    if len(initial_parameters) == 0:
        raise ValueError(
            "At least one initial parameter set is required."
        )

    results = []

    for parameters in initial_parameters:
        result = run_em(
            times,
            magnitudes,
            parameters,
            m0=m0,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )

        results.append(result)

    # Select the solution with the highest final likelihood.
    best_result = max(
        results,
        key=lambda result: result.log_likelihood,
    )

    return best_result       