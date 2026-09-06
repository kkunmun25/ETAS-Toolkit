"""ETAS parameter uncertainty quantification module.

Implements two complementary methods for estimating parameter uncertainties:
1. Observed Fisher Information / Inverse Hessian method using central finite differences.
2. Parametric bootstrap using Hawkes branching simulation and EM refitting.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from eq_toolkit.calibrate.em import run_em
from eq_toolkit.calibrate.mstep import ETASParameters, observed_log_likelihood
from eq_toolkit.model.likelihood import temporal_log_likelihood
from eq_toolkit.model.simulate import simulate_etas


PARAM_NAMES = ["mu", "K", "alpha", "c", "p"]


@dataclass
class HessianResult:
    """Results from the inverse Hessian / observed Fisher information method."""

    mle: np.ndarray
    hessian: np.ndarray
    fisher_info: np.ndarray
    eigenvalues: np.ndarray
    covariance: np.ndarray
    standard_errors: np.ndarray
    ci_95_lower: np.ndarray
    ci_95_upper: np.ndarray
    correlation_matrix: np.ndarray
    is_positive_definite: bool
    likelihood_name: str


@dataclass
class BootstrapResult:
    """Results from the parametric bootstrap method."""

    n_requested: int
    n_successful: int
    n_converged: int
    n_iteration_limit: int
    n_boundary_hits: int
    exclusion_rate: float
    param_estimates: np.ndarray  # Shape: (n_valid, 5)
    bootstrap_mean: np.ndarray
    bootstrap_se: np.ndarray
    ci_95_lower: np.ndarray  # 2.5 percentile
    ci_95_upper: np.ndarray  # 97.5 percentile
    warning_compensator_bug: bool


@dataclass
class UncertaintyResult:
    """Combined uncertainty quantification results."""

    mle: ETASParameters
    hessian_res: HessianResult
    bootstrap_res: Optional[BootstrapResult] = None


def check_compensator_bug() -> bool:
    """Check whether the compensator-term bug in mstep.py is still present.

    Returns
    -------
    bool
        True if the bug is detected, False otherwise.
    """
    import inspect
    from eq_toolkit.calibrate import mstep

    src_c = inspect.getsource(mstep.update_c)
    src_p = inspect.getsource(mstep.update_p)

    # In the buggy implementation, update_c and update_p maximize only the
    # event log-kernel without subtracting the integrated temporal compensator
    bug_c = "delays" in src_c and "np.log(c + delays)" in src_c and "integral" not in src_c
    bug_p = "delays" in src_p and "np.log(c + delays)" in src_p and "integral" not in src_p

    return bug_c or bug_p


def compute_hessian_uncertainty(
    times: np.ndarray,
    magnitudes: np.ndarray,
    parameters: ETASParameters,
    m0: float = 2.5,
    likelihood_func: str = "likelihood.py",
    step_scale: float = 1e-3,
) -> HessianResult:
    """Compute parameter uncertainties using the observed Fisher information.

    Numerically evaluates the Hessian matrix of the log-likelihood at the MLE
    using central finite differences scaled to each parameter's magnitude.

    Parameters
    ----------
    times : np.ndarray
        Array of event times in days.
    magnitudes : np.ndarray
        Array of event magnitudes.
    parameters : ETASParameters
        Fitted ETAS MLE point estimates.
    m0 : float
        Reference magnitude / magnitude of completeness.
    likelihood_func : str
        Which log-likelihood function to use: "likelihood.py" (default) or "mstep.py".
    step_scale : float
        Relative step size epsilon for central differences (h_i = epsilon * theta_i).

    Returns
    -------
    HessianResult
        Structured container with Hessian, Fisher Information, Covariance, SE, CI, and correlations.
    """
    theta_0 = np.array([
        parameters.mu,
        parameters.K,
        parameters.alpha,
        parameters.c,
        parameters.p,
    ], dtype=float)

    n_params = len(theta_0)

    # Pre-select log-likelihood function
    if likelihood_func == "likelihood.py":
        def ll_eval(theta: np.ndarray) -> float:
            if theta[0] <= 0 or theta[1] < 0 or theta[2] < 0 or theta[3] <= 0 or theta[4] <= 1.0:
                return -np.inf
            return temporal_log_likelihood(
                times=times,
                magnitudes=magnitudes,
                mu=float(theta[0]),
                K=float(theta[1]),
                alpha=float(theta[2]),
                M0=m0,
                c=float(theta[3]),
                p=float(theta[4]),
            )
    elif likelihood_func == "mstep.py":
        # Fast vectorized version with precomputed safe_dt
        n_events = len(times)
        duration = times[-1] - times[0]
        valid_mask = np.tril(np.ones((n_events, n_events), dtype=bool), k=-1)
        dt_mat = times[:, None] - times[None, :]
        safe_dt = np.where(valid_mask, dt_mat, 1.0)
        delta_m = magnitudes - m0
        rem = times[-1] - times

        def ll_eval(theta: np.ndarray) -> float:
            mu_val, K_val, alpha_val, c_val, p_val = theta
            if mu_val <= 0 or K_val < 0 or c_val <= 0 or p_val <= 0:
                return -np.inf
            prod = np.exp(alpha_val * delta_m)
            trig = K_val * prod[None, :] / np.power(safe_dt + c_val, p_val)
            trig = np.where(valid_mask, trig, 0.0)
            intensity = mu_val + trig.sum(axis=1)
            if np.any(intensity <= 0):
                return -np.inf
            log_event = np.sum(np.log(intensity))
            int_bg = mu_val * duration
            rem_c = rem + c_val
            if np.isclose(p_val, 1.0):
                temp_int = np.log(rem_c / c_val)
            else:
                temp_int = (rem_c ** (1.0 - p_val) - c_val ** (1.0 - p_val)) / (1.0 - p_val)
            int_trig = K_val * np.sum(prod * temp_int)
            return float(log_event - int_bg - int_trig)
    else:
        raise ValueError(f"Unknown likelihood_func: {likelihood_func}")

    # Central difference step sizes scaled to parameter magnitudes
    h = np.maximum(step_scale * np.abs(theta_0), 1e-6)

    # Evaluate base log-likelihood
    f0 = ll_eval(theta_0)
    if not np.isfinite(f0):
        raise ValueError(f"Log-likelihood at MLE is non-finite: {f0}")

    H = np.zeros((n_params, n_params), dtype=float)

    # 1. Diagonal elements: d^2 f / d theta_i^2
    for i in range(n_params):
        step_i = np.zeros(n_params)
        step_i[i] = h[i]
        f_plus = ll_eval(theta_0 + step_i)
        f_minus = ll_eval(theta_0 - step_i)
        H[i, i] = (f_plus - 2.0 * f0 + f_minus) / (h[i] ** 2)

    # 2. Off-diagonal elements: d^2 f / (d theta_i d theta_j)
    for i in range(n_params):
        for j in range(i + 1, n_params):
            step_i = np.zeros(n_params)
            step_i[i] = h[i]
            step_j = np.zeros(n_params)
            step_j[j] = h[j]

            f_pp = ll_eval(theta_0 + step_i + step_j)
            f_pm = ll_eval(theta_0 + step_i - step_j)
            f_mp = ll_eval(theta_0 - step_i + step_j)
            f_mm = ll_eval(theta_0 - step_i - step_j)

            val = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h[i] * h[j])
            H[i, j] = val
            H[j, i] = val

    # Observed Fisher information
    fisher_info = -H

    # Check eigenvalues for positive definiteness
    eigenvals = np.linalg.eigvalsh(fisher_info)
    is_pos_def = bool(np.all(eigenvals > 0))

    # Invert to obtain asymptotic covariance matrix
    cov = np.full((n_params, n_params), np.nan)
    try:
        cov = np.linalg.inv(fisher_info)
    except np.linalg.LinAlgError:
        warnings.warn("Observed Fisher information matrix is singular and cannot be inverted.")

    # Standard errors and 95% Wald confidence intervals
    diag_cov = np.diag(cov)
    se = np.where(diag_cov > 0, np.sqrt(np.maximum(diag_cov, 0.0)), np.nan)
    ci_lower = theta_0 - 1.959964 * se
    ci_upper = theta_0 + 1.959964 * se

    # Parameter correlation matrix
    outer_se = np.outer(se, se)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(outer_se > 0, cov / outer_se, np.nan)

    return HessianResult(
        mle=theta_0,
        hessian=H,
        fisher_info=fisher_info,
        eigenvalues=eigenvals,
        covariance=cov,
        standard_errors=se,
        ci_95_lower=ci_lower,
        ci_95_upper=ci_upper,
        correlation_matrix=corr,
        is_positive_definite=is_pos_def,
        likelihood_name=likelihood_func,
    )


def parametric_bootstrap_uncertainty(
    parameters: ETASParameters,
    m0: float = 2.5,
    duration: float = 4017.58,
    beta: float = 2.28,
    n_bootstraps: int = 200,
    max_iterations: int = 60,
    tolerance: float = 1e-4,
    max_events: int = 15000,
    seed: int = 42,
    verbose: bool = True,
) -> BootstrapResult:
    """Run parametric bootstrap to estimate parameter sampling distributions.

    Parameters
    ----------
    parameters : ETASParameters
        Fitted ETAS model parameters used to generate synthetic catalogs.
    m0 : float
        Reference magnitude / magnitude of completeness.
    duration : float
        Duration of catalog observation window (days).
    beta : float
        Gutenberg-Richter magnitude exponent beta = b * ln(10).
    n_bootstraps : int
        Number of synthetic catalogs to simulate and refit (recommended ~200-500).
    max_iterations : int
        Maximum EM iterations per refit.
    tolerance : float
        EM convergence tolerance.
    max_events : int
        Safety cap on event count to avoid out-of-memory errors on large bursts.
    seed : int
        Base random seed for reproducibility.
    verbose : bool
        Whether to print periodic progress updates.

    Returns
    -------
    BootstrapResult
        Detailed bootstrap summary statistics and convergence diagnostics.
    """
    has_compensator_bug = check_compensator_bug()
    if has_compensator_bug:
        warnings.warn(
            "\n" + "=" * 78 + "\n"
            "CRITICAL WARNING: COMPENSATOR-TERM BUG DETECTED IN mstep.py!\n"
            "update_c and update_p omit the integrated compensator term and use an incompatible\n"
            "normalized Omori log-kernel. During EM refits, parameter estimates will be heavily\n"
            "biased towards boundaries (c -> lower bound, p -> upper bound, alpha -> 5, K -> 0).\n"
            "Bootstrap refit results will reflect this structural EM bias.\n"
            + "=" * 78,
            UserWarning,
            stacklevel=2,
        )

    rng = np.random.default_rng(seed)
    valid_fits = []
    n_converged = 0
    n_iter_limit = 0
    n_boundary = 0

    init_params = ETASParameters(
        mu=parameters.mu,
        K=parameters.K,
        alpha=parameters.alpha,
        c=parameters.c,
        p=parameters.p,
    )

    t_start_all = time.time()

    for b_idx in range(1, n_bootstraps + 1):
        cat_seed = int(rng.integers(0, 1_000_000_000))
        try:
            # Simulate synthetic catalog
            sim_times, sim_mags = simulate_etas(
                mu=parameters.mu,
                K=parameters.K,
                alpha=parameters.alpha,
                c=parameters.c,
                p=parameters.p,
                m0=m0,
                duration=duration,
                beta=beta,
                max_events=max_events,
                seed=cat_seed,
            )

            # Refit with EM
            result = run_em(
                times=sim_times,
                magnitudes=sim_mags,
                initial_parameters=init_params,
                m0=m0,
                max_iterations=max_iterations,
                tolerance=tolerance,
            )

            prm = result.parameters
            is_at_boundary = (
                prm.alpha >= 4.95
                or prm.c <= 1.05e-5
                or prm.p >= 2.95
                or prm.K <= 1e-5
            )

            if is_at_boundary:
                n_boundary += 1

            if result.converged and not is_at_boundary:
                n_converged += 1
                valid_fits.append([prm.mu, prm.K, prm.alpha, prm.c, prm.p])
            else:
                if result.iterations >= max_iterations:
                    n_iter_limit += 1

        except (MemoryError, np.core._exceptions._ArrayMemoryError):
            n_iter_limit += 1
        except Exception as e:
            warnings.warn(f"Bootstrap sample {b_idx} failed with error: {e}")

        if verbose and (b_idx % 10 == 0 or b_idx == n_bootstraps):
            elapsed = time.time() - t_start_all
            print(
                f"Bootstrap [{b_idx:3d}/{n_bootstraps}] "
                f"Valid: {len(valid_fits)} | "
                f"Converged: {n_converged} | "
                f"Boundary hits: {n_boundary} | "
                f"Time: {elapsed:.1f}s"
            )

    n_valid = len(valid_fits)
    exclusion_rate = 1.0 - (n_valid / float(n_bootstraps)) if n_bootstraps > 0 else 0.0

    if n_valid > 1:
        estimates = np.asarray(valid_fits, dtype=float)
        b_mean = np.mean(estimates, axis=0)
        b_se = np.std(estimates, axis=0, ddof=1)
        ci_lower = np.percentile(estimates, 2.5, axis=0)
        ci_upper = np.percentile(estimates, 97.5, axis=0)
    else:
        estimates = np.empty((0, 5))
        b_mean = np.full(5, np.nan)
        b_se = np.full(5, np.nan)
        ci_lower = np.full(5, np.nan)
        ci_upper = np.full(5, np.nan)

    return BootstrapResult(
        n_requested=n_bootstraps,
        n_successful=n_valid,
        n_converged=n_converged,
        n_iteration_limit=n_iter_limit,
        n_boundary_hits=n_boundary,
        exclusion_rate=exclusion_rate,
        param_estimates=estimates,
        bootstrap_mean=b_mean,
        bootstrap_se=b_se,
        ci_95_lower=ci_lower,
        ci_95_upper=ci_upper,
        warning_compensator_bug=has_compensator_bug,
    )


def format_summary_table(
    mle: ETASParameters,
    hessian_res: HessianResult,
    boot_res: Optional[BootstrapResult] = None,
) -> str:
    """Format comparison summary table as markdown text."""
    headers = [
        "Parameter",
        "MLE",
        "Hessian SE",
        "Hessian 95% CI",
        "Bootstrap SE",
        "Bootstrap 95% CI",
    ]
    rows = []
    mle_vals = [mle.mu, mle.K, mle.alpha, mle.c, mle.p]

    for idx, name in enumerate(PARAM_NAMES):
        mle_str = f"{mle_vals[idx]:.6g}"
        h_se = hessian_res.standard_errors[idx]
        h_se_str = f"{h_se:.4e}" if np.isfinite(h_se) else "NaN (indef)"
        if np.isfinite(h_se):
            h_ci_str = f"[{hessian_res.ci_95_lower[idx]:.4g}, {hessian_res.ci_95_upper[idx]:.4g}]"
        else:
            h_ci_str = "[NaN, NaN]"

        if boot_res is not None and boot_res.n_successful > 1:
            b_se = boot_res.bootstrap_se[idx]
            b_se_str = f"{b_se:.4e}" if np.isfinite(b_se) else "NaN"
            b_ci_str = f"[{boot_res.ci_95_lower[idx]:.4g}, {boot_res.ci_95_upper[idx]:.4g}]"
        else:
            b_se_str = "N/A"
            b_ci_str = "N/A"

        rows.append([name, mle_str, h_se_str, h_ci_str, b_se_str, b_ci_str])

    # Build markdown table
    col_widths = [max(len(row[col]) for row in [headers] + rows) for col in range(len(headers))]
    lines = []
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    lines.append(f"| {header_line} |")
    sep_line = " | ".join("-" * w for w in col_widths)
    lines.append(f"| {sep_line} |")
    for row in rows:
        row_line = " | ".join(c.ljust(w) for c, w in zip(row, col_widths))
        lines.append(f"| {row_line} |")

    return "\n".join(lines)


def format_correlation_matrix(corr_mat: np.ndarray, names: List[str] = PARAM_NAMES) -> str:
    """Format parameter correlation matrix as markdown table."""
    headers = [""] + names
    rows = []
    for i, name in enumerate(names):
        row = [name]
        for j in range(len(names)):
            val = corr_mat[i, j]
            row.append(f"{val:+.4f}" if np.isfinite(val) else "   NaN  ")
        rows.append(row)

    col_widths = [max(len(row[col]) for row in [headers] + rows) for col in range(len(headers))]
    lines = []
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    lines.append(f"| {header_line} |")
    sep_line = " | ".join("-" * w for w in col_widths)
    lines.append(f"| {sep_line} |")
    for row in rows:
        row_line = " | ".join(c.ljust(w) for c, w in zip(row, col_widths))
        lines.append(f"| {row_line} |")

    return "\n".join(lines)


def load_scsn_data(catalog_path: Path = Path("sc-catalog.txt"), mc: float = 2.5) -> Tuple[np.ndarray, np.ndarray]:
    """Parse SCSN catalog file into elapsed days and magnitudes."""
    rows = []
    with catalog_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("<"):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            try:
                dt_str = parts[0] + " " + parts[1]
                mag = float(parts[4])
                if mag >= mc:
                    rows.append((dt_str, mag))
            except (ValueError, IndexError):
                continue

    df = pd.DataFrame(rows, columns=["time", "mag"])
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna().sort_values("time").reset_index(drop=True)

    times = ((df["time"] - df["time"].iloc[0]).dt.total_seconds() / 86400.0).to_numpy()
    mags = df["mag"].to_numpy()
    return times, mags


def main():
    """Main CLI entrypoint for running uncertainty analysis."""
    parser = argparse.ArgumentParser(description="ETAS Parameter Uncertainty Analysis")
    parser.add_argument("--catalog", type=str, default="sc-catalog.txt", help="Path to catalog file")
    parser.add_argument("--m0", type=float, default=2.5, help="Reference magnitude / Mc")
    parser.add_argument("--n-bootstraps", type=int, default=200, help="Number of bootstrap refits")
    parser.add_argument("--skip-bootstrap", action="store_true", help="Skip parametric bootstrap")
    parser.add_argument("--step-scale", type=float, default=1e-3, help="Finite difference relative step size")
    args = parser.parse_args()

    # User's MLE estimates
    mle = ETASParameters(
        mu=0.6124868102,
        K=0.03689194717,
        alpha=1.235114703,
        c=0.04092132959,
        p=1.457064597,
    )
    logL = 11090.492067

    print("=" * 80)
    print("ETAS PARAMETER UNCERTAINTY QUANTIFICATION")
    print("=" * 80)
    print("Fitted Model Estimates:")
    print(f"  mu         = {mle.mu:.10f}")
    print(f"  K          = {mle.K:.11f}")
    print(f"  alpha      = {mle.alpha:.9f}")
    print(f"  c          = {mle.c:.11f}")
    print(f"  p          = {mle.p:.9f}")
    print(f"  logL       = {logL:.6f}")
    print()

    # Check for mstep bug
    has_bug = check_compensator_bug()
    if has_bug:
        print("!" * 80)
        print("COMPENSATOR-TERM BUG WARNING:")
        print("  Detected compensator-term bug in src/eq_toolkit/calibrate/mstep.py.")
        print("  update_c() and update_p() omit the integrated intensity compensator and")
        print("  use an uncoordinated normalized Omori log-kernel.")
        print("  Bootstrap refits using calibrate/em.py will be biased toward parameter")
        print("  boundaries (alpha -> 5, K -> 0, c -> lower bound, p -> upper bound).")
        print("!" * 80)
        print()

    # Load catalog data
    catalog_path = Path(args.catalog)
    if not catalog_path.exists():
        print(f"Error: Catalog file {catalog_path} not found.")
        sys.exit(1)

    print(f"Loading catalog from {catalog_path} (Mc = {args.m0})...")
    times, mags = load_scsn_data(catalog_path, mc=args.m0)
    duration = times[-1] - times[0]
    beta = 1.0 / (np.mean(mags) - args.m0)
    print(f"Loaded {len(times)} events spanning {duration:.2f} days. (beta = {beta:.3f})")
    print()

    # 1. Observed Fisher Information / Inverse Hessian
    print("1. COMPUTING OBSERVED FISHER INFORMATION / INVERSE HESSIAN...")
    print("   Using existing likelihood.py log-likelihood function (central differences)...")
    t0_hess = time.time()
    hess_res_temp = compute_hessian_uncertainty(
        times=times,
        magnitudes=mags,
        parameters=mle,
        m0=args.m0,
        likelihood_func="likelihood.py",
        step_scale=args.step_scale,
    )
    print(f"   Hessian evaluation completed in {time.time() - t0_hess:.2f}s.")
    print(f"   Positive definite (-H): {hess_res_temp.is_positive_definite}")
    print(f"   Eigenvalues of -H: {hess_res_temp.eigenvalues}")
    print()

    # Also compute Hessian on observed_log_likelihood (mstep.py) for comparison
    print("   Computing Hessian under observed_log_likelihood (mstep.py) for comparison...")
    hess_res_obs = compute_hessian_uncertainty(
        times=times,
        magnitudes=mags,
        parameters=mle,
        m0=args.m0,
        likelihood_func="mstep.py",
        step_scale=args.step_scale,
    )
    print(f"   Positive definite (-H_obs): {hess_res_obs.is_positive_definite}")
    print(f"   Eigenvalues of -H_obs: {hess_res_obs.eigenvalues}")
    print()

    # 2. Parametric Bootstrap
    boot_res = None
    if not args.skip_bootstrap:
        print(f"2. RUNNING PARAMETRIC BOOTSTRAP (n = {args.n_bootstraps})...")
        boot_res = parametric_bootstrap_uncertainty(
            parameters=mle,
            m0=args.m0,
            duration=duration,
            beta=beta,
            n_bootstraps=args.n_bootstraps,
            max_iterations=60,
            tolerance=1e-4,
            max_events=12000,
            seed=42,
            verbose=True,
        )
        print()
        print(f"   Total refits requested: {boot_res.n_requested}")
        print(f"   Successful / Valid refits: {boot_res.n_successful}")
        print(f"   Converged refits: {boot_res.n_converged}")
        print(f"   Boundary hits: {boot_res.n_boundary_hits}")
        print(f"   Iteration limit / memory hits: {boot_res.n_iteration_limit}")
        print(f"   Exclusion rate: {boot_res.exclusion_rate * 100:.1f}%")
        print()

    # 3. Output Summary Table
    print("=" * 80)
    print("SUMMARY TABLE: MLE vs HESSIAN SE vs BOOTSTRAP SE (temporal_log_likelihood)")
    print("=" * 80)
    print(format_summary_table(mle, hess_res_temp, boot_res))
    print()

    print("=" * 80)
    print("SUMMARY TABLE: MLE vs HESSIAN SE (observed_log_likelihood from mstep.py)")
    print("=" * 80)
    print(format_summary_table(mle, hess_res_obs, boot_res))
    print()

    # 4. Correlation Matrices
    print("=" * 80)
    print("PARAMETER CORRELATION MATRIX (from likelihood.py Hessian):")
    print("=" * 80)
    print(format_correlation_matrix(hess_res_temp.correlation_matrix))
    print()

    print("=" * 80)
    print("PARAMETER CORRELATION MATRIX (from mstep.py Hessian):")
    print("=" * 80)
    print(format_correlation_matrix(hess_res_obs.correlation_matrix))
    print()


if __name__ == "__main__":
    main()
