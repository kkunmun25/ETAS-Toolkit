from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EStepResult:
    """Result of the ETAS E-step.

    Parameters
    ----------
    rho
        Triggering-probability matrix. rho[i, j] is the probability
        that event j triggered event i.
    bg
        Background probability vector. bg[i] is the probability
        that event i is a background event.
    """

    rho: np.ndarray
    bg: np.ndarray

    def validate(self, atol: float = 1e-10) -> None:
        """Check the E-step probability invariants."""

        if self.rho.ndim != 2:
            raise ValueError("rho must be a 2-D matrix.")

        if self.rho.shape[0] != self.rho.shape[1]:
            raise ValueError("rho must be a square matrix.")

        n = self.rho.shape[0]

        if self.bg.shape != (n,):
            raise ValueError("bg must have shape (n_events,).")

        if not np.all(np.isfinite(self.rho)):
            raise ValueError("rho contains non-finite values.")

        if not np.all(np.isfinite(self.bg)):
            raise ValueError("bg contains non-finite values.")

        if np.any(self.rho < -atol):
            raise ValueError("rho contains negative probabilities.")

        if np.any(self.bg < -atol):
            raise ValueError("bg contains negative probabilities.")

        # An event cannot trigger itself.
        if not np.allclose(np.diag(self.rho), 0.0, atol=atol):
            raise ValueError("rho diagonal must be zero.")

        # With a chronological catalog, future events cannot trigger
        # earlier events.
        if not np.allclose(np.triu(self.rho), 0.0, atol=atol):
            raise ValueError(
                "rho must be lower triangular for a chronological catalog."
            )

        # Core E-step invariant:
        #
        # bg[i] + sum_j rho[i, j] = 1
        row_sums = self.bg + self.rho.sum(axis=1)

        if not np.allclose(row_sums, 1.0, atol=atol):
            raise ValueError(
                "E-step probability invariant violated: "
                "bg[i] + sum_j rho[i,j] must equal 1."
            )


def make_estep_result(
    rho: np.ndarray,
    bg: np.ndarray,
    *,
    atol: float = 1e-10,
) -> EStepResult:
    """Create and validate an E-step result."""

    rho = np.asarray(rho, dtype=float)
    bg = np.asarray(bg, dtype=float)

    result = EStepResult(rho=rho, bg=bg)
    result.validate(atol=atol)

    return result

def compute_estep(
    times: np.ndarray,
    magnitudes: np.ndarray,
    *,
    mu: float,
    K: float,
    alpha: float,
    c: float,
    p: float,
    m0: float = 0.0,
) -> EStepResult:
    """Compute ETAS E-step probabilities using NumPy broadcasting."""

    times = np.asarray(times, dtype=float)
    magnitudes = np.asarray(magnitudes, dtype=float)

    if times.ndim != 1:
        raise ValueError("times must be a 1-D array.")

    if magnitudes.ndim != 1:
        raise ValueError("magnitudes must be a 1-D array.")

    if len(times) != len(magnitudes):
        raise ValueError("times and magnitudes must have the same length.")

    if len(times) == 0:
        raise ValueError("catalog must contain at least one event.")

    if not np.all(np.isfinite(times)):
        raise ValueError("times contain non-finite values.")

    if not np.all(np.isfinite(magnitudes)):
        raise ValueError("magnitudes contain non-finite values.")

    if np.any(np.diff(times) < 0):
        raise ValueError("times must be sorted chronologically.")

    if mu <= 0:
        raise ValueError("mu must be positive.")

    if K < 0:
        raise ValueError("K must be non-negative.")

    if c <= 0:
        raise ValueError("c must be positive.")

    if p <= 0:
        raise ValueError("p must be positive.")

    n = len(times)

    
    # 1. Time differences
    # dt[i, j] = times[i] - times[j]


    dt = times[:, None] - times[None, :]

    # Only earlier events can trigger the current event.
    valid = np.tril(np.ones((n, n), dtype=bool), k=-1)

    # Avoid invalid calculations for diagonal/upper triangle.
    safe_dt = np.where(valid, dt, 1.0)

    
    # 2. Productivity of each possible parent earthquake


    productivity = K * np.exp( alpha * (magnitudes - m0))

    
    # 3. ETAS temporal triggering kernel
   

    triggering = (productivity[None, :] / np.power(safe_dt + c, p))

    # Remove impossible parent-child pairs.
    triggering = np.where(valid, triggering, 0.0)

    # 4. Total conditional intensity for every event


    total_intensity = mu + triggering.sum(axis=1)

    if not np.all(np.isfinite(total_intensity)):
        raise ValueError("Invalid total intensity.")

    if np.any(total_intensity <= 0):
        raise ValueError("Total intensity must be positive.")

    # 5. Normalize to obtain probabilities

    rho = triggering / total_intensity[:, None]
    bg = mu / total_intensity

    # 6. Package and validate

    result = EStepResult(rho=rho,bg=bg,)

    result.validate()

    return result