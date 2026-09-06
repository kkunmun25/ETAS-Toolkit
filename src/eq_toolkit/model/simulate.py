"""ETAS catalog simulation module.

Simulates synthetic earthquake catalogs using a branching (Hawkes) point process
formulation of the Epidemic-Type Aftershock Sequence (ETAS) model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class SyntheticCatalog:
    """Container for simulated ETAS catalog events."""

    times: np.ndarray
    magnitudes: np.ndarray
    generations: np.ndarray
    parents: np.ndarray

    def __len__(self) -> int:
        return len(self.times)


def simulate_etas(
    mu: float,
    K: float,
    alpha: float,
    c: float,
    p: float,
    m0: float = 2.5,
    duration: float = 1000.0,
    beta: float = 2.3,
    max_events: int = 25000,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate synthetic earthquake times and magnitudes from an ETAS model.

    Parameters
    ----------
    mu : float
        Background Poisson event rate (events per day).
    K : float
        ETAS productivity parameter.
    alpha : float
        Magnitude scaling parameter.
    c : float
        Omori time offset (days).
    p : float
        Omori decay exponent (must be > 1.0).
    m0 : float
        Magnitude of completeness / reference magnitude.
    duration : float
        Duration of the simulation observation window [0, duration] (days).
    beta : float
        Gutenberg-Richter magnitude exponent beta = b * ln(10).
    max_events : int
        Maximum number of events before terminating (to guard against supercritical explosion).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    times : np.ndarray
        Chronologically sorted event times in days.
    magnitudes : np.ndarray
        Corresponding earthquake magnitudes.
    """
    if mu <= 0:
        raise ValueError("mu must be positive.")
    if K < 0:
        raise ValueError("K must be non-negative.")
    if c <= 0:
        raise ValueError("c must be positive.")
    if p <= 1.0:
        raise ValueError("p must be greater than 1 for finite Omori integral.")
    if duration <= 0:
        raise ValueError("duration must be positive.")
    if beta <= 0:
        raise ValueError("beta must be positive.")

    rng = np.random.default_rng(seed)

    # 1. Background generation (homogeneous Poisson process on [0, duration])
    n_bg = rng.poisson(mu * duration)
    times_list = list(rng.uniform(0.0, duration, n_bg))
    mags_list = list(m0 + rng.exponential(1.0 / beta, n_bg))
    generations_list = [0] * n_bg
    parents_list = [-1] * n_bg

    # Integral of unnormalized triggering kernel (t + c)^(-p) on [0, inf) is c^(1-p) / (p - 1)
    kernel_factor = (c ** (1.0 - p)) / (p - 1.0)

    # 2. Branching process (offspring generation)
    idx = 0
    while idx < len(times_list) and len(times_list) < max_events:
        t_parent = times_list[idx]
        m_parent = mags_list[idx]
        gen_parent = generations_list[idx]

        # Expected number of direct daughters
        expected_daughters = K * np.exp(alpha * (m_parent - m0)) * kernel_factor
        n_daughters = rng.poisson(expected_daughters)

        for _ in range(n_daughters):
            # Sample time delay from normalized Omori distribution:
            # F(t) = 1 - (1 + t/c)^(1-p) = u => t = c * ((1 - u)^(-1/(p-1)) - 1)
            u = rng.random()
            delay = c * ((1.0 - u) ** (-1.0 / (p - 1.0)) - 1.0)
            t_child = t_parent + delay

            if t_child < duration:
                times_list.append(t_child)
                mags_list.append(m0 + rng.exponential(1.0 / beta))
                generations_list.append(gen_parent + 1)
                parents_list.append(idx)

                if len(times_list) >= max_events:
                    break

        idx += 1

    # Chronological sort
    times_arr = np.asarray(times_list, dtype=float)
    mags_arr = np.asarray(mags_list, dtype=float)

    order = np.argsort(times_arr)
    return times_arr[order], mags_arr[order]
