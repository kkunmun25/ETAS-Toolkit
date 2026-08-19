"""
Temporal kernels for the ETAS model.

This module implements the normalized modified Omori kernel
used by the temporal ETAS model.
"""

import numpy as np


def omori_kernel(t, c, p):
    """
    Normalized modified Omori temporal kernel.

    g(t) = (p - 1) * c^(p - 1) * (t + c)^(-p)

    Parameters
    ----------
    t : Time since the triggering earthquake.
        Must be non-negative.
    c : Omori time offset. Must be positive.
    p : Omori decay exponent. Must be greater than 1.

    """
    if c <= 0:
        raise ValueError("c must be positive.")

    if p <= 1:
        raise ValueError("p must be greater than 1.")

    t = np.asarray(t, dtype=float)

    if np.any(t < 0):
        raise ValueError("t must be non-negative.")

    return (p - 1.0) * c ** (p - 1.0) * (t + c) ** (-p)


def omori_kernel_log(t, c, p):
    """
    Logarithm of the normalized modified Omori kernel.

    log(g(t)) =log(p - 1) + (p - 1) log(c) - p log(t + c)

    """
    if c <= 0:
        raise ValueError("c must be positive.")

    if p <= 1:
        raise ValueError("p must be greater than 1.")

    t = np.asarray(t, dtype=float)

    if np.any(t < 0):
        raise ValueError("t must be non-negative.")

    return (
        np.log(p - 1.0)
        + (p - 1.0) * np.log(c)
        - p * np.log(t + c)
    )


def omori_integral(a, b, c, p):
    """
    Integral of the normalized modified Omori kernel
    between times a and b.

    Computes

        integral_a^b g(t) dt

    where

        g(t) = (p - 1)c^(p-1)(t+c)^(-p).

    Parameters
    ----------
    a : Lower integration limit. Must be non-negative.
    b : Upper integration limit. Must satisfy b >= a.
    c : Omori time offset. Must be positive.
    p : Omori exponent. Must be greater than 1.

    """
    if c <= 0:
        raise ValueError("c must be positive.")

    if p <= 1:
        raise ValueError("p must be greater than 1.")

    if a < 0:
        raise ValueError("a must be non-negative.")

    if b < a:
        raise ValueError("b must be greater than or equal to a.")

    return (c ** (p - 1.0)* ((a + c) ** (1.0 - p)- (b + c) ** (1.0 - p)))