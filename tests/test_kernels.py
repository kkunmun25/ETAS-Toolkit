import numpy as np
import pytest

from eq_toolkit.model.kernels import (
    omori_kernel,
    omori_kernel_log,
    omori_integral,
)


def test_omori_kernel_positive():
    value = omori_kernel(1.0, c=0.1, p=1.2)

    assert value > 0


def test_omori_kernel_log_matches_kernel():
    value = omori_kernel(1.0, c=0.1, p=1.2)
    log_value = omori_kernel_log(1.0, c=0.1, p=1.2)

    assert np.isclose(np.exp(log_value), value)


def test_omori_kernel_normalizes_to_one():
    integral = omori_integral(
        0.0,
        np.inf,
        c=0.1,
        p=1.2,
    )

    assert np.isclose(integral, 1.0)


def test_omori_integral_between_zero_and_one():
    integral = omori_integral(
        0.0,
        10.0,
        c=0.1,
        p=1.2,
    )

    assert 0.0 < integral < 1.0


def test_invalid_c():
    with pytest.raises(ValueError):
        omori_kernel(1.0, c=0.0, p=1.2)


def test_invalid_p():
    with pytest.raises(ValueError):
        omori_kernel(1.0, c=0.1, p=1.0)


def test_negative_time():
    with pytest.raises(ValueError):
        omori_kernel(-1.0, c=0.1, p=1.2)