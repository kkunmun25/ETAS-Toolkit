import numpy as np

from eq_toolkit.quality.bvalue import (
    aki_utsu_b_value,
    shi_bolt_sigma,
    tinti_mulargia_b_value,
    b_positive,
    b_more_positive,
    ols_b_value,
    a_value,
    annualized_rate,
)


def test_aki_utsu():
    magnitudes = np.array([
        2.1, 2.2, 2.3, 2.4, 2.5,
        2.6, 2.7, 2.8, 3.0, 3.2,
    ])

    b = aki_utsu_b_value(
        magnitudes,
        mc=2.0,
        delta_m=0.1,
    )

    assert b > 0
    assert np.isfinite(b)


def test_shi_bolt():
    magnitudes = np.array([
        2.1, 2.2, 2.3, 2.4, 2.5,
        2.6, 2.7, 2.8, 3.0, 3.2,
    ])

    sigma = shi_bolt_sigma(
        magnitudes,
        mc=2.0,
    )

    assert sigma > 0
    assert np.isfinite(sigma)


def test_tinti_mulargia():
    magnitudes = np.array([
        2.1, 2.2, 2.3, 2.4, 2.5,
        2.6, 2.7, 2.8, 3.0, 3.2,
    ])

    b = tinti_mulargia_b_value(
        magnitudes,
        mc=2.0,
    )

    assert b > 0
    assert np.isfinite(b)


def test_b_positive():
    magnitudes = np.array([
        2.0, 2.2, 2.1, 2.4,
        2.3, 2.7, 2.5, 2.9,
    ])

    times = np.arange(len(magnitudes))

    b = b_positive(
        magnitudes,
        times,
    )

    assert b > 0
    assert np.isfinite(b)


def test_b_more_positive():
    magnitudes = np.array([
        2.0, 2.2, 2.1, 2.4,
        2.3, 2.7, 2.5, 2.9,
    ])

    times = np.arange(len(magnitudes))

    b = b_more_positive(
        magnitudes,
        times,
    )

    assert b > 0
    assert np.isfinite(b)


def test_ols():
    magnitudes = np.array([
        2.1, 2.2, 2.3, 2.4, 2.5,
        2.6, 2.7, 2.8, 3.0, 3.2,
    ])

    b = ols_b_value(
        magnitudes,
        mc=2.0,
    )

    assert b > 0
    assert np.isfinite(b)


def test_a_value():
    magnitudes = np.array([
        2.1, 2.2, 2.3, 2.4, 2.5,
    ])

    a = a_value(
        magnitudes,
        b=1.0,
        mc=2.0,
    )

    assert np.isfinite(a)


def test_annualized_rate():
    rate = annualized_rate(
        n_events=500,
        duration_years=10,
    )

    assert rate == 50.0

def test_synthetic_catalog_recovers_b_value():
    rng = np.random.default_rng(42)

    true_b = 1.0
    mc = 2.0
    n = 5000

    beta = true_b * np.log(10.0)

    magnitudes = mc + rng.exponential(
        scale=1.0 / beta,
        size=n,
    )

    b = aki_utsu_b_value(
        magnitudes,
        mc=mc,
        delta_m=0.1,
    )

    sigma = shi_bolt_sigma(
        magnitudes,
        mc=mc,
        delta_m=0.1,
    )

    assert abs(b - true_b) < 0.1
    assert sigma > 0    

def test_tinti_mulargia_close_to_aki_utsu():
    rng = np.random.default_rng(123)

    true_b = 1.0
    mc = 2.0
    n = 3000

    beta = true_b * np.log(10.0)

    magnitudes = mc + rng.exponential(
        scale=1.0 / beta,
        size=n,
    )

    b_aki = aki_utsu_b_value(
        magnitudes,
        mc=mc,
        delta_m=0.1,
    )

    b_tm = tinti_mulargia_b_value(
        magnitudes,
        mc=mc,
        delta_m=0.1,
    )

    assert abs(b_aki - b_tm) < 0.15    

def test_a_value_with_known_catalog():
    magnitudes = np.array([
        2.1, 2.2, 2.3, 2.4, 2.5,
        2.6, 2.7, 2.8, 3.0, 3.2,
    ])

    a = a_value(
        magnitudes,
        b=1.0,
        mc=2.0,
    )

    expected = np.log10(10) + 1.0 * 2.0

    assert np.isclose(a, expected)


def test_annualized_rate_known_duration():
    assert np.isclose(
        annualized_rate(1000, 20),
        50.0,
    )

def test_b_positive_with_changing_completeness():
    rng = np.random.default_rng(2026)

    true_b = 1.0
    beta = true_b * np.log(10.0)

    n = 3000

    magnitudes_1 = 2.0 + rng.exponential(
        scale=1.0 / beta,
        size=n // 2,
    )

    magnitudes_2 = 2.5 + rng.exponential(
        scale=1.0 / beta,
        size=n // 2,
    )

    magnitudes = np.concatenate(
        [magnitudes_1, magnitudes_2]
    )

    times = np.arange(n)

    b = b_positive(
        magnitudes,
        times,
    )

    assert np.isfinite(b)
    assert b > 0    
