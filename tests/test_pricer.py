"""Automated tests for the pricers (run with: .pytest)."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__),".."))

import numpy as np
import pytest

from src.rng import LCG
from src.paths import simulate_gbm
from src.european import bs_price, mc_european
from src.lsm import lsm_price
from src.binomial import crr_price

S0, K, r, q, sigma, T = 100.0, 100.0, 0.05, 0.02, 0.20, 1.0

# ----- Random number generation -----

def test_lcg_reproductible():
    assert np.array_equal(LCG(seed=1).uniform(500), LCG(seed=1).uniform(500))

def test_lcg_uniform_range_and_mean():
    u = LCG(seed=1).uniform(50_000)
    assert u.min() > 0.0 and u.max() < 1.0
    assert abs(u.mean() - 0.5) < 0.01

def test_box_muller_moments_and_length():
    z = LCG(seed=1).normal(50_001) # odd length on purpose
    assert len(z) == 50_001
    assert abs(z.mean()) < 0.02
    assert abs(z.std() - 1.0) < 0.02

# ----- Paths -----

def test_gbm_shape_start_and_positivity():
    paths = simulate_gbm(S0, r, q, sigma, T, 10, 200, LCG(seed=1))
    assert paths.shape == (200, 11)
    assert np.all(paths[:, 0] == S0)
    assert np.all(paths > 0)

# ----- European options -----

def test_put_call_parity():
    c = bs_price(S0, K, r, q, sigma, T, "call")
    p = bs_price(S0, K, r, q, sigma, T, "put")
    assert np.isclose(c - p, S0 * np.exp(-q * T) - K * np.exp(-r * T))

@pytest.mark.parametrize("opt", ["call", "put"])
def test_mc_european_close_to_bsm(opt):
    res = mc_european(S0, K, r, q, sigma, T, 20_000, LCG(seed=7), opt)
    ref = bs_price(S0, K, r, q, sigma, T, opt)
    assert abs(res["price"] - ref) < 4 * res["std_error"]

def test_invalid_option_raises():
    with pytest.raises(ValueError):
        bs_price(S0, K, r, q, sigma, T, "straddle")

#----- Binomial tree -----

def test_crr_european_close_to_bsm():
    tree = crr_price(S0, K, r, q, sigma, T, 500, "put", american=False)
    assert abs(tree - bs_price(S0, K, r, q, sigma, T, "put")) < 0.02

# ----- Longstaff-Schwartz -----

@pytest.mark.parametrize("opt", ["call", "put"])
def test_lsm_one_step_equals_european_mc(opt):
    a = lsm_price(S0, K, r, q, sigma, T, 1, 5_000, LCG(seed=3), opt)
    b = mc_european(S0, K, r, q, sigma, T, 5_000, LCG(seed=3), opt)
    assert np.isclose(a["price"], b["price"])

def test_lsm_american_put_above_european():
    res = lsm_price(S0, K, r, q, sigma, T, 20, 10_000, LCG(seed=42), "put")
    eur = bs_price(S0, K, r, q, sigma, T, "put")
    assert res["price"] > eur - 3 * res["std_error"]

def test_lsm_put_close_to_bermudan_tree():
    res = lsm_price(S0, K, r, q, sigma, T, 20, 10_000, LCG(seed=42), "put")
    berm = crr_price(S0, K, r, q, sigma, T, 1000, "put", american=True, exercise_every=50) # 20 exercise dates

@pytest.mark.parametrize("itm_only", [True, False])
@pytest.mark.parametrize("degree", [2, 3])
def test_lsm_options_run(degree, itm_only):
    res = lsm_price(S0, K, r, q, sigma, T, 10, 2_000, LCG(seed=5), "put", degree=degree, itm_only=itm_only)
    assert np.isfinite(res["price"]) and res["std_error"] > 0