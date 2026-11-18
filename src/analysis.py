"""Analysis helpers: American vs European comparison"""

import pandas as pd
import numpy as np

from src.rng import LCG
from src.european import bs_price
from src.lsm import lsm_price

def compare_american_european(S0, strikes, r, q, sigma, T, n_steps, n_paths, seed=42, degree=2, itm_only=True, options=("call", "put")):
    """Compare LSM American prices with BSM European prices over several strikes.

    The same seed is used for every strike (common random numbers), which reduces the noise on the differences between strikes.

    Returns
    -------
    DataFrame with one row per (option, strike): european, american, std_error, premium (= american - european), premium in % of the European price.
    """

    rows = []
    for opt in options:
        for K in strikes:
            eur = bs_price(S0, K, r, q, sigma, T, opt)
            amer = lsm_price(S0, K, r, q, sigma, T, n_steps, n_paths, LCG(seed=seed), opt, degree=degree, itm_only=itm_only)
            rows.append({
                "option": opt,
                "strike": K,
                "european": eur,
                "american": amer["price"],
                "std_error": amer["std_error"],
                "premium": amer["price"] - eur,
                "premium_pct": 100 * (amer["price"] - eur) / eur,
            })
    return pd.DataFrame(rows)

def convergence_study(S0, K, r, q, sigma, T, n_steps, n_grid, n_rep=10, option="put", degree=2, itm_only=True, base_seed=1000):
    """Study how the LSM confidence interval shrinks with the number of paths.

    For each N in n_grid, the pricer is run n_rep times with different seeds.

    Returns
    -------
    DataFrame with one row per (N, repetition): price, std_error, ci-low, ci-high.
    """

    rows = []
    for N in n_grid:
        for k in range(n_rep):
            res = lsm_price(S0, K, r, q, sigma, T, n_steps, N, LCG(seed=base_seed + 100 * k + N % 97), option, degree=degree, itm_only=itm_only)
            rows.append({"N": N, "rep": k, **res})
    return pd.DataFrame(rows)