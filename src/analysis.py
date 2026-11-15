"""Analysis helpers: American vs European comparison"""

import pandas as pd

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