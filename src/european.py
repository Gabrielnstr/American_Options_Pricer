"""European option pricing: closed-form Black-Scholes-Merton and Monte Carlo."""

import numpy as np
from scipy.stats import norm
from src.paths import simulate_gbm

def bs_price(S0, K, r, q, sigma, T, option="call"):
    """Black-Scholes-Merton price of a European option with dividend yield q.

    Parameters
    ----------
    S0 : spot price
    K : strike
    r : risk-free rate (continuous)
    q : dividend yield (continuous)
    sigma : volatility
    T : maturity in years
    option : "call" or "put"
    """

    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option == "call":
        return S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    if option == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)
    raise ValueError("option must be 'call' or 'put'")

def mc_european(S0, K, r, q, sigma, T, n_paths, rng, option="call"):
    """Monte Carlo price of a European option (one-step exact simulation).
    
    Returns a dict with price, standard error and 95% confidence interval.
    """

    paths = simulate_gbm(S0, r, q, sigma, T, 1, n_paths, rng)
    ST = paths[:, -1]

    if option == "call":
        payoff = np.maximum(ST - K, 0.0)
    elif option == "put":
        payoff = np.maximum(K - ST, 0.0)
    else:
        raise ValueError("option must be 'call' or 'put'")

    disc = np.exp(-r * T) * payoff
    price = disc.mean()
    se = disc.std(ddof=1) / np.sqrt(n_paths)

    return {
        "price": price,
        "std_error": se,
        "ci_low": price - 1.96 * se,
        "ci_high": price + 1.96 * se,
    }