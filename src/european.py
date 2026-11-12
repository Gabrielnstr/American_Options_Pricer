"""European option pricing: closed-form Black-Scholes-Merton and Monte Carlo."""

import numpy as np
from scipy.stats import norm

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