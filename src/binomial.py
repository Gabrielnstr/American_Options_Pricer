"""Cox-Ross-Rubinstein binomial tree (benchmark for the LSM pricer)."""

import numpy as np
from src.lsm import payoff

def crr_price(S0, K, r, q, sigma, T, n, option="put", american=True, exercise_every=1):
    """Price an option on a CRR tree withn n steps.

    Parameters
    ----------
    american : if False, European option (no early exercise)
    exercise_every : early exercise allowed every 'exercise_every' steps. 1 -> American; n / n_dates -> Bermudan with n_dates exercise dates.
    """

    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # Terminal nodes: j = number of up moves, S = S0 * u^(2j - n)
    j = np.arange(n + 1)
    V = payoff(S0 * u ** (2 * j - n), K, option)

    for i in range(n - 1, -1, -1):
        V = disc * (p * V[1:] + (1 - p) * V[:-1]) # continuation value
        if american and i % exercise_every == 0:
            j = np.arange(i + 1) 
            V = np.maximum(V, payoff(S0 * u ** (2 * j - i), K, option))

    return V[0]