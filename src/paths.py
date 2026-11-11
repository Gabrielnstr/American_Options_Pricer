"""Simulation of Black-Scholes (GBM) paths with contininuous dividend yield."""

import numpy as np

def simulate_gbm(S0, r, q, sigma, T, n_steps, n_paths, rng):
    """Simulate GBM paths with the exact discretisation scheme.
    
    Parameters
    ----------
    S0 : spot price
    r : risk-free rate (continuous)
    q : dividend yield (continuous)
    sigma : volatility
    T : maturity in years
    n_steps : number of time steps (= number of exercise dates)
    n_paths : number of Monte Carlo paths
    rng : random generator exposing .normal(n), e.g. src.rng.LCG

    Returns
    -------
    paths : ndarray of shape (n_paths, n_steps + 1), paths[:, 0] = S0
    """

    dt = T / n_steps
    z = rng.normal(n_paths * n_steps).reshape(n_paths, n_steps)

    drift = (r - q - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    log_paths = np.cumsum(drift + diffusion * z, axis=1)

    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(log_paths)
    return paths