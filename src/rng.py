""" Custom random number generation (no numpy.random)."""

import numpy as np

class LCG:
    """ Linear Congruential Generator.

    X_{n+1} = (a * X_n + C) mod m
    U_n = (X_n + 0.5) / m , strictly in (0, 1)

    Default parameters are the ones from Numerical Recipes.
    """

    def __init__(self, seed=12345, a=1664525, c=1012904223, m=2**32):
        self.a = a
        self.c = c
        self.m = m
        self.state = seed % m

    def uniform(self, n):
        """ Return n pseudo-random uniforms in(0, 1) as a numpy array."""
        out = np.empty(n)
        x = self.state
        a, c, m = self.a, self.c, self.m

        for i in range(n):
            x = (a * x + c) % m
            out[i] = (x + 0.5) / m
        self.state = x # keep the state so successive calls continue the sequence
        return out