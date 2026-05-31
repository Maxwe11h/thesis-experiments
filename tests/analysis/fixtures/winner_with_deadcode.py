import numpy as np


class FixtureWinner:
    def __init__(self, budget, dim=5):
        self.budget = budget
        self.dim = dim
        self.lb, self.ub = -5.0, 5.0

    def _never_called_helper(self):       # dead: never invoked
        return np.zeros(self.dim)

    def __call__(self, func):
        best = np.inf
        x = np.random.uniform(self.lb, self.ub, self.dim)
        for _ in range(self.budget):
            cand = np.clip(x + 0.1 * np.random.randn(self.dim), self.lb, self.ub)
            try:
                y = float(func(cand))
            except Exception:
                y = 1e30                  # dead: func never raises here
            if y < best:
                best = y
                x = cand
        return best, x
