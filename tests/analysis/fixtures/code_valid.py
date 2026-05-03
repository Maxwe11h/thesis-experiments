import numpy as np

class ValidRandomSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        for _ in range(self.budget):
            x = np.random.uniform(-5.0, 5.0, self.dim)
            func(x)
