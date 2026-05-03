import numpy as np

class BadCall:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self):
        return None
