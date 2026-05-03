import numpy as np

class RuntimeBoom:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        return 1 / 0
