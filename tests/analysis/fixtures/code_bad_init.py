import numpy as np

class BadInit:
    def __init__(self):
        self.budget = 100

    def __call__(self, func):
        for _ in range(self.budget):
            func(np.zeros(2))
