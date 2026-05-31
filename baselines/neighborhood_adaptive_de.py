# NeighborhoodAdaptiveDE -- van Stein (2025), GECCO Companion;
# winner of the 2025 MA-BBOB anytime competition (LLaMEA-SAGE ref [16]).
#
# Verbatim source kept in baselines/neighborhood_adaptive_de_original.py.
# This file changes ONLY the I/O plumbing so the algorithm runs under the
# section 5.4.6 full-suite harness (experiments/phase4_full_suite_runner.py). The
# DE logic (neighbourhood mutation, orthogonal crossover, selection, adaptive
# F/CR, stagnation restart) is unchanged. Changes:
#   1. __init__ receives the harness's pre-multiplied budget (= 2000*dim)
#      instead of budget_factor; self.budget = budget.
#   2. Bounds hardcoded to the MA-BBOB box [-5, 5]; dim taken from the
#      constructor. The harness passes a plain FE-counting closure that has no
#      .bounds / .meta_data (the four LLM winners hardcode the box the same way).
#   3. __call__(self, func) draws randomness from the harness's per-run seed;
#      the original's internal np.random.seed(seed) is removed so the five
#      eval-seeds differ (otherwise NADE's seed variance would be exactly zero).
#   4. np.Inf -> np.inf (np.Inf was removed in NumPy 2.0).

import numpy as np


class NeighborhoodAdaptiveDE:
    def __init__(self, budget, dim=10, pop_size=50, adapt_freq=50,
                 stagnation_threshold=1000, learning_rate=0.1, neighborhood_size=5):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.pop_size = pop_size
        self.adapt_freq = adapt_freq
        self.stagnation_threshold = stagnation_threshold
        self.learning_rate = learning_rate
        self.neighborhood_size = neighborhood_size
        self.f = 0.5
        self.cr = 0.9
        self.success_f = []
        self.success_cr = []
        self.best_fitness_history = []

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None
        self.population = np.random.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.evals = self.pop_size
        self.last_improvement = 0

        best_idx = np.argmin(self.fitness)
        self.f_opt = self.fitness[best_idx]
        self.x_opt = self.population[best_idx]
        self.best_fitness_history.append(self.f_opt)

        while self.evals < self.budget:
            for i in range(self.pop_size):
                # Neighborhood-based Mutation
                neighbors = np.random.choice(np.arange(self.pop_size), self.neighborhood_size, replace=False)
                best_neighbor_idx = neighbors[np.argmin(self.fitness[neighbors])]

                idxs = np.random.choice(np.arange(self.pop_size), 2, replace=False)
                x1, x2 = self.population[idxs]

                v = self.population[best_neighbor_idx] + self.f * (x1 - x2)
                v = np.clip(v, self.lb, self.ub)

                # Orthogonal Crossover
                u = np.copy(self.population[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.cr or j == j_rand:
                        u[j] = v[j]

                # Evaluation
                f_new = func(u)
                self.evals += 1

                # Selection
                if f_new < self.fitness[i]:
                    self.success_f.append(self.f)
                    self.success_cr.append(self.cr)
                    self.fitness[i] = f_new
                    self.population[i] = u

                    if f_new < self.f_opt:
                        self.f_opt = f_new
                        self.x_opt = u
                        self.best_fitness_history.append(self.f_opt)
                        self.last_improvement = self.evals

            # Adaptive Parameter Control
            if self.evals % self.adapt_freq == 0:
                if self.success_f:
                    self.f = (1 - self.learning_rate) * self.f + self.learning_rate * np.mean(self.success_f)
                    self.cr = (1 - self.learning_rate) * self.cr + self.learning_rate * np.mean(self.success_cr)
                self.f = np.clip(self.f, 0.1, 0.9)
                self.cr = np.clip(self.cr, 0.1, 1.0)
                self.success_f = []
                self.success_cr = []

            # Stagnation Check and Restart
            if self.evals - self.last_improvement > self.stagnation_threshold:
                self.population = np.random.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
                self.fitness = np.array([func(x) for x in self.population])
                self.evals += self.pop_size
                best_idx = np.argmin(self.fitness)
                self.f_opt = self.fitness[best_idx]
                self.x_opt = self.population[best_idx]
                self.last_improvement = self.evals
                self.best_fitness_history.append(self.f_opt)

            if self.evals >= self.budget:
                break

        return self.f_opt, self.x_opt
