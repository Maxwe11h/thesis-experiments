# condition: neutral
# seed: 0
# AOCC: 0.9545
# name: BIPOP_MA_CMA_ES

import numpy as np

class BIPOP_MA_CMA_ES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.eval_count = 0
        
        # Global best tracking
        self.f_opt = np.inf
        self.x_opt = None
        
        # BIPOP Logic
        self.base_pop_size = int(4 + np.floor(3 * np.log(dim)))
        self.pop_size = self.base_pop_size
        self.n_restarts = 0
        self.last_restart_eval = 0
        
        # Memory-augmented parameters
        self.success_history = []
        self.reset_parameters()

    def reset_parameters(self):
        if self.pop_size % 2 != 0: self.pop_size += 1
        self.mu = self.pop_size // 2
        
        # Log-linear weights with sharper selection pressure
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights / np.sum(weights)
        self.mueff = (np.sum(self.weights)**2) / np.sum(self.weights**2)
        
        # Adaptation constants (standard CMA-ES defaults with slight tuning)
        self.cc = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.c1 = 2 / ((self.dim + 1.3)**2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1/self.mueff) / ((self.dim + 2)**2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs
        self.chiN = self.dim**0.5 * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))
        
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.C = np.eye(self.dim)
        
        # BIPOP initialization logic
        if self.n_restarts > 0:
            if self.n_restarts % 2 != 0: # Small-pop local search
                self.mean = np.clip(self.x_opt + 0.1 * self.sigma_last * np.random.randn(self.dim), self.lb, self.ub)
                self.sigma = 0.01 * (self.ub - self.lb)
            else: # Large-pop global search
                self.mean = np.random.uniform(self.lb, self.ub, self.dim)
                self.sigma = 0.3 * (self.ub - self.lb)
        else:
            self.mean = np.zeros(self.dim)
            self.sigma = 0.2 * (self.ub - self.lb)
        
        self.sigma_last = self.sigma
        self.hist_best = []

    def repair(self, x):
        """Saturation with reflection-like behavior to keep points in bounds."""
        for i in range(self.dim):
            if x[i] < self.lb:
                x[i] = self.lb + np.random.rand() * 1e-6
            elif x[i] > self.ub:
                x[i] = self.ub - np.random.rand() * 1e-6
        return x

    def __call__(self, func):
        while self.eval_count < self.budget:
            try:
                # Eigen-decomposition for sampling
                self.C = (self.C + self.C.T) / 2
                vals, vecs = np.linalg.eigh(self.C)
                vals = np.maximum(vals, 1e-18)
                BD = vecs @ np.diag(np.sqrt(vals))
            except (np.linalg.LinAlgError, ValueError):
                self.reset_parameters()
                continue
            
            pop = []
            # Orthogonal Mirrored Sampling
            for i in range(self.pop_size // 2):
                if self.eval_count >= self.budget: break
                z = np.random.standard_normal(self.dim)
                # Ensure local orthogonality for small dimensions to improve coverage
                if self.dim < 30 and i > 0:
                    for prev in [p['z'] for p in pop[::2]]:
                        z -= np.dot(z, prev) * prev / (np.linalg.norm(prev)**2 + 1e-10)
                    z = (z / (np.linalg.norm(z) + 1e-10)) * np.sqrt(self.dim)

                for mult in [1.0, -1.0]:
                    if self.eval_count >= self.budget: break
                    y = BD @ (mult * z)
                    x = self.mean + self.sigma * y
                    x_rep = self.repair(x.copy())
                    f = func(x_rep)
                    self.eval_count += 1
                    pop.append({'x': x_rep, 'y': y, 'z': mult * z, 'f': f})
                    
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = x_rep.copy()

            if not pop: break
            pop.sort(key=lambda x: x['f'])
            self.hist_best.append(pop[0]['f'])
            
            # Mean update
            old_mean = self.mean.copy()
            self.mean = np.zeros(self.dim)
            for i in range(self.mu):
                self.mean += self.weights[i] * pop[i]['x']
            
            # Paths and Covariance Update
            invsqrtC = vecs @ np.diag(1/np.sqrt(vals)) @ vecs.T
            z_mean = (invsqrtC @ (self.mean - old_mean)) / self.sigma
            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * z_mean
            
            hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * (self.eval_count - self.last_restart_eval) / self.pop_size + 1)) / self.chiN < (1.4 + 2 / (self.dim + 1))
            self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * ((self.mean - old_mean) / self.sigma)
            
            # Active CMA: Positive and Negative Updates
            pos_update = np.zeros((self.dim, self.dim))
            for i in range(self.mu):
                pos_update += self.weights[i] * np.outer(pop[i]['y'], pop[i]['y'])
            
            # Negative update (Active CMA) from worst samples
            neg_update = np.zeros((self.dim, self.dim))
            for i in range(self.mu):
                neg_update += self.weights[i] * np.outer(pop[-(i+1)]['y'], pop[-(i+1)]['y'])
            
            cmu_neg = min(1 - self.c1 - self.cmu, 0.5 * self.cmu) # Constraint-based scaling
            self.C = (1 - self.c1 - self.cmu + cmu_neg) * self.C \
                     + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C) \
                     + self.cmu * pos_update - cmu_neg * neg_update
            
            # Step size adaptation
            self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
            
            # Restart triggers
            stagnation = len(self.hist_best) > 20 and self.hist_best[-1] >= self.hist_best[-20] * (1 - 1e-13)
            cond = np.max(vals) / np.min(vals) > 1e13
            
            if self.sigma < 1e-11 or stagnation or cond:
                self.sigma_last = self.sigma
                self.n_restarts += 1
                self.last_restart_eval = self.eval_count
                if self.n_restarts % 2 == 0:
                    self.pop_size = min(int(self.pop_size * 2), 1024)
                else:
                    self.pop_size = self.base_pop_size
                self.reset_parameters()
                
        return self.f_opt, self.x_opt