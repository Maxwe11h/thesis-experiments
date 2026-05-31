# condition: sage
# seed: 5
# AOCC: 0.9554
# name: CMA_EES

import numpy as np

class CMA_EES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        
        # Population sizing (slightly larger to support edge density)
        self.pop_size = 4 + int(4 * np.log(dim))
        if self.pop_size % 2 != 0:
            self.pop_size += 1
            
        self.mu = self.pop_size // 2
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights / np.sum(weights)
        self.mueff = 1.0 / np.sum(self.weights**2)
        
        # CMA-ES adaptation parameters
        self.cc = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.c1 = 2 / ((self.dim + 1.3)**2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.dim + 2)**2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs
        
        self.chi_n = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim**2))
        self.reset_state()

    def reset_state(self):
        self.x_mean = np.random.uniform(self.lb * 0.2, self.ub * 0.2, self.dim)
        self.sigma = 0.3 * (self.ub - self.lb)
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.C = np.eye(self.dim)
        self.gen = 0

    def _generate_edge_biased_samples(self, n_half):
        """Generates orthogonal samples with a bias toward the edges of the search space."""
        # QR for orthogonality
        q, _ = np.linalg.qr(np.random.normal(0, 1, (self.dim, self.dim)))
        indices = np.arange(n_half) % self.dim
        z_half = q[:, indices].T
        
        # Scaling mixture: standard normal + heavy-tailed Cauchy + Edge-Pushing Beta
        # The Beta distribution (0.5, 0.5) pushes points toward the extreme scales
        r_chi = np.sqrt(np.random.chisquare(self.dim, n_half))
        r_edge = np.random.beta(0.5, 0.5, n_half) * self.dim 
        scales = (0.8 * r_chi + 0.2 * r_edge).reshape(-1, 1)
        
        return z_half * scales

    def __call__(self, func):
        evals = 0
        best_f = float('inf')
        best_x = None
        
        while evals < self.budget:
            self.gen += 1
            self.C = (self.C + self.C.T) / 2
            try:
                vals, vecs = np.linalg.eigh(self.C)
                vals = np.clip(vals, 1e-16, 1e16)
                BD = vecs * np.sqrt(vals)
            except:
                BD = np.eye(self.dim)
            
            # Mirrored sampling for variance reduction and edge exploration
            z_half = self._generate_edge_biased_samples(self.pop_size // 2)
            z = np.vstack([z_half, -z_half])
            
            y = z @ BD.T
            x = self.x_mean + self.sigma * y
            
            # Edge-density handling: Reflective boundaries with momentum
            x_eval = np.copy(x)
            for i in range(self.pop_size):
                for j in range(self.dim):
                    if x_eval[i, j] < self.lb:
                        x_eval[i, j] = self.lb + (self.lb - x_eval[i, j]) % (self.ub - self.lb)
                    elif x_eval[i, j] > self.ub:
                        x_eval[i, j] = self.ub - (x_eval[i, j] - self.ub) % (self.ub - self.lb)
                    x_eval[i, j] = np.clip(x_eval[i, j], self.lb, self.ub)
            
            fitness = []
            for i in range(self.pop_size):
                if evals < self.budget:
                    f = func(x_eval[i])
                    evals += 1
                    fitness.append(f)
                    if f < best_f:
                        best_f, best_x = f, x_eval[i].copy()
                else:
                    fitness.append(float('inf'))
            
            if evals >= self.budget: break
            
            # Selection
            indices = np.argsort(fitness)
            selected = indices[:self.mu]
            old_mean = self.x_mean.copy()
            
            # Update mean using recombination weights
            self.x_mean = self.weights @ x_eval[selected]
            
            # Evolution paths
            inv_sqrt_C = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
            y_w = (self.x_mean - old_mean) / self.sigma
            z_w = y_w @ inv_sqrt_C.T
            
            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * z_w
            
            ps_norm = np.linalg.norm(self.ps)
            h_sig = ps_norm / np.sqrt(1 - (1 - self.cs)**(2 * evals / self.pop_size)) / self.chi_n < (1.4 + 2 / (self.dim + 1))
            
            self.pc = (1 - self.cc) * self.pc + h_sig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y_w
            
            # Covariance Matrix Update (Rank-1 and Rank-mu)
            y_diff = (x_eval[selected] - old_mean) / self.sigma
            self.C = ((1 - self.c1 - self.cmu) * self.C + 
                      self.c1 * (np.outer(self.pc, self.pc) + (1 - h_sig) * self.cc * (2 - self.cc) * self.C) + 
                      self.cmu * (y_diff.T * self.weights) @ y_diff)
            
            # Step size adaptation
            self.sigma *= np.exp((self.cs / self.damps) * (ps_norm / self.chi_n - 1))
            
            # Global restart if stagnated
            if self.sigma < 1e-12 or (self.gen > 100 and np.std(fitness[:self.mu]) < 1e-14):
                if evals < self.budget * 0.8:
                    self.reset_state()
                else:
                    break
                    
        return best_f, best_x