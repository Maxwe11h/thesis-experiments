"""Run a single (algorithm, dim, instance-batch) shard of the §5.4.6 sweep.

The runner deliberately uses no LLM. It compiles the saved winner code,
instantiates it, and evaluates each instance with the BBOB convention. CMA-ES
is supplied by `cma` (pycma) at the same budget for direct comparison.

Output: parquet under <RESULTS_DIR>/<alg>/dim<d>/instances_<batch>.parquet.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from experiments.phase4_full_suite_config import (
    ALGORITHMS,
    BUDGET_FACTOR,
    EVAL_SEEDS,
    EVAL_TIMEOUT,
    RESULTS_DIR,
)


def _load_user_algo(path: str):
    """exec the saved winner-code file and return the leaf class object."""
    src = Path(path).read_text()
    ns: dict = {}
    exec(src, ns, ns)
    classes = [v for v in ns.values() if isinstance(v, type)]
    if not classes:
        raise RuntimeError(f'no class defined in {path}')
    return classes[-1]   # the last-defined class is the algorithm


def _aocc(curve: np.ndarray, budget: int, lb: float = 1e-8, ub: float = 1e2) -> float:
    """Per-run AOCC, identical to thesis §4.5 eq. (4.4)."""
    log_curve = np.log10(np.clip(curve, lb, ub))
    log_lb, log_ub = np.log10(lb), np.log10(ub)
    return float(np.mean(1.0 - (log_curve - log_lb) / (log_ub - log_lb)))


_MA_BBOB_DATA = None  # cached MA_BBOB instance for instance-data lookups


def _ensure_ma_bbob_data():
    """Lazily instantiate MA_BBOB once to load self.weights, self.iids, self.opt_locs."""
    global _MA_BBOB_DATA
    if _MA_BBOB_DATA is None:
        from iohblade.benchmarks.BBOB.mabbob import MA_BBOB
        # We never call this instance — only read its weight/iid/opt_locs tables.
        _MA_BBOB_DATA = MA_BBOB(training_instances=[0], dims=[5], budget_factor=BUDGET_FACTOR)
    return _MA_BBOB_DATA


def _run_once(algo_factory, dim: int, instance_idx: int, seed: int) -> float:
    """Drive one (instance, seed) optimisation run and return its per-run AOCC."""
    import ioh  # imported lazily so unit tests don't require it

    np.random.seed(seed)

    # Pull the MA-BBOB instance the same way as `MaBBOBProblem.evaluate`.
    data = _ensure_ma_bbob_data()
    f_new = ioh.problem.ManyAffine(
        xopt=np.array(data.opt_locs.iloc[instance_idx])[:dim],
        weights=np.array(data.weights.iloc[instance_idx]),
        instances=np.array(data.iids.iloc[instance_idx], dtype=int),
        n_variables=dim,
    )
    f_new.set_id(100)
    f_new.set_instance(instance_idx)

    budget = BUDGET_FACTOR * dim
    curve = np.full(budget, np.inf)
    pos = [0]

    def wrapped(x):
        if pos[0] >= budget:
            return 1e30
        y = float(f_new(x))
        curve[pos[0]] = min(y, curve[pos[0] - 1] if pos[0] > 0 else y)
        pos[0] += 1
        return y

    algo = algo_factory(budget=budget, dim=dim)
    try:
        algo(wrapped)
    except Exception:
        # Score the run with whatever curve we have so far.
        pass

    # Pad if the algorithm exited early.
    if pos[0] < budget:
        curve[pos[0]:] = curve[pos[0] - 1] if pos[0] > 0 else 1e2

    return _aocc(curve, budget)


class _CMAESWrapper:
    def __init__(self, budget: int, dim: int):
        self._budget = budget
        self._dim = dim

    def __call__(self, func):
        import cma
        es = cma.CMAEvolutionStrategy(
            np.zeros(self._dim), 1.0,
            {'bounds': [-5.0, 5.0], 'maxfevals': self._budget,
             'verbose': -9},
        )
        while not es.stop():
            xs = es.ask()
            ys = [func(x) for x in xs]
            es.tell(xs, ys)


def _factory(name: str):
    spec = ALGORITHMS[name]
    if spec.startswith('BUILTIN:cma_es'):
        return _CMAESWrapper
    cls = _load_user_algo(spec)
    return cls


def run_shard(alg_name: str, dim: int, instance_indices: Iterable[int],
              out_dir: Path) -> Path:
    factory = _factory(alg_name)
    instance_indices = list(instance_indices)
    rows = []
    t0 = time.monotonic()
    for idx in instance_indices:
        for seed in range(EVAL_SEEDS):
            if time.monotonic() - t0 > EVAL_TIMEOUT:
                raise TimeoutError(
                    f'shard {alg_name} dim={dim} timed out at instance={idx}'
                )
            score = _run_once(factory, dim, idx, seed)
            rows.append({'algorithm': alg_name, 'dim': dim, 'instance': idx,
                         'eval_seed': seed, 'aocc': score})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        out_dir
        / f'{alg_name}_dim{dim}_inst{min(instance_indices)}-{max(instance_indices)}.parquet'
    )
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    return out_path
