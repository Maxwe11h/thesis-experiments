"""NADE harness-adapter contract tests (NumPy-only; no ioh needed).

Verifies the adapted NeighborhoodAdaptiveDE conforms to the full-suite harness
interface: cls(budget, dim) + __call__(func), draws from the global RNG (so the
harness per-run seed governs), respects the budget, and runs on NumPy 2.x.
"""
from pathlib import Path

import numpy as np

ADAPTED = Path(__file__).resolve().parents[1] / "baselines" / "neighborhood_adaptive_de.py"


def _load():
    src = ADAPTED.read_text()
    ns: dict = {}
    exec(compile(src, str(ADAPTED), "exec"), ns)
    classes = [v for v in ns.values() if isinstance(v, type)]
    return classes[-1]


def _counting_quadratic(budget):
    calls = [0]

    def f(x):
        calls[0] += 1
        return float(np.sum(np.asarray(x, dtype=float) ** 2))

    return f, calls


def test_runs_and_returns_finite_within_budget():
    NADE = _load()
    budget, dim = 600, 5
    f, calls = _counting_quadratic(budget)
    np.random.seed(0)
    algo = NADE(budget=budget, dim=dim)
    f_opt, x_opt = algo(f)
    assert np.isfinite(f_opt)
    assert x_opt is not None and len(x_opt) == dim
    # NADE evaluates whole generations; allow one extra generation of slack.
    assert calls[0] <= budget + algo.pop_size


def test_harness_seed_governs_so_seeds_differ():
    NADE = _load()

    def run(seed):
        f, _ = _counting_quadratic(600)
        np.random.seed(seed)
        return float(NADE(budget=600, dim=5)(f)[0])

    # No internal reseed -> different harness seeds give different runs.
    assert run(0) != run(1)


def test_no_np_inf_attribute_used():
    # np.Inf was removed in NumPy 2.0; the *code* must not reference it
    # (a comment documenting the np.Inf -> np.inf change is fine, so we parse
    # the AST rather than grep the raw text).
    import ast

    tree = ast.parse(ADAPTED.read_text())
    bad = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Attribute)
        and n.attr == "Inf"
        and isinstance(n.value, ast.Name)
        and n.value.id == "np"
    ]
    assert not bad, "code references np.Inf (removed in NumPy 2.0)"
