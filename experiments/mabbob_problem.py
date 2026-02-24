"""MaBBOBProblem: MA_BBOB subclass with behavioral metrics for thesis experiments.

Inherits the full MA-BBOB evaluation infrastructure (CSV loading, smoke test,
ManyAffine setup, AOCC scoring, prompts) from BLADE's MA_BBOB class. Adds:
- TrajectoryLogger for per-evaluation behavioral profiling
- compute_behavior_metrics for 11 behavioral features
- Pluggable feedback formatting via make_feedback()
- prepare_namespace() for safer code compilation
"""

import os
import random
from pathlib import Path

import numpy as np

from iohblade.problems.mabbob import MA_BBOB

_THESIS_ROOT = Path(__file__).resolve().parents[1]


class MaBBOBProblem(MA_BBOB):
    """Extend MA_BBOB with behavioral metrics collection and custom feedback."""

    def __init__(
        self,
        *,
        make_feedback,
        training_instances,
        dims=(5,),
        budget_factor=2000,
        bbob_bounds=None,
        allowed_imports=None,
        use_worker_pool=True,
        worker_recycle_interval=50,
        eval_timeout=6000,
    ):
        if bbob_bounds is None:
            bbob_bounds = [(-5.0, 5.0)]
        if allowed_imports is None:
            allowed_imports = ["numpy"]

        # MA_BBOB.__init__ handles: CSV loading (self.weights, self.iids,
        # self.opt_locs), prompts, dims, budget_factor, and calls
        # Problem.__init__ for subprocess infrastructure.
        super().__init__(
            training_instances=training_instances,
            dims=list(dims),
            budget_factor=budget_factor,
            eval_timeout=eval_timeout,
            dependencies=["ioh", "pandas", "scipy", "scikit-learn", "jsonlines", "configspace"],
        )

        # Worker pool settings — not accepted by MA_BBOB.__init__, set directly.
        self.use_worker_pool = use_worker_pool
        self.worker_recycle_interval = worker_recycle_interval
        self.extra_pythonpath = [
            str(_THESIS_ROOT),
            str(_THESIS_ROOT / "LLaMEA"),
            str(_THESIS_ROOT / "BLADE"),
        ]

        # Thesis-specific additions
        self.make_feedback = make_feedback
        self.bbob_bounds = bbob_bounds
        self.allowed_imports = allowed_imports

    def __call__(self, solution, logger=None):
        """Override to prevent LLaMEA's ExperimentLogger from being assigned
        to BLADE's logger slot (which expects .log_individual())."""
        return super().__call__(solution, logger=None)

    def evaluate(self, solution):
        """Run inside subprocess: compile, smoke-test, evaluate with behavioral metrics.

        Reuses parent's self.weights, self.iids, self.opt_locs (loaded by
        MA_BBOB.__init__) for MA-BBOB instance data. Adds TrajectoryLogger
        and compute_behavior_metrics on top of standard AOCC evaluation.
        """
        # Lazy imports — must resolve inside the venv subprocess
        import ioh
        from ioh import get_problem, logger as ioh_logger

        # Import llamea.utils directly to avoid triggering llamea/__init__.py
        # which pulls in lizard, networkx, etc.
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "llamea.utils",
            os.path.join(str(_THESIS_ROOT), "LLaMEA", "llamea", "utils.py"),
        )
        _llamea_utils = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_llamea_utils)
        prepare_namespace = _llamea_utils.prepare_namespace
        clean_local_namespace = _llamea_utils.clean_local_namespace

        from iohblade.utils import aoc_logger, correct_aoc, OverBudgetException
        from iohblade.behaviour_metrics import compute_behavior_metrics
        from experiments.trajectory_logger import TrajectoryLogger

        code = solution.code
        algorithm_name = solution.name
        possible_issue = None
        local_ns = {}

        # --- compile the candidate code ---
        try:
            global_ns, possible_issue = prepare_namespace(
                code, allowed=self.allowed_imports
            )
            exec(code, global_ns, local_ns)
            local_ns = clean_local_namespace(local_ns, global_ns)
        except Exception as e:
            feedback = str(e)
            if possible_issue:
                feedback = f"{possible_issue}. {feedback}"
            solution.set_scores(float("-inf"), feedback)
            return solution

        # --- smoke test on plain BBOB ---
        try:
            l_tmp = aoc_logger(100, upper=1e2, triggers=[ioh_logger.trigger.ALWAYS])
            prob_tmp = get_problem(11, 1, 2)
            prob_tmp.attach_logger(l_tmp)
            alg_tmp = local_ns[algorithm_name](budget=100, dim=2)
            alg_tmp(prob_tmp)
        except OverBudgetException:
            pass
        except Exception as e:
            solution.set_scores(float("-inf"), str(e))
            return solution

        # --- full MA-BBOB evaluation (reuses parent's CSV data) ---
        aucs = []
        all_metrics = []

        for dim in self.dims:
            budget = self.budget_factor * dim
            for idx in self.training_instances:
                f_new = ioh.problem.ManyAffine(
                    xopt=np.array(self.opt_locs.iloc[idx])[:dim],
                    weights=np.array(self.weights.iloc[idx]),
                    instances=np.array(self.iids.iloc[idx], dtype=int),
                    n_variables=dim,
                )
                f_new.set_id(100)
                f_new.set_instance(idx)

                l_aoc = aoc_logger(
                    budget, upper=1e2,
                    triggers=[ioh_logger.trigger.ALWAYS],
                )
                l_traj = TrajectoryLogger(
                    dim, triggers=[ioh_logger.trigger.ALWAYS],
                )
                combined = ioh_logger.Combine([l_aoc, l_traj])
                f_new.attach_logger(combined)

                random.seed(idx)
                np.random.seed(idx)
                try:
                    algorithm = local_ns[algorithm_name](budget=budget, dim=dim)
                    algorithm(f_new)
                except OverBudgetException:
                    pass
                except Exception as e:
                    solution.set_scores(float("-inf"), str(e))
                    return solution

                auc = correct_aoc(f_new, l_aoc, budget)
                aucs.append(auc)

                df = l_traj.to_dataframe()
                if len(df) > 1:
                    metrics = compute_behavior_metrics(
                        df, bounds=self.bbob_bounds * dim,
                    )
                    all_metrics.append(metrics)

                f_new.reset()

        auc_mean = np.mean(aucs)
        auc_std = np.std(aucs)
        avg_met = self._average_metrics(all_metrics)

        feedback = self.make_feedback(algorithm_name, auc_mean, auc_std, avg_met)
        solution.set_scores(auc_mean, feedback)
        solution.add_metadata("aucs", aucs)
        solution.add_metadata("behavioral_features", avg_met)

        return solution

    def test(self, solution):
        return self.evaluate(solution)

    def to_dict(self):
        return {
            "name": self.name,
            "eval_timeout": self.eval_timeout,
            "training_instances": self.training_instances,
            "dims": self.dims,
            "budget_factor": self.budget_factor,
            "bbob_bounds": self.bbob_bounds,
        }

    @staticmethod
    def _average_metrics(metrics_list):
        if not metrics_list:
            return {}
        keys = metrics_list[0].keys()
        return {k: np.mean([m[k] for m in metrics_list]) for k in keys}
