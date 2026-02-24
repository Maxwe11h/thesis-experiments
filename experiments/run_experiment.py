"""Main runner: executes thesis experiments using BLADE's Experiment infrastructure.

Uses BLADE's Experiment for seed management, logging, progress display,
and problem lifecycle. Each condition is a (method, problem) pair run
across multiple seeds.
"""

from iohblade.experiment import Experiment
from iohblade.llm import Ollama_LLM
from iohblade.loggers import ExperimentLogger
from iohblade.methods import LLaMEA

from .config import (
    TRAINING_INSTANCES,
    DIMS,
    BUDGET_FACTOR,
    BBOB_BOUNDS,
    ALLOWED_IMPORTS,
    N_PARENTS,
    N_OFFSPRING,
    LLAMEA_BUDGET,
    OLLAMA_MODEL,
    SEEDS,
)
from .feedback import vanilla_feedback, behavioral_feedback
from .mabbob_problem import MaBBOBProblem


CONDITIONS = {
    "vanilla": {
        "make_feedback": vanilla_feedback,
        "feature_guided_mutation": False,
    },
    "behavioral": {
        "make_feedback": behavioral_feedback,
        "feature_guided_mutation": False,
    },
    "sage": {
        "make_feedback": vanilla_feedback,
        "feature_guided_mutation": True,
    },
    "combined": {
        "make_feedback": behavioral_feedback,
        "feature_guided_mutation": True,
    },
}


def make_problem(make_feedback, use_worker_pool=True):
    """Create a MaBBOBProblem with thesis config."""
    return MaBBOBProblem(
        make_feedback=make_feedback,
        training_instances=TRAINING_INSTANCES,
        dims=DIMS,
        budget_factor=BUDGET_FACTOR,
        bbob_bounds=BBOB_BOUNDS,
        allowed_imports=ALLOWED_IMPORTS,
        use_worker_pool=use_worker_pool,
    )


def make_method(name, feature_guided_mutation=False):
    """Create a BLADE LLaMEA method with thesis config."""
    llm = Ollama_LLM(model=OLLAMA_MODEL)
    return LLaMEA(
        llm=llm,
        budget=LLAMEA_BUDGET,
        name=name,
        n_parents=N_PARENTS,
        n_offspring=N_OFFSPRING,
        elitism=True,
        HPO=False,
        feature_guided_mutation=feature_guided_mutation,
    )


def run_condition(condition_name, seeds=None, show_stdout=True,
                  log_stdout=True, use_worker_pool=True):
    """Run a single condition across seeds using BLADE's Experiment.

    Results are saved to ``results/<condition_name>/`` with structured
    logging (experimentlog.jsonl, per-run directories with log.jsonl,
    conversationlog.jsonl, progress.json).
    """
    if seeds is None:
        seeds = list(SEEDS)
    cfg = CONDITIONS[condition_name]

    problem = make_problem(cfg["make_feedback"], use_worker_pool=use_worker_pool)
    method = make_method(condition_name, cfg["feature_guided_mutation"])
    logger = ExperimentLogger(f"results/{condition_name}")

    experiment = Experiment(
        methods=[method],
        problems=[problem],
        budget=LLAMEA_BUDGET,
        seeds=seeds,
        show_stdout=show_stdout,
        log_stdout=log_stdout,
        exp_logger=logger,
        n_jobs=1,
    )
    experiment()
    return logger


def run_full_experiment(seeds=None, show_stdout=True, use_worker_pool=True):
    """Run all 4 conditions sequentially."""
    for name in CONDITIONS:
        print(f"\n{'#'*60}")
        print(f"  CONDITION: {name}")
        print(f"{'#'*60}")
        run_condition(name, seeds, show_stdout, use_worker_pool=use_worker_pool)


def main():
    run_full_experiment()


if __name__ == "__main__":
    main()
