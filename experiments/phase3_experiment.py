"""Phase 3 experiment: behavioural feature screening (neutral vs directional vs comparative feedback).

Reuses the Phase 1 infrastructure (shared initial population, (1+1)-ES, same
benchmark) but varies the feedback formatter per condition.

The module exposes:
  - run_condition()      — run all seeds for one condition
  - run_single_seed()    — run one (condition, seed) pair
  - main()               — CLI entry point
"""

import argparse
import os
import sys
import time
from pathlib import Path

from .feedback import (
    make_comparative_feature_feedback,
    make_directional_feature_feedback,
    make_single_feature_feedback,
)
from .initial_population import get_initial_solutions
from .mabbob_problem import MaBBOBProblem
from .phase1_experiment import (
    Phase1LLaMEA,
    make_llm,
    summarise_run,
    write_summary_csv,
)
from .phase3_config import (
    ALLOWED_IMPORTS,
    BBOB_BOUNDS,
    BUDGET_FACTOR,
    DIMS,
    ELITISM,
    EVAL_SEEDS,
    EVAL_TIMEOUT,
    LLAMEA_BUDGET,
    MODEL_CFG,
    MODEL_TAG,
    MUTATION_PROMPTS,
    N_OFFSPRING,
    N_PARENTS,
    RESULTS_DIR,
    RUN_SEEDS,
    TRAINING_INSTANCES,
    get_conditions,
)

from iohblade.experiment import Experiment
from iohblade.loggers import ExperimentLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_feedback_fn(condition_tag):
    """Return the feedback function for a given condition tag."""
    conditions = get_conditions()
    feature_name, fmt = conditions[condition_tag]
    if fmt == "neutral":
        return make_single_feature_feedback(feature_name)
    elif fmt == "directional":
        return make_directional_feature_feedback(feature_name)
    elif fmt == "comparative":
        return make_comparative_feature_feedback(feature_name)
    else:
        raise ValueError(f"Unknown feedback format: {fmt!r}")


def make_problem(condition_tag, use_worker_pool=True, eval_seeds=None,
                 training_instances=None, eval_timeout=None):
    """Create a MaBBOBProblem with the feedback formatter for this condition."""
    feedback_fn = make_feedback_fn(condition_tag)
    return MaBBOBProblem(
        make_feedback=feedback_fn,
        training_instances=training_instances or TRAINING_INSTANCES,
        eval_seeds=eval_seeds or EVAL_SEEDS,
        dims=DIMS,
        budget_factor=BUDGET_FACTOR,
        bbob_bounds=BBOB_BOUNDS,
        allowed_imports=ALLOWED_IMPORTS,
        use_worker_pool=use_worker_pool,
        eval_timeout=eval_timeout or EVAL_TIMEOUT,
    )


def make_method(condition_tag, llm, budget=None, initial_solutions=None):
    """Create a Phase1LLaMEA method for the given condition."""
    return Phase1LLaMEA(
        llm=llm,
        budget=budget or LLAMEA_BUDGET,
        name=condition_tag,
        initial_solutions=initial_solutions,
        n_parents=N_PARENTS,
        n_offspring=N_OFFSPRING,
        elitism=ELITISM,
        mutation_prompts=MUTATION_PROMPTS,
        HPO=False,
        feature_guided_mutation=False,
    )


# ---------------------------------------------------------------------------
# Run functions
# ---------------------------------------------------------------------------

def run_single_seed(
    condition_tag,
    seed,
    budget=None,
    eval_seeds=None,
    training_instances=None,
    eval_timeout=None,
    use_worker_pool=True,
    show_stdout=True,
    results_dir=None,
):
    """Run one (condition, seed) pair.

    Returns:
        str: path to the result directory.
    """
    results_dir = results_dir or RESULTS_DIR
    result_dir = f"{results_dir}/{condition_tag}/seed-{seed}"

    llm = make_llm(MODEL_CFG)
    problem = make_problem(
        condition_tag,
        use_worker_pool=use_worker_pool,
        eval_seeds=eval_seeds,
        training_instances=training_instances,
        eval_timeout=eval_timeout,
    )
    initial_solutions = get_initial_solutions()
    method = make_method(
        condition_tag, llm, budget=budget, initial_solutions=initial_solutions,
    )
    os.makedirs(result_dir, exist_ok=True)
    logger = ExperimentLogger(result_dir)

    experiment = Experiment(
        methods=[method],
        problems=[problem],
        budget=budget or LLAMEA_BUDGET,
        seeds=[seed],
        show_stdout=show_stdout,
        log_stdout=True,
        exp_logger=logger,
        n_jobs=1,
    )

    print(f"[Phase3] condition={condition_tag}  seed={seed}  model={MODEL_TAG}")
    print(f"  Budget: {budget or LLAMEA_BUDGET} candidates")
    print(f"  Strategy: ({N_PARENTS}+{N_OFFSPRING})-ES")
    print(f"  Eval: {len(training_instances or TRAINING_INSTANCES)} "
          f"instances x {eval_seeds or EVAL_SEEDS} seeds")
    print(f"  Results: {result_dir}")
    start = time.time()

    experiment()

    elapsed = time.time() - start
    print(f"  Completed seed {seed} in {elapsed/3600:.2f}h")
    return result_dir


def run_condition(
    condition_tag,
    seeds=None,
    budget=None,
    eval_seeds=None,
    training_instances=None,
    eval_timeout=None,
    use_worker_pool=True,
    show_stdout=True,
    results_dir=None,
):
    """Run all seeds for one condition sequentially.

    Returns:
        list of result directory paths.
    """
    seeds = seeds or RUN_SEEDS
    dirs = []
    for seed in seeds:
        d = run_single_seed(
            condition_tag=condition_tag,
            seed=seed,
            budget=budget,
            eval_seeds=eval_seeds,
            training_instances=training_instances,
            eval_timeout=eval_timeout,
            use_worker_pool=use_worker_pool,
            show_stdout=show_stdout,
            results_dir=results_dir,
        )
        dirs.append(d)
    return dirs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def list_conditions():
    """Print all available conditions."""
    conditions = get_conditions()
    print(f"Phase 3 conditions ({len(conditions)} total):")
    print(f"  Model: {MODEL_TAG} ({MODEL_CFG['model']})")
    print()
    for i, (tag, spec) in enumerate(conditions.items(), 1):
        if spec is None:
            print(f"  {i:2d}. {tag:50s}  [baseline]")
        else:
            feature, fmt = spec
            print(f"  {i:2d}. {tag:50s}  [{fmt}] {feature}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: behavioural feature screening (neutral vs directional vs comparative)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # List all conditions
  python run_phase3.py --list

  # Run vanilla baseline
  python run_phase3.py vanilla

  # Run one feature condition
  python run_phase3.py neutral-avg_improvement

  # Run all conditions
  python run_phase3.py all

  # Run all neutral conditions
  python run_phase3.py neutral

  # Run all directional conditions
  python run_phase3.py directional

  # Sanity check
  python run_phase3.py vanilla --sanity

  # Generate summary CSVs for existing results
  python run_phase3.py --summarise
""",
    )
    parser.add_argument(
        "conditions", nargs="*",
        help=(
            "Condition tag(s) to run. Use 'all' for every condition, "
            "'neutral' for all neutral, 'directional' for all directional, "
            "'comparative' for all comparative."
        ),
    )
    parser.add_argument("--list", action="store_true", help="List all conditions")
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help=f"Run seeds (default: {RUN_SEEDS})",
    )
    parser.add_argument(
        "--budget", type=int, default=None,
        help=f"Candidates per run (default: {LLAMEA_BUDGET})",
    )
    parser.add_argument(
        "--eval-seeds", type=int, default=None,
        help=f"Inner evaluation seeds per instance (default: {EVAL_SEEDS})",
    )
    parser.add_argument(
        "--eval-timeout", type=int, default=None,
        help=f"Max seconds per candidate evaluation (default: {EVAL_TIMEOUT})",
    )
    parser.add_argument(
        "--no-worker-pool", action="store_true",
        help="Disable persistent worker pool",
    )
    parser.add_argument(
        "--sanity", action="store_true",
        help="Sanity-check mode: 2 instances, 1 eval seed, 1 run seed, 10 candidates",
    )
    parser.add_argument(
        "--summarise", action="store_true",
        help="Only generate summary CSVs for existing results (no new runs)",
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help=f"Base results directory (default: {RESULTS_DIR})",
    )
    args = parser.parse_args()

    if args.list:
        list_conditions()
        return

    results_dir = args.results_dir or RESULTS_DIR

    if args.summarise:
        base = Path(results_dir)
        for cond_dir in sorted(base.iterdir()):
            if not cond_dir.is_dir():
                continue
            for seed_dir in sorted(cond_dir.glob("seed-*")):
                write_summary_csv(str(seed_dir))
        return

    if not args.conditions:
        parser.error("Provide condition tag(s), 'all', 'neutral', 'directional', 'comparative', or --list")

    all_conditions = get_conditions()

    # Apply sanity-check overrides
    training_instances = None
    eval_seeds = args.eval_seeds
    budget = args.budget
    seeds = args.seeds

    if args.sanity:
        training_instances = list(range(2))
        eval_seeds = eval_seeds or 1
        budget = budget or 10
        seeds = seeds or [0]
        print("[Phase3] SANITY MODE: 2 instances, 1 eval seed, 1 run seed, 10 candidates")

    # Resolve condition list
    requested = args.conditions
    if requested == ["all"]:
        condition_list = list(all_conditions.keys())
    elif requested == ["neutral"]:
        condition_list = [t for t in all_conditions if t.startswith("neutral-")]
    elif requested == ["directional"]:
        condition_list = [t for t in all_conditions if t.startswith("directional-")]
    elif requested == ["comparative"]:
        condition_list = [t for t in all_conditions if t.startswith("comparative-")]
    else:
        condition_list = requested

    # Validate
    for tag in condition_list:
        if tag not in all_conditions:
            print(f"ERROR: unknown condition '{tag}'", file=sys.stderr)
            print(f"Use --list to see available conditions.", file=sys.stderr)
            sys.exit(1)

    print(f"\nPhase 3: {len(condition_list)} condition(s) × "
          f"{len(seeds or RUN_SEEDS)} seed(s)")
    print(f"Model: {MODEL_TAG}\n")

    for tag in condition_list:
        spec = all_conditions[tag]
        label = "baseline" if spec is None else f"{spec[1]} / {spec[0]}"
        print(f"\n{'#'*60}")
        print(f"  CONDITION: {tag} ({label})")
        print(f"{'#'*60}")

        result_dirs = run_condition(
            condition_tag=tag,
            seeds=seeds or RUN_SEEDS,
            budget=budget,
            eval_seeds=eval_seeds,
            training_instances=training_instances,
            eval_timeout=args.eval_timeout,
            use_worker_pool=not args.no_worker_pool,
            show_stdout=True,
            results_dir=results_dir,
        )

        for d in result_dirs:
            write_summary_csv(d)

    print("\nAll conditions complete.")


if __name__ == "__main__":
    main()
