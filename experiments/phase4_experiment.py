"""Phase 4 experiment: full benchmark comparison of feedback conditions.

Compares vanilla, behavioural, SAGE, and combined feedback on 20 MA-BBOB
instances with 500 candidates per run (10 seeds each).

The module exposes:
  - run_condition()      — run all seeds for one condition
  - run_single_seed()    — run one (condition, seed) pair
  - is_seed_complete()   — check if a seed has already finished
  - main()               — CLI entry point
"""

import argparse
import os
import sys
import time
from pathlib import Path

from .feedback import make_multi_feature_neutral_feedback, vanilla_feedback
from .initial_population import get_initial_solutions
from .mabbob_problem import MaBBOBProblem
from .phase1_experiment import (
    Phase1LLaMEA,
    make_llm,
    summarise_run,
    write_summary_csv,
)
from .phase4_config import (
    ALLOWED_IMPORTS,
    BBOB_BOUNDS,
    BUDGET_FACTOR,
    CONDITIONS,
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
    NEUTRAL_FEATURES,
    RESULTS_DIR,
    RUN_SEEDS,
    TRAINING_INSTANCES,
)

from iohblade.experiment import Experiment
from iohblade.loggers import ExperimentLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_feedback_fn(condition_tag):
    """Return the feedback function for a given condition tag."""
    spec = CONDITIONS[condition_tag]
    if spec["feedback"] == "behavioural":
        return make_multi_feature_neutral_feedback(NEUTRAL_FEATURES)
    return vanilla_feedback


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
    spec = CONDITIONS[condition_tag]
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
        feature_guided_mutation=spec["sage"],
    )


def is_seed_complete(results_dir, condition_tag, seed, budget=None):
    """Check if a (condition, seed) run is already complete.

    A run is complete if its log.jsonl has >= budget lines.
    """
    budget = budget or LLAMEA_BUDGET
    seed_dir = Path(results_dir) / condition_tag / f"seed-{seed}"
    for log_file in seed_dir.glob("run-*/log.jsonl"):
        try:
            with open(log_file) as f:
                count = sum(1 for _ in f)
            if count >= budget:
                return True
        except OSError:
            pass
    return False


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

    spec = CONDITIONS[condition_tag]
    print(f"[Phase4] condition={condition_tag}  seed={seed}  model={MODEL_TAG}")
    print(f"  Feedback: {spec['feedback']}  SAGE: {spec['sage']}")
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
    skip_complete=False,
):
    """Run all seeds for one condition sequentially.

    Returns:
        list of result directory paths.
    """
    seeds = seeds or RUN_SEEDS
    results_dir = results_dir or RESULTS_DIR
    dirs = []
    for seed in seeds:
        if skip_complete and is_seed_complete(results_dir, condition_tag, seed, budget):
            print(f"  SKIP {condition_tag}/seed-{seed} (already complete)")
            continue
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
    print(f"Phase 4 conditions ({len(CONDITIONS)} total):")
    print(f"  Model: {MODEL_TAG} ({MODEL_CFG['model']})")
    print(f"  Budget: {LLAMEA_BUDGET} candidates")
    print(f"  Instances: {len(TRAINING_INSTANCES)}")
    print(f"  Seeds: {len(RUN_SEEDS)}")
    print()
    for i, (tag, spec) in enumerate(CONDITIONS.items(), 1):
        sage_str = "SAGE" if spec["sage"] else "    "
        print(f"  {i}. {tag:15s}  feedback={spec['feedback']:12s}  {sage_str}")
    print(f"\n  Neutral features: {', '.join(NEUTRAL_FEATURES)}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4: full benchmark comparison of feedback conditions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # List all conditions
  python run_phase4.py --list

  # Run one condition
  python run_phase4.py vanilla

  # Run all conditions
  python run_phase4.py all

  # Sanity check
  python run_phase4.py all --sanity

  # Skip already-completed seeds
  python run_phase4.py all --skip-complete

  # Generate summary CSVs for existing results
  python run_phase4.py --summarise
""",
    )
    parser.add_argument(
        "conditions", nargs="*",
        help="Condition tag(s) to run. Use 'all' for every condition.",
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
        "--skip-complete", action="store_true",
        help="Skip seeds that already have a complete log.jsonl",
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
        parser.error("Provide condition tag(s), 'all', or --list")

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
        print("[Phase4] SANITY MODE: 2 instances, 1 eval seed, 1 run seed, 10 candidates")

    # Resolve condition list
    requested = args.conditions
    if requested == ["all"]:
        condition_list = list(CONDITIONS.keys())
    else:
        condition_list = requested

    # Validate
    for tag in condition_list:
        if tag not in CONDITIONS:
            print(f"ERROR: unknown condition '{tag}'", file=sys.stderr)
            print(f"Valid: {', '.join(CONDITIONS)}", file=sys.stderr)
            sys.exit(1)

    print(f"\nPhase 4: {len(condition_list)} condition(s) x "
          f"{len(seeds or RUN_SEEDS)} seed(s)")
    print(f"Model: {MODEL_TAG}")
    print(f"Instances: {len(training_instances or TRAINING_INSTANCES)}")
    print(f"Budget: {budget or LLAMEA_BUDGET}\n")

    for tag in condition_list:
        spec = CONDITIONS[tag]
        sage_str = " + SAGE" if spec["sage"] else ""
        print(f"\n{'#'*60}")
        print(f"  CONDITION: {tag} ({spec['feedback']}{sage_str})")
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
            skip_complete=args.skip_complete,
        )

        for d in result_dirs:
            write_summary_csv(d)

    print("\nAll conditions complete.")


if __name__ == "__main__":
    main()
