"""Model selection experiment: compare SLMs on vanilla LLaMEA to pick the best
model for the full feasibility study.

Each model runs a vanilla condition (AOCC-only feedback) for a configurable
number of candidates. We track:
- Failure rate and failure categories
- AOCC of successful candidates (best, mean)
- LLM inference time vs evaluation time
- Per-candidate timing breakdown

Results are saved to results_model_selection/<model_tag>/.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from iohblade.experiment import Experiment
from iohblade.llm import Ollama_LLM
from iohblade.loggers import ExperimentLogger
from iohblade.methods import LLaMEA

from .config import (
    TRAINING_INSTANCES,
    EVAL_SEEDS,
    DIMS,
    BUDGET_FACTOR,
    BBOB_BOUNDS,
    ALLOWED_IMPORTS,
    OLLAMA_PORT,
)
from .feedback import vanilla_feedback
from .mabbob_problem import MaBBOBProblem

# Models to compare (tag -> ollama model name)
# Sizes are approximate parameter counts for reference.
CANDIDATE_MODELS = {
    "qwen3-8b":          "qwen3:8b",
    "qwen3-coder-30b":   "qwen3-coder:30b",
}

# Default budget for model selection runs
MODEL_SELECTION_BUDGET = 100


def make_problem(use_worker_pool=True, eval_seeds=None, training_instances=None):
    """Create vanilla MaBBOBProblem with optional reduced config."""
    return MaBBOBProblem(
        make_feedback=vanilla_feedback,
        training_instances=training_instances or TRAINING_INSTANCES,
        eval_seeds=eval_seeds or EVAL_SEEDS,
        dims=DIMS,
        budget_factor=BUDGET_FACTOR,
        bbob_bounds=BBOB_BOUNDS,
        allowed_imports=ALLOWED_IMPORTS,
        use_worker_pool=use_worker_pool,
    )


def make_method(model_tag, ollama_model, budget, port=None):
    """Create a LLaMEA method configured for a specific Ollama model."""
    llm = Ollama_LLM(model=ollama_model, port=port or OLLAMA_PORT)
    return LLaMEA(
        llm=llm,
        budget=budget,
        name=model_tag,
        n_parents=1,
        n_offspring=1,
        elitism=True,
        HPO=False,
        feature_guided_mutation=False,
    )


def run_model(
    model_tag,
    ollama_model=None,
    budget=None,
    port=None,
    eval_seeds=None,
    training_instances=None,
    use_worker_pool=True,
    show_stdout=True,
):
    """Run vanilla condition for a single model.

    Returns path to result directory.
    """
    if ollama_model is None:
        ollama_model = CANDIDATE_MODELS[model_tag]
    if budget is None:
        budget = MODEL_SELECTION_BUDGET

    result_dir = f"results_model_selection/{model_tag}"
    problem = make_problem(
        use_worker_pool=use_worker_pool,
        eval_seeds=eval_seeds,
        training_instances=training_instances,
    )
    method = make_method(model_tag, ollama_model, budget, port)
    logger = ExperimentLogger(result_dir)

    experiment = Experiment(
        methods=[method],
        problems=[problem],
        budget=budget,
        seeds=[0],
        show_stdout=show_stdout,
        log_stdout=True,
        exp_logger=logger,
        n_jobs=1,
    )

    print(f"Starting model: {model_tag} ({ollama_model})")
    print(f"  Budget: {budget} candidates")
    print(f"  Eval config: {len(training_instances or TRAINING_INSTANCES)} instances x "
          f"{eval_seeds or EVAL_SEEDS} seeds")
    print(f"  Ollama port: {port or OLLAMA_PORT}")
    start = time.time()
    experiment()
    elapsed = time.time() - start
    print(f"  Completed in {elapsed/3600:.2f}h")

    return result_dir


def list_models():
    """Print all candidate models."""
    print("Available models for comparison:")
    for i, (tag, model) in enumerate(CANDIDATE_MODELS.items(), 1):
        print(f"  {i:2d}. {tag:25s} -> {model}")


def main():
    parser = argparse.ArgumentParser(
        description="Model selection experiment: compare SLMs on vanilla LLaMEA"
    )
    parser.add_argument(
        "models", nargs="*",
        help="Model tags to run (see --list for options). "
             "Use 'all' to run all models sequentially.",
    )
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument(
        "--budget", type=int, default=MODEL_SELECTION_BUDGET,
        help=f"Candidates per model (default: {MODEL_SELECTION_BUDGET})",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Ollama port (default: from OLLAMA_PORT env or 11434)",
    )
    parser.add_argument(
        "--eval-seeds", type=int, default=None,
        help="Seeds per instance (default: from config, currently 5)",
    )
    parser.add_argument(
        "--training-instances", type=int, default=None,
        help="Number of training instances (default: from config, currently 10)",
    )
    parser.add_argument(
        "--no-worker-pool", action="store_true",
        help="Disable worker pool (run evaluations in-process)",
    )
    parser.add_argument(
        "--custom-model", type=str, default=None,
        help="Run a custom Ollama model not in the predefined list "
             "(use with a single model tag argument as the label)",
    )
    args = parser.parse_args()

    if args.list:
        list_models()
        return

    if not args.models:
        parser.error("provide model tag(s) or --list")

    training_instances = None
    if args.training_instances is not None:
        training_instances = list(range(args.training_instances))

    # Handle 'all'
    if args.models == ["all"]:
        model_list = list(CANDIDATE_MODELS.keys())
    else:
        model_list = args.models

    for tag in model_list:
        if args.custom_model and len(model_list) == 1:
            ollama_model = args.custom_model
        elif tag in CANDIDATE_MODELS:
            ollama_model = CANDIDATE_MODELS[tag]
        else:
            print(f"ERROR: unknown model tag '{tag}'", file=sys.stderr)
            print(f"Valid tags: {', '.join(CANDIDATE_MODELS)}", file=sys.stderr)
            print("Or use --custom-model to specify an arbitrary Ollama model", file=sys.stderr)
            sys.exit(1)

        print(f"\n{'#'*60}")
        print(f"  MODEL: {tag} ({ollama_model})")
        print(f"{'#'*60}")

        run_model(
            model_tag=tag,
            ollama_model=ollama_model,
            budget=args.budget,
            port=args.port,
            eval_seeds=args.eval_seeds,
            training_instances=training_instances,
            use_worker_pool=not args.no_worker_pool,
        )

    print("\nAll models complete.")


if __name__ == "__main__":
    main()
