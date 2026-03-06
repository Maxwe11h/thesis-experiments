"""Phase 1 experiment: screen LLMs on MA-BBOB with (1+1)-ES and shared initial algorithm.

Builds on the existing BLADE + LLaMEA infrastructure. Key differences from
the feature-selection experiment:
  - 5 independent runs per model (different BLADE seeds)
  - Shared initial algorithm (RandomSearch) injected before the evolutionary
    loop so all models start from the same baseline
  - Support for both Ollama (local) and Gemini (API) models
  - 90% refinement / 10% exploration mutation mix

The module exposes:
  - run_model()        — run all seeds for one model
  - run_single_seed()  — run one (model, seed) pair
  - summarise_run()    — extract summary CSV from a finished run directory
  - main()             — CLI entry point
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from iohblade.experiment import Experiment
from iohblade.llm import Gemini_LLM, Ollama_LLM, VLLM_LLM
from iohblade.loggers import ExperimentLogger
from iohblade.methods import LLaMEA as BladeLLaMEA
from llamea import LLaMEA as LLaMEA_Algorithm
from iohblade.solution import Solution

from .feedback import vanilla_feedback
from .initial_population import get_initial_solutions
from .mabbob_problem import MaBBOBProblem
from .phase1_config import (
    ALLOWED_IMPORTS,
    BBOB_BOUNDS,
    BUDGET_FACTOR,
    CANDIDATE_MODELS,
    DIMS,
    ELITISM,
    EVAL_SEEDS,
    EVAL_TIMEOUT,
    LLAMEA_BUDGET,
    MUTATION_PROMPTS,
    N_OFFSPRING,
    N_PARENTS,
    OLLAMA_PORT,
    RESULTS_DIR,
    RUN_SEEDS,
    TRAINING_INSTANCES,
    VLLM_BASE_URL,
)


# ---------------------------------------------------------------------------
# Custom method: LLaMEA with injected initial population
# ---------------------------------------------------------------------------

class Phase1LLaMEA(BladeLLaMEA):
    """BLADE LLaMEA wrapper that injects a fixed initial algorithm.

    Instead of letting LLaMEA generate the initial solution via the LLM,
    we pre-evaluate a known algorithm (RandomSearch) and inject it so every
    model and seed starts from the same baseline.
    """

    def __init__(self, llm, budget, name, initial_solutions=None, **kwargs):
        super().__init__(llm, budget, name, **kwargs)
        # List of unevaluated Solution objects to use as starting population
        self._initial_solutions = initial_solutions or []

    def __call__(self, problem):
        """Create the LLaMEA instance, inject initial population, then run."""
        self.llamea_instance = LLaMEA_Algorithm(
            f=problem,
            llm=self.llm,
            role_prompt="You are an excellent Python programmer.",
            task_prompt=problem.task_prompt,
            example_prompt=problem.example_prompt,
            output_format_prompt=problem.format_prompt,
            log=None,   # BLADE handles logging, not LLaMEA's native logger
            budget=self.budget,
            max_workers=1,  # no parallelisation inside LLaMEA (BLADE manages it)
            **self.kwargs,
        )

        if self._initial_solutions:
            # Evaluate each initial solution through the problem's full pipeline
            # (compile → smoke test → MA-BBOB eval, all in subprocess).
            evaluated = []
            for sol in self._initial_solutions:
                # Deep-copy so original templates are not mutated
                import copy
                s = copy.deepcopy(sol)
                s.task_prompt = problem.task_prompt
                s.generation = 0
                s = problem(s)  # full evaluation via BLADE subprocess
                if math.isnan(s.fitness):
                    s.fitness = -np.inf
                evaluated.append(s)

            # Inject into the LLaMEA instance; initialize() will skip LLM
            # generation because len(population) == n_parents already.
            self.llamea_instance.population = evaluated

        return self.llamea_instance.run()


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_llm(model_cfg, port=None, base_url=None):
    """Instantiate an LLM object from a model registry entry.

    Args:
        model_cfg: dict with 'type' and 'model' keys (from CANDIDATE_MODELS).
        port: Ollama port override (ignored for API models).
        base_url: vLLM base URL override (ignored for non-vllm models).

    Returns:
        An LLM instance (Ollama_LLM, VLLM_LLM, or Gemini_LLM).
    """
    mtype = model_cfg["type"]
    model = model_cfg["model"]

    if mtype == "ollama":
        return Ollama_LLM(model=model, port=port or OLLAMA_PORT)

    if mtype == "vllm":
        return VLLM_LLM(model=model, base_url=base_url or VLLM_BASE_URL)

    if mtype == "gemini":
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Gemini models require GOOGLE_API_KEY or GEMINI_API_KEY env var."
            )
        return Gemini_LLM(api_key=api_key, model=model)

    raise ValueError(f"Unknown model type: {mtype!r}")


def make_problem(use_worker_pool=True, eval_seeds=None, training_instances=None, eval_timeout=None):
    """Create a vanilla MaBBOBProblem with Phase 1 evaluation config."""
    return MaBBOBProblem(
        make_feedback=vanilla_feedback,
        training_instances=training_instances if training_instances is not None else TRAINING_INSTANCES,
        eval_seeds=eval_seeds or EVAL_SEEDS,
        dims=DIMS,
        budget_factor=BUDGET_FACTOR,
        bbob_bounds=BBOB_BOUNDS,
        allowed_imports=ALLOWED_IMPORTS,
        use_worker_pool=use_worker_pool,
        eval_timeout=eval_timeout or EVAL_TIMEOUT,
    )


def make_method(model_tag, llm, budget=None, initial_solutions=None):
    """Create a Phase1LLaMEA method configured for the (1+1) strategy."""
    return Phase1LLaMEA(
        llm=llm,
        budget=budget or LLAMEA_BUDGET,
        name=model_tag,
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
    model_tag,
    seed,
    model_cfg=None,
    budget=None,
    port=None,
    base_url=None,
    eval_seeds=None,
    training_instances=None,
    eval_timeout=None,
    use_worker_pool=True,
    show_stdout=True,
    results_dir=None,
):
    """Run one (model, seed) pair.

    Returns:
        str: path to the result directory.
    """
    if model_cfg is None:
        model_cfg = CANDIDATE_MODELS[model_tag]
    if results_dir is None:
        results_dir = RESULTS_DIR

    result_dir = f"{results_dir}/{model_tag}/seed-{seed}"

    llm = make_llm(model_cfg, port=port, base_url=base_url)
    problem = make_problem(
        use_worker_pool=use_worker_pool,
        eval_seeds=eval_seeds,
        training_instances=training_instances,
        eval_timeout=eval_timeout,
    )
    initial_solutions = get_initial_solutions()
    method = make_method(
        model_tag, llm, budget=budget, initial_solutions=initial_solutions,
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

    print(f"[Phase1] model={model_tag}  seed={seed}  "
          f"type={model_cfg['type']}  backend={model_cfg['model']}")
    print(f"  Budget: {budget or LLAMEA_BUDGET} candidates")
    print(f"  Strategy: ({N_PARENTS}+{N_OFFSPRING})-ES")
    print(f"  Eval: {len(training_instances) if training_instances is not None else len(TRAINING_INSTANCES)} "
          f"instances x {eval_seeds or EVAL_SEEDS} seeds")
    print(f"  Results: {result_dir}")
    start = time.time()

    experiment()

    elapsed = time.time() - start
    print(f"  Completed seed {seed} in {elapsed/3600:.2f}h")
    return result_dir


def run_model(
    model_tag,
    model_cfg=None,
    seeds=None,
    budget=None,
    port=None,
    base_url=None,
    eval_seeds=None,
    training_instances=None,
    eval_timeout=None,
    use_worker_pool=True,
    show_stdout=True,
    results_dir=None,
):
    """Run all seeds for one model sequentially.

    Returns:
        list of result directory paths.
    """
    if seeds is None:
        seeds = RUN_SEEDS
    dirs = []
    for seed in seeds:
        d = run_single_seed(
            model_tag=model_tag,
            seed=seed,
            model_cfg=model_cfg,
            budget=budget,
            port=port,
            base_url=base_url,
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
# Summary extraction
# ---------------------------------------------------------------------------

def summarise_run(result_dir):
    """Extract a summary from a finished run's log.jsonl.

    Scans the result directory for the run sub-directory containing log.jsonl,
    then extracts per-candidate records.

    Returns:
        list of dicts, one per candidate evaluated.
    """
    result_path = Path(result_dir)
    records = []

    # Find run sub-directories (pattern: run-{method}-{problem}-{seed}/)
    run_dirs = sorted(result_path.glob("run-*/"))
    if not run_dirs:
        return records

    for run_dir in run_dirs:
        log_file = run_dir / "log.jsonl"
        if not log_file.exists():
            continue

        # Parse the directory name: run-{method_tag}-{problem}-{seed}
        parts = run_dir.name.split("-")
        # Seed is the last part
        dir_seed = parts[-1] if parts else "?"

        with open(log_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                try:
                    fitness = float(entry.get("fitness", "nan"))
                except (TypeError, ValueError):
                    fitness = float("nan")
                status = "success" if not math.isinf(fitness) and not math.isnan(fitness) else "failure"
                aucs = entry.get("metadata", {}).get("aucs", [])
                behavioral = entry.get("metadata", {}).get("behavioral_features", {})

                records.append({
                    "model_name": parts[1] if len(parts) > 1 else "unknown",
                    "seed": dir_seed,
                    "generation": entry.get("generation", "?"),
                    "algorithm_name": entry.get("name", ""),
                    "AOCC": fitness if status == "success" else "",
                    "final_best_value": fitness,
                    "run_status": status,
                    "error": entry.get("error", ""),
                    "n_aucs": len(aucs),
                    **{f"bm_{k}": v for k, v in behavioral.items()},
                })

    return records


def write_summary_csv(result_dir, output_path=None):
    """Write a summary CSV for one run directory.

    Args:
        result_dir: path to the seed-level result directory.
        output_path: where to write the CSV; defaults to {result_dir}/summary.csv.

    Returns:
        str: path to the written CSV.
    """
    records = summarise_run(result_dir)
    if not records:
        print(f"  WARNING: no records found in {result_dir}")
        return None

    if output_path is None:
        output_path = str(Path(result_dir) / "summary.csv")

    fieldnames = list(records[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"  Summary CSV: {output_path} ({len(records)} records)")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def list_models():
    """Print all candidate models."""
    print("Phase 1 candidate models:")
    for i, (tag, cfg) in enumerate(CANDIDATE_MODELS.items(), 1):
        print(f"  {i:2d}. {tag:30s} [{cfg['type']:6s}]  {cfg['model']}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: LLM screening on MA-BBOB with (1+1)-ES",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # List available models
  python run_phase1.py --list

  # Run all 5 seeds for one model
  python run_phase1.py qwen3.5-4b

  # Run specific seeds
  python run_phase1.py qwen3.5-4b --seeds 0 1

  # Run all models
  python run_phase1.py all

  # Sanity check (2 instances, 1 eval seed, 1 run seed, 10 candidates)
  python run_phase1.py qwen3.5-4b --sanity

  # Custom Ollama model
  python run_phase1.py my-model --custom-ollama "mistral:7b"

  # Gemini API model
  GOOGLE_API_KEY=... python run_phase1.py gemini-3-pro
""",
    )
    parser.add_argument(
        "models", nargs="*",
        help="Model tag(s) to run. Use 'all' for every model in the registry.",
    )
    parser.add_argument("--list", action="store_true", help="List registered models")
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help=f"Run seeds (default: {RUN_SEEDS})",
    )
    parser.add_argument(
        "--budget", type=int, default=None,
        help=f"Candidates per run (default: {LLAMEA_BUDGET})",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Ollama port (default: OLLAMA_PORT env or 11434)",
    )
    parser.add_argument(
        "--eval-seeds", type=int, default=None,
        help=f"Inner evaluation seeds per instance (default: {EVAL_SEEDS})",
    )
    parser.add_argument(
        "--training-instances", type=int, default=None,
        help=f"Number of MA-BBOB training instances (default: {len(TRAINING_INSTANCES)})",
    )
    parser.add_argument(
        "--eval-timeout", type=int, default=None,
        help=f"Max seconds per candidate evaluation (default: {EVAL_TIMEOUT})",
    )
    parser.add_argument(
        "--no-worker-pool", action="store_true",
        help="Disable persistent worker pool (evaluate via subprocess per call)",
    )
    parser.add_argument(
        "--custom-ollama", type=str, default=None,
        help="Run a custom Ollama model (use with a single model tag as label)",
    )
    parser.add_argument(
        "--custom-gemini", type=str, default=None,
        help="Run a custom Gemini model (use with a single model tag as label)",
    )
    parser.add_argument(
        "--custom-vllm", type=str, default=None,
        help="Run a custom vLLM model (use with a single model tag as label)",
    )
    parser.add_argument(
        "--vllm-base-url", type=str, default=None,
        help=f"vLLM base URL (default: VLLM_BASE_URL env or {VLLM_BASE_URL})",
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
        list_models()
        return

    if args.summarise:
        base = Path(args.results_dir or RESULTS_DIR)
        for model_dir in sorted(base.iterdir()):
            if not model_dir.is_dir():
                continue
            for seed_dir in sorted(model_dir.glob("seed-*")):
                write_summary_csv(str(seed_dir))
        return

    if not args.models:
        parser.error("Provide model tag(s), 'all', or --list")

    # Apply sanity-check overrides
    training_instances = None
    if args.training_instances is not None:
        training_instances = list(range(args.training_instances))
    eval_seeds = args.eval_seeds
    budget = args.budget
    seeds = args.seeds

    if args.sanity:
        training_instances = training_instances or list(range(2))
        eval_seeds = eval_seeds or 1
        budget = budget or 10
        seeds = seeds or [0]
        print("[Phase1] SANITY MODE: 2 instances, 1 eval seed, 1 run seed, 10 candidates")

    # Resolve model list
    if args.models == ["all"]:
        model_list = list(CANDIDATE_MODELS.keys())
    else:
        model_list = args.models

    results_dir = args.results_dir or RESULTS_DIR

    for tag in model_list:
        # Determine model config
        if args.custom_ollama and len(model_list) == 1:
            model_cfg = {"type": "ollama", "model": args.custom_ollama}
        elif args.custom_gemini and len(model_list) == 1:
            model_cfg = {"type": "gemini", "model": args.custom_gemini}
        elif args.custom_vllm and len(model_list) == 1:
            model_cfg = {"type": "vllm", "model": args.custom_vllm}
        elif tag in CANDIDATE_MODELS:
            model_cfg = CANDIDATE_MODELS[tag]
        else:
            print(f"ERROR: unknown model tag '{tag}'", file=sys.stderr)
            print(f"Valid: {', '.join(CANDIDATE_MODELS)}", file=sys.stderr)
            print("Or use --custom-ollama / --custom-gemini / --custom-vllm", file=sys.stderr)
            sys.exit(1)

        print(f"\n{'#'*60}")
        print(f"  MODEL: {tag} ({model_cfg['model']})")
        print(f"{'#'*60}")

        result_dirs = run_model(
            model_tag=tag,
            model_cfg=model_cfg,
            seeds=seeds or RUN_SEEDS,
            budget=budget,
            port=args.port,
            base_url=args.vllm_base_url,
            eval_seeds=eval_seeds,
            training_instances=training_instances,
            eval_timeout=args.eval_timeout,
            use_worker_pool=not args.no_worker_pool,
            show_stdout=True,
            results_dir=results_dir,
        )

        # Generate summary CSVs for finished runs
        for d in result_dirs:
            write_summary_csv(d)

    print("\nAll models complete.")


if __name__ == "__main__":
    main()
