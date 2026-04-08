"""Phase 4 token-usage test: compare thinking configurations on Vertex AI.

Runs small LLaMEA experiments (default 20 candidates) with different Gemini 3
Flash thinking configs and captures per-call token usage to project full Phase 4
costs.

Thinking configs tested:
  1. default     — no thinking_config (whatever the model default is)
  2. disabled    — thinking_budget=0
  3. minimal     — thinking_budget=1024
  4. medium      — thinking_budget=8192

Usage:
  python -m experiments.phase4_token_test                  # all configs, 20 iters
  python -m experiments.phase4_token_test --configs default disabled --budget 10
  python -m experiments.phase4_token_test --list           # show available configs
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types

from .phase4_experiment import (
    make_feedback_fn,
    make_problem,
)
from .phase1_experiment import Phase1LLaMEA, make_llm
from .initial_population import get_initial_solutions
from .phase4_config import (
    CONDITIONS,
    MODEL_CFG,
    MODEL_TAG,
    N_PARENTS,
    N_OFFSPRING,
    ELITISM,
    MUTATION_PROMPTS,
    TRAINING_INSTANCES,
)

from iohblade.experiment import Experiment
from iohblade.llm import Gemini_LLM
from iohblade.loggers import ExperimentLogger


# ---------------------------------------------------------------------------
# Thinking configurations to test
# ---------------------------------------------------------------------------
THINKING_CONFIGS = {
    "default": None,  # no thinking_config at all
    "disabled": types.ThinkingConfig(thinking_budget=0),
    "minimal": types.ThinkingConfig(thinking_budget=1024),
    "medium": types.ThinkingConfig(thinking_budget=8192),
}


# ---------------------------------------------------------------------------
# Token-tracking wrapper
# ---------------------------------------------------------------------------
class TokenTrackingGeminiLLM(Gemini_LLM):
    """Gemini LLM that captures per-call token usage metadata."""

    def __init__(self, *args, thinking_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.thinking_config = thinking_config
        self.call_log = []  # list of dicts with token counts per call

    def _query(self, session_messages, max_retries=5, default_delay=10, **kwargs):
        """Override to capture usage_metadata from each response."""
        import re

        history = [
            {"role": m["role"], "parts": [m["content"]]} for m in session_messages[:-1]
        ]
        last = session_messages[-1]["content"]

        attempt = 0
        while True:
            try:
                config = self.generation_config.copy()
                config.update(**kwargs)
                # Inject thinking_config if set
                if self.thinking_config is not None:
                    config["thinking_config"] = self.thinking_config

                chat = self.client.chats.create(
                    model=self.model, history=history, config=config
                )
                response = chat.send_message(last)

                # Capture token usage
                usage = response.usage_metadata
                entry = {
                    "call_index": len(self.call_log),
                    "prompt_tokens": getattr(usage, "prompt_token_count", 0) or 0,
                    "candidates_tokens": getattr(usage, "candidates_token_count", 0) or 0,
                    "thoughts_tokens": getattr(usage, "thoughts_token_count", 0) or 0,
                    "total_tokens": getattr(usage, "total_token_count", 0) or 0,
                    "cached_tokens": getattr(usage, "cached_content_token_count", 0) or 0,
                }
                # Measure visible output length
                text = response.text
                entry["visible_output_chars"] = len(text)
                self.call_log.append(entry)

                return text

            except Exception as err:
                attempt += 1
                if attempt > max_retries:
                    raise
                delay = getattr(err, "retry_delay", None)
                if delay is not None:
                    wait = delay.seconds + 1
                else:
                    m = re.search(r"retry_delay\s*{\s*seconds:\s*(\d+)", str(err))
                    wait = int(m.group(1)) if m else default_delay * attempt
                time.sleep(wait)

    def __getstate__(self):
        state = super().__getstate__()
        # thinking_config is a genai types object — not reliably picklable
        # Store the budget int instead so we can reconstruct
        tc = self.thinking_config
        state["_thinking_budget_val"] = (
            getattr(tc, "thinking_budget", None) if tc is not None else "NONE"
        )
        state.pop("thinking_config", None)
        return state

    def __setstate__(self, state):
        budget_val = state.pop("_thinking_budget_val", "NONE")
        super().__setstate__(state)
        if budget_val == "NONE":
            self.thinking_config = None
        else:
            self.thinking_config = types.ThinkingConfig(thinking_budget=budget_val)
        if not hasattr(self, "call_log"):
            self.call_log = []

    def __deepcopy__(self, memo):
        import copy
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if k == "client":
                continue
            elif k == "thinking_config":
                setattr(new, k, v)  # shared ref is fine, it's immutable
            elif k == "call_log":
                setattr(new, k, v)  # SHARE the list so copies log to same place
            else:
                setattr(new, k, copy.deepcopy(v, memo))
        # Rebuild client
        if getattr(new, 'vertexai', False):
            new.client = genai.Client(
                vertexai=True, project=new.project, location=new.location,
            )
        else:
            new.client = genai.Client(api_key=new.api_key)
        return new


VERTEXAI_PROJECT = "thesis-mabbob-2026"
VERTEXAI_LOCATION = "global"


def make_tracking_llm(thinking_config=None):
    """Create a TokenTrackingGeminiLLM using Vertex AI with ADC."""
    return TokenTrackingGeminiLLM(
        api_key="",
        model=MODEL_CFG["model"],
        vertexai=True,
        project=VERTEXAI_PROJECT,
        location=VERTEXAI_LOCATION,
        thinking_config=thinking_config,
    )


# ---------------------------------------------------------------------------
# Run one thinking config
# ---------------------------------------------------------------------------
def run_config(config_name, thinking_config, budget=20, results_dir=None,
               condition_tag="vanilla", eval_seeds=None, training_instances=None,
               eval_timeout=None):
    """Run a mini LLaMEA experiment and return the token usage log."""
    results_dir = results_dir or f"results_token_test/{config_name}"
    os.makedirs(results_dir, exist_ok=True)

    llm = make_tracking_llm(thinking_config)

    spec = CONDITIONS[condition_tag]
    problem = make_problem(
        condition_tag,
        use_worker_pool=True,
        eval_seeds=eval_seeds,
        training_instances=training_instances,
        eval_timeout=eval_timeout,
    )
    initial_solutions = get_initial_solutions()

    method = Phase1LLaMEA(
        llm=llm,
        budget=budget,
        name=f"token_test_{config_name}",
        initial_solutions=initial_solutions,
        n_parents=N_PARENTS,
        n_offspring=N_OFFSPRING,
        elitism=ELITISM,
        mutation_prompts=MUTATION_PROMPTS,
        HPO=False,
        feature_guided_mutation=spec["sage"],
    )

    logger = ExperimentLogger(results_dir)
    experiment = Experiment(
        methods=[method],
        problems=[problem],
        budget=budget,
        seeds=[0],
        show_stdout=True,
        log_stdout=True,
        exp_logger=logger,
        n_jobs=1,
    )

    n_inst = len(training_instances) if training_instances else len(TRAINING_INSTANCES)
    n_eval = eval_seeds if eval_seeds else 5
    sage_str = " + SAGE" if spec["sage"] else ""
    print(f"\n{'='*60}")
    print(f"  CONDITION: {condition_tag} ({spec['feedback']}{sage_str})")
    print(f"  Thinking: {config_name}")
    print(f"  LLaMEA budget: {budget} candidates")
    print(f"  Eval: {n_inst} instances x {n_eval} seeds")
    print(f"{'='*60}\n")

    t0 = time.time()
    experiment()
    elapsed = time.time() - t0

    return {
        "config_name": config_name,
        "condition": condition_tag,
        "budget": budget,
        "elapsed_s": elapsed,
        "call_log": llm.call_log,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
# Gemini 3 Flash pricing (per 1M tokens, USD → EUR at ~0.85)
INPUT_PRICE_PER_M = 0.15 * 0.85    # $0.15/1M input → ~€0.1275/1M
OUTPUT_PRICE_PER_M = 3.50 * 0.85   # $3.50/1M output → ~€2.975/1M  (thinking billed as output)
# Note: cached input is cheaper but we'll use uncached for conservative estimate

FULL_PHASE4_CANDIDATES = 20_000  # 4 conditions × 10 seeds × 500 candidates (worst case)


def print_report(results):
    """Print a comparison table and cost projections."""

    print("\n" + "=" * 80)
    print("  TOKEN USAGE & TIMING — GEMINI 3 FLASH ON VERTEX AI")
    print("=" * 80)

    summaries = []
    for r in results:
        log = r["call_log"]
        n_calls = len(log)
        if n_calls == 0:
            summaries.append(None)
            continue

        total_prompt = sum(e["prompt_tokens"] for e in log)
        total_candidates = sum(e["candidates_tokens"] for e in log)
        total_thoughts = sum(e["thoughts_tokens"] for e in log)
        total_cached = sum(e["cached_tokens"] for e in log)
        total_all = sum(e["total_tokens"] for e in log)

        avg_prompt = total_prompt / n_calls
        avg_candidates = total_candidates / n_calls
        avg_thoughts = total_thoughts / n_calls
        avg_total = total_all / n_calls

        label = r.get("condition", r["config_name"])

        summaries.append({
            "label": label,
            "config": r["config_name"],
            "condition": r.get("condition", "vanilla"),
            "n_calls": n_calls,
            "elapsed_s": r["elapsed_s"],
            "avg_prompt": avg_prompt,
            "avg_candidates": avg_candidates,
            "avg_thoughts": avg_thoughts,
            "avg_total": avg_total,
            "total_prompt": total_prompt,
            "total_candidates": total_candidates,
            "total_thoughts": total_thoughts,
            "total_cached": total_cached,
            "total_all": total_all,
            "sec_per_candidate": r["elapsed_s"] / r["budget"],
        })

    # Per-call averages table
    print(f"\n{'Condition':<14} {'Calls':>6} {'Avg In':>9} {'Avg Out':>9} "
          f"{'Avg Think':>11} {'Time':>8} {'s/cand':>8}")
    print("-" * 72)
    for s in summaries:
        if s is None:
            continue
        print(f"{s['label']:<14} {s['n_calls']:>6} {s['avg_prompt']:>9,.0f} "
              f"{s['avg_candidates']:>9,.0f} {s['avg_thoughts']:>11,.0f} "
              f"{s['elapsed_s']:>7.0f}s {s['sec_per_candidate']:>7.1f}s")

    # Timing extrapolation
    print(f"\n{'='*80}")
    print(f"  TIMING PROJECTION → 500 candidates/seed")
    print(f"{'='*80}")
    print(f"\n{'Condition':<14} {'s/cand':>8} {'Hours/seed':>12} {'Cands/day':>12} {'RPD/seed':>10}")
    print("-" * 60)
    for s in summaries:
        if s is None:
            continue
        spc = s["sec_per_candidate"]
        hours_per_seed = (500 * spc) / 3600
        cands_per_day = 86400 / spc
        print(f"{s['label']:<14} {spc:>7.1f}s {hours_per_seed:>11.1f}h "
              f"{cands_per_day:>12,.0f} {cands_per_day:>10,.0f}")

    # Cost projection
    # Actual rates from March billing CSV
    INPUT_EUR_PER_M = 0.424
    OUTPUT_EUR_PER_M = 2.548
    SEEDS = {"vanilla": 5, "behavioural": 5, "sage": 10, "combined": 10}

    print(f"\n{'='*80}")
    print(f"  COST PROJECTION (30 seeds: 5v + 5b + 10s + 10c × 500 candidates)")
    print(f"{'='*80}")

    total_cost = 0
    total_rpd_needed = 0
    print(f"\n{'Condition':<14} {'Seeds':>6} {'Candidates':>11} {'Output Tok':>12} "
          f"{'Cost':>10} {'RPD total':>10}")
    print("-" * 68)
    for s in summaries:
        if s is None:
            continue
        cond = s["condition"]
        n_seeds = SEEDS.get(cond, 0)
        n_cands = n_seeds * 500
        # Scale tokens from test to projected
        scale = n_cands / s["n_calls"] if s["n_calls"] > 0 else 0
        proj_in = s["total_prompt"] * scale
        proj_out = (s["total_candidates"] + s["total_thoughts"]) * scale
        cost = (proj_in / 1e6) * INPUT_EUR_PER_M + (proj_out / 1e6) * OUTPUT_EUR_PER_M
        rpd = n_seeds * (86400 / s["sec_per_candidate"])
        total_cost += cost
        total_rpd_needed = max(total_rpd_needed, rpd)
        print(f"{cond:<14} {n_seeds:>6} {n_cands:>11,} {proj_out:>12,.0f} "
              f"{cost:>9.2f}€ {rpd:>10,.0f}")

    with_tax = total_cost * 1.19
    print(f"\n  Subtotal:   €{total_cost:.2f}")
    print(f"  + 19% VAT:  €{with_tax:.2f}")
    print(f"  Free credits: €254")
    status = "WITHIN budget" if with_tax <= 254 else f"OVER by €{with_tax - 254:.0f}"
    print(f"  → {status}")

    # Save raw data
    output_file = "results_token_test/token_usage_summary.json"
    os.makedirs("results_token_test", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump({
            "model": MODEL_TAG,
            "results": [
                {
                    "config": r["config_name"],
                    "condition": r.get("condition", "vanilla"),
                    "budget": r["budget"],
                    "elapsed_s": r["elapsed_s"],
                    "call_log": r["call_log"],
                }
                for r in results
            ],
        }, f, indent=2)
    print(f"\n  Raw data saved to {output_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Test Gemini 3 Flash thinking configs and project Phase 4 costs",
    )
    parser.add_argument(
        "--configs", nargs="+", default=None,
        help=f"Thinking configs to test (default: all). Options: {', '.join(THINKING_CONFIGS)}",
    )
    parser.add_argument(
        "--conditions", nargs="+", default=None,
        help=f"Phase 4 conditions to test (default: all). Options: {', '.join(CONDITIONS)}",
    )
    parser.add_argument(
        "--budget", type=int, default=20,
        help="LLaMEA candidates per config (default: 20)",
    )
    parser.add_argument(
        "--eval-seeds", type=int, default=None,
        help="Inner evaluation seeds per instance (default: Phase 4 config)",
    )
    parser.add_argument(
        "--eval-timeout", type=int, default=None,
        help="Max seconds per candidate evaluation",
    )
    parser.add_argument(
        "--full-eval", action="store_true",
        help="Use full Phase 4 eval config (20 instances, 5 seeds, 1200s timeout)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available configs and conditions",
    )
    args = parser.parse_args()

    if args.list:
        print("Thinking configs:")
        for name, cfg in THINKING_CONFIGS.items():
            if cfg is None:
                print(f"  {name:<12} — no thinking_config (model default)")
            else:
                budget = getattr(cfg, "thinking_budget", "N/A")
                print(f"  {name:<12} — thinking_budget={budget}")
        print("\nPhase 4 conditions:")
        for tag, spec in CONDITIONS.items():
            sage_str = " + SAGE" if spec["sage"] else ""
            print(f"  {tag:<14} — {spec['feedback']}{sage_str}")
        return

    config_names = args.configs or ["disabled"]
    for name in config_names:
        if name not in THINKING_CONFIGS:
            print(f"ERROR: unknown config '{name}'", file=sys.stderr)
            sys.exit(1)

    condition_names = args.conditions or list(CONDITIONS.keys())
    for tag in condition_names:
        if tag not in CONDITIONS:
            print(f"ERROR: unknown condition '{tag}'", file=sys.stderr)
            sys.exit(1)

    # Eval config
    eval_seeds = args.eval_seeds
    training_instances = None
    eval_timeout = args.eval_timeout
    if args.full_eval:
        training_instances = TRAINING_INSTANCES
        eval_seeds = eval_seeds or 5
        eval_timeout = eval_timeout or 1200

    n_inst = len(training_instances) if training_instances else len(TRAINING_INSTANCES)
    n_eval = eval_seeds or 5
    print(f"Token/timing test: {len(config_names)} thinking config(s) x "
          f"{len(condition_names)} condition(s) x {args.budget} candidates")
    print(f"Model: {MODEL_TAG} ({MODEL_CFG['model']})")
    print(f"Eval: {n_inst} instances x {n_eval} seeds"
          f"{' (full Phase 4 config)' if args.full_eval else ''}")

    results = []
    for thinking_name in config_names:
        cfg = THINKING_CONFIGS[thinking_name]
        for cond in condition_names:
            r = run_config(
                thinking_name, cfg, budget=args.budget,
                results_dir=f"results_token_test/{thinking_name}_{cond}",
                condition_tag=cond,
                eval_seeds=eval_seeds,
                training_instances=training_instances,
                eval_timeout=eval_timeout,
            )
            results.append(r)

    print_report(results)


if __name__ == "__main__":
    main()
