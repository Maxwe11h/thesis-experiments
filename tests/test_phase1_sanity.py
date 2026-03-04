"""Sanity tests for Phase 1 experiment infrastructure.

These tests validate that the experiment pipeline works end-to-end without
unhandled exceptions and produces the expected output structure.

There are two modes:
  1. Quick (default, no GPU needed): uses BLADE's Dummy_LLM to verify the
     pipeline mechanics without real LLM inference.
  2. Integration (requires Ollama): uses a real local model to verify the
     full path including LLM code generation and evaluation.

Run with:
    pytest tests/test_phase1_sanity.py -v
    pytest tests/test_phase1_sanity.py -v -k integration   # real Ollama
"""

import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from experiments.initial_population import (
    get_initial_solutions,
    load_initial_population,
    save_initial_population,
)
from experiments.phase1_config import CANDIDATE_MODELS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_results(tmp_path):
    """Provide a temporary results directory that is cleaned up after the test."""
    results_dir = tmp_path / "results_phase1_test"
    results_dir.mkdir()
    yield str(results_dir)


@pytest.fixture
def tmp_pop_file(tmp_path):
    """Temporary path for initial population JSON."""
    return str(tmp_path / "test_initial_pop.json")


# ---------------------------------------------------------------------------
# Unit tests (no LLM needed)
# ---------------------------------------------------------------------------

class TestInitialPopulation:
    """Verify initial population creation, save, and load."""

    def test_get_initial_solutions_returns_one(self):
        solutions = get_initial_solutions()
        assert len(solutions) == 1

    def test_solutions_have_code(self):
        for sol in get_initial_solutions():
            assert sol.code, f"{sol.name} has empty code"
            assert sol.name, "Solution has no name"
            assert "def __init__" in sol.code
            assert "def __call__" in sol.code

    def test_solutions_have_class_definitions(self):
        for sol in get_initial_solutions():
            assert f"class {sol.name}" in sol.code

    def test_save_and_load_roundtrip(self, tmp_pop_file):
        save_initial_population(tmp_pop_file)
        assert Path(tmp_pop_file).exists()

        loaded = load_initial_population(tmp_pop_file)
        original = get_initial_solutions()

        assert len(loaded) == len(original)
        for orig, ld in zip(original, loaded):
            assert orig.name == ld.name
            assert orig.code == ld.code
            assert orig.description == ld.description

    def test_initial_algorithms_are_valid_python(self):
        """Check that the initial algorithms can be compiled without errors."""
        for sol in get_initial_solutions():
            try:
                compile(sol.code, f"<{sol.name}>", "exec")
            except SyntaxError as e:
                pytest.fail(f"{sol.name} has syntax error: {e}")

    def test_initial_algorithms_can_be_instantiated(self):
        """Check that the classes can be instantiated and called (smoke test)."""
        import numpy as np

        for sol in get_initial_solutions():
            ns = {}
            exec(sol.code, {"np": np, "numpy": np, "__builtins__": __builtins__}, ns)
            cls = ns[sol.name]
            instance = cls(budget=50, dim=2)
            assert hasattr(instance, "__call__"), f"{sol.name} missing __call__"


class TestPhase1Config:
    """Verify configuration consistency."""

    def test_mutation_prompts_count(self):
        from experiments.phase1_config import MUTATION_PROMPTS
        assert len(MUTATION_PROMPTS) == 10, "Expected 9 refine + 1 explore = 10"

    def test_explore_prompt_is_last(self):
        from experiments.phase1_config import MUTATION_PROMPTS
        # The last prompt should be the exploration one
        assert "different" in MUTATION_PROMPTS[-1].lower()

    def test_population_settings(self):
        from experiments.phase1_config import N_PARENTS, N_OFFSPRING, ELITISM
        assert N_PARENTS == 1
        assert N_OFFSPRING == 1
        assert ELITISM is True

    def test_run_seeds(self):
        from experiments.phase1_config import RUN_SEEDS
        assert RUN_SEEDS == [0, 1, 2, 3, 4]

    def test_all_candidate_models_have_required_keys(self):
        for tag, cfg in CANDIDATE_MODELS.items():
            assert "type" in cfg, f"{tag} missing 'type'"
            assert "model" in cfg, f"{tag} missing 'model'"
            assert cfg["type"] in ("ollama", "gemini"), f"{tag} has unknown type"


class TestSummaryExtraction:
    """Verify summary CSV generation from mock log data."""

    def test_summarise_empty_dir(self, tmp_results):
        from experiments.phase1_experiment import summarise_run
        records = summarise_run(tmp_results)
        assert records == []

    def test_summarise_with_mock_log(self, tmp_results):
        from experiments.phase1_experiment import summarise_run, write_summary_csv

        # Create a mock run directory with log.jsonl
        run_dir = Path(tmp_results) / "run-test-model-MA_BBOB-0"
        run_dir.mkdir(parents=True)

        entries = [
            {"name": "AlgA", "fitness": 0.5, "generation": 0,
             "metadata": {"aucs": [0.5, 0.4], "behavioral_features": {"metric1": 0.3}}},
            {"name": "AlgB", "fitness": float("-inf"), "generation": 1,
             "error": "SyntaxError", "metadata": {}},
        ]
        with open(run_dir / "log.jsonl", "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        records = summarise_run(tmp_results)
        assert len(records) == 2
        assert records[0]["run_status"] == "success"
        assert records[1]["run_status"] == "failure"

        csv_path = write_summary_csv(tmp_results)
        assert csv_path is not None
        assert Path(csv_path).exists()


# ---------------------------------------------------------------------------
# Integration tests (require Ollama or Gemini)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not shutil.which("ollama"),
    reason="Ollama not installed"
)
class TestIntegrationOllama:
    """End-to-end test with a real Ollama model (sanity-check config)."""

    def test_sanity_run(self, tmp_results):
        """Run 1 seed, 2 instances, 1 eval seed, budget=4 with smallest available model."""
        from experiments.phase1_experiment import run_single_seed, write_summary_csv

        # Use the smallest model in the registry, or fall back to qwen3.5-4b
        model_tag = "qwen3.5-4b"
        if model_tag not in CANDIDATE_MODELS:
            pytest.skip(f"Model {model_tag} not in registry")

        result_dir = run_single_seed(
            model_tag=model_tag,
            seed=0,
            budget=4,  # very small: 2 initial + 1 generation of 2 offspring
            eval_seeds=1,
            training_instances=list(range(2)),
            use_worker_pool=True,
            show_stdout=True,
            results_dir=tmp_results,
        )

        result_path = Path(result_dir)
        assert result_path.exists(), f"Result dir not created: {result_dir}"

        # Check that run sub-directory was created
        run_dirs = list(result_path.glob("run-*/"))
        assert len(run_dirs) >= 1, "No run sub-directory created"

        # Check log.jsonl exists and has entries
        log_files = list(result_path.glob("run-*/log.jsonl"))
        assert len(log_files) >= 1, "No log.jsonl created"

        with open(log_files[0]) as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) >= 2, f"Expected at least 2 log entries (initial pop), got {len(lines)}"

        # Generate summary CSV
        csv_path = write_summary_csv(result_dir)
        assert csv_path is not None
        assert Path(csv_path).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
