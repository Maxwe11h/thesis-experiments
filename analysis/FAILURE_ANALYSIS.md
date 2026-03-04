# Failure Analysis: Feature-Selection Experiment

## 1. Overview

Of the 1,200 candidates generated across 12 conditions (100 per condition), **922 (76.8%) failed** to produce a valid fitness score. This document categorises every failure by root cause, traces each cause to a specific point in the code generation or evaluation pipeline, and identifies actionable fixes.

## 2. The Pipeline

When LLaMEA generates a candidate, it passes through four stages before receiving a fitness score:

```
LLM response → Code extraction → Compilation & smoke test → Full MA-BBOB evaluation
```

A failure at any stage results in `fitness = "-inf"` and the candidate is discarded.

**Stage 1 — Code extraction** (`LLaMEA/llamea/llm.py:218-243`):
- Regex `r"```(?:python|diff)?\n(.*?)\n```"` extracts content from markdown code blocks.
- A second regex `r"(?:def|class)\s*(\w*).*\:"` extracts the class/function name.
- If no code block is found, `NoCodeException` is raised.

**Stage 2 — Compilation** (`experiments/mabbob_problem.py:107-119`):
- `prepare_namespace()` validates imports (only `numpy` allowed).
- `exec(code, global_ns, local_ns)` compiles the code.
- Syntax errors, indentation errors, and import violations are caught here.

**Stage 3 — Smoke test** (`experiments/mabbob_problem.py:121-132`):
- Instantiates: `local_ns[algorithm_name](budget=100, dim=2)`
- Calls: `algorithm(prob_tmp)` on a plain BBOB problem.
- Any `__init__` or `__call__` signature mismatch is caught here.

**Stage 4 — Full evaluation** (`experiments/mabbob_problem.py:134-183`):
- Runs the algorithm on 10 MA-BBOB instances x 5 seeds (50 total evaluations).
- Runtime errors (array shape mismatches, undefined variables, etc.) are caught here.

## 3. Root Cause Summary

| Root Cause | Count | % of Failures | Pipeline Stage |
|-----------|-------|--------------|----------------|
| **Interface mismatch** (wrong `__init__`/`__call__` signature) | 459 | 49.8% | Stage 3 |
| **Code generation failure** (no valid Python class produced) | 286 | 31.0% | Stage 1 |
| **Runtime errors** (correct structure, bugs in logic) | 177 | 19.2% | Stage 3-4 |
| **Total** | **922** | **100%** | |

## 4. Detailed Breakdown

### 4.1 Interface Mismatch (459 failures, 49.8%)

The LLM produced syntactically valid Python classes, but with constructor or call signatures that don't match the framework's expectations.

**What the framework does:**
```python
# Smoke test (mabbob_problem.py:126-127)
alg = local_ns[algorithm_name](budget=100, dim=2)
alg(prob_tmp)

# Full evaluation (mabbob_problem.py:165-166)
algorithm = local_ns[algorithm_name](budget=budget, dim=dim)
algorithm(f_new)
```

**What the prompt says** (`BLADE/iohblade/problems/mabbob.py:79`):
> "The code should contain an `__init__(self, budget, dim)` function with optional additional arguments and the function `def __call__(self, func)`"

**What the example shows** (`BLADE/iohblade/problems/mabbob.py:87-103`):
```python
class RandomSearch:
    def __init__(self, budget=10000, dim=10):
        ...
    def __call__(self, func):
        x = np.random.uniform(func.bounds.lb, func.bounds.ub)
        ...
```

Despite this, the LLM frequently deviates:

| Sub-type | Count | What the LLM wrote | What was expected |
|----------|-------|--------------------|--------------------|
| `__init__` rejects `budget` kwarg | 303 | `__init__(self)` or `__init__(self, pop_size, step_size)` | `__init__(self, budget, dim)` |
| `__call__` expects `bounds` arg | 107 | `__call__(self, func, bounds)` | `__call__(self, func)` — bounds via `func.bounds.lb/ub` |
| `__init__` rejects `dim` kwarg | 31 | `__init__(self, budget)` without `dim` | `__init__(self, budget, dim)` |
| Other arg mismatches | 18 | Various | Various |

**Observation — the `success_rate` pathology:** The `success_rate` condition generated 85 `__call__` signature failures in a row. The LLM got stuck producing variations of `AdaptiveDirectionalSearch.__call__(self, func, bounds)` across 98 consecutive generations (gen 0-98), only producing one valid candidate at gen 99. The (1+1)-ES provides the current best's code in the mutation prompt, but since nearly every generation failed, the LLM had almost no positive signal to learn from, creating a vicious cycle.

**Observation — `init_kwarg` is persistent across all conditions:** The `budget` keyword argument failure appeared in every single condition (7-40 instances each), suggesting this is a fundamental tendency of Qwen3 8B, not caused by the feedback content. The model has a strong prior toward writing `__init__` methods that don't accept `budget` and `dim` — likely because most optimization code in its training data uses different constructor conventions.

### 4.2 Code Generation Failure (286 failures, 31.0%)

The LLM failed to produce a Python class at all:

| Sub-type | Count | Description |
|----------|-------|-------------|
| **Wrote function instead of class** | 224 | `def optimize(f, bounds, budget):` instead of `class Optimizer: def __init__... def __call__...` |
| **Produced pseudocode or text** | 43 | Natural language algorithm descriptions, bullet-point outlines, or partial code fragments |
| **No code block found** | 18 | Response lacked `` ```python ... ``` `` markers entirely. Some of these contained the explicit exception message `"Could not extract generated code. The code should be encapsulated with ``` in your response."` |
| **Class present but extraction failed** | 1 | Edge case in regex matching |

**Observation — function vs class:** The 224 function-instead-of-class failures are the second largest failure mode overall. The code itself is often a perfectly reasonable optimization algorithm, but written as a standalone function:

```python
# What the LLM produced (fails):
def improved_optimization(f, bounds, num_iterations=1000):
    best_x = np.random.uniform(bounds[0], bounds[1])
    ...

# What was needed (succeeds):
class ImprovedOptimization:
    def __init__(self, budget=1000, dim=10):
        self.budget = budget
        ...
    def __call__(self, func):
        ...
```

The name extraction regex (`r"(?:def|class)\s*(\w*).*\:"`) matches `def` and returns the function name, but the evaluation framework tries to instantiate it as `function_name(budget=100, dim=2)`, which fails because plain functions don't accept keyword arguments this way.

**Observation — pseudocode increases with context length:** Among the 43 pseudocode cases, many occur in the middle of a run (gen 30-50+) when the LLM's conversation context is already long, suggesting Qwen3 8B may struggle to maintain the output format over extended prompting sequences.

### 4.3 Runtime Errors (177 failures, 19.2%)

The LLM produced a correctly structured class, but the algorithm logic contained bugs:

| Sub-type | Count | Example |
|----------|-------|---------|
| **Undefined variable** (`NameError`) | 24 | `name 'initial_exploration' is not defined` |
| **Missing attribute** (`AttributeError`) | 20 | `'Algorithm' object has no attribute 'exploration_rate'` |
| **Object not callable** | 20 | Tried to call `bounds()` as a function |
| **Wrong bounds API** (`RealBounds not subscriptable`) | 16 | Used `func.bounds[0]` instead of `func.bounds.lb` |
| **Array shape mismatch** (`a must be 1-dimensional`) | 28 | Incorrect use of `np.random.choice` or matrix ops |
| **Boolean ambiguity** (`Truth value of array`) | 17 | `if array:` instead of `if array.any():` |
| **Syntax/indentation** | 3 | Minor formatting errors |
| **Other** (ZeroDivision, overflow, etc.) | 49 | Various numerical issues |

**Observation — IOH bounds API:** 16 failures used `func.bounds[0]` instead of `func.bounds.lb`. The example in the prompt shows `func.bounds.lb` and `func.bounds.ub`, but the LLM frequently defaults to subscript notation, likely from training data that uses `bounds[0]` for lower bound.

## 5. Failure Rate by Condition

| Condition | Failure Rate | Dominant Failure Mode | Notes |
|-----------|-------------|----------------------|-------|
| average_convergence_rate | 50% | function_not_class (20), init_kwarg (25) | Lowest failure rate; LLM produced functions often |
| avg_exploration_pct | 55% | init_kwarg (17), runtime (21) | Second-best; diverse failure modes |
| avg_improvement | 63% | init_kwarg (22), function_not_class (14), runtime (14) | Balanced across modes |
| vanilla | 67% | init_kwarg (40), runtime (19) | `init_kwarg` heavily dominant |
| longest_no_improvement_streak | 68% | runtime (39), init_kwarg (20) | Most runtime errors of any condition |
| dispersion | 74% | init_kwarg (29), function_not_class (29) | Even split between two modes |
| avg_exploitation_pct | 85% | function_not_class (38), init_kwarg (27) | |
| avg_nearest_neighbor_distance | 86% | init_kwarg (40), runtime (27) | |
| last_improvement_fraction | 87% | last_improvement_fraction (39), init_kwarg (35) | High function_not_class rate |
| avg_distance_to_best | 94% | function_not_class (67), init_kwarg (24) | Extreme function_not_class rate |
| intensification_ratio | 94% | function_not_class (53), init_kwarg (17), kwarg_other (12) | |
| success_rate | 99% | call_signature (85), init_kwarg (7) | Pathological repetition of same error |

### 5.1 Failure Rate Does Not Consistently Worsen Over Time

Comparing failure rates in the first 50 evaluations vs the last 50:

| Condition | First 50 | Last 50 | Trend |
|-----------|----------|---------|-------|
| vanilla | 70% | 64% | improving |
| average_convergence_rate | 58% | 42% | improving |
| avg_exploration_pct | 58% | 52% | improving |
| avg_nearest_neighbor_distance | 90% | 82% | improving |
| avg_exploitation_pct | 88% | 82% | improving |
| avg_improvement | 62% | 64% | stable |
| longest_no_improvement_streak | 68% | 68% | stable |
| intensification_ratio | 96% | 92% | stable |
| success_rate | 100% | 98% | stable |
| avg_distance_to_best | 94% | 94% | stable |
| dispersion | 68% | 80% | worsening |
| last_improvement_fraction | 84% | 90% | worsening |

Most conditions stay stable or improve slightly, suggesting the failure rate is not primarily a context-length degradation issue. The LLM makes the same types of mistakes throughout the entire run.

## 6. The Prompt-Code Gap

The current prompt chain (task prompt + example + format prompt) from `BLADE/iohblade/problems/mabbob.py:74-113`:

```
TASK PROMPT:
  "The code should contain an `__init__(self, budget, dim)` function with
   optional additional arguments and the function `def __call__(self, func)`"

EXAMPLE:
  class RandomSearch:
      def __init__(self, budget=10000, dim=10):  ...
      def __call__(self, func):
          x = np.random.uniform(func.bounds.lb, func.bounds.ub)
          ...

FORMAT PROMPT:
  "Give the response in the format:
   # Description: <short-description>
   # Code:
   ```python
   <code>
   ```"
```

The constraint is stated once in the task prompt and demonstrated once in the example. For Qwen3 8B, this is not sufficient. The model does not reliably infer from a single example that:
1. It **must** write a class, not a function.
2. The `__init__` **must** accept `budget` and `dim` as keyword arguments.
3. The `__call__` **must** accept only `func` (no extra args like `bounds`).
4. Bounds are accessed via `func.bounds.lb` / `func.bounds.ub`, not `func.bounds[0]`.

## 7. Potential Fixes

### 7.1 Prompt Engineering (targets: interface mismatch + code generation)

Strengthen the task prompt with explicit constraints and negative examples. The prompt is customisable via `MaBBOBProblem` (which inherits from `MA_BBOB`): the `task_prompt`, `example_prompt`, and `format_prompt` attributes can be overridden after `super().__init__()`.

Additions to consider:
- Explicit "IMPORTANT" block repeating the class interface requirements
- A negative example showing what NOT to do (`def optimize(...)` without a class)
- Explicit mention of `func.bounds.lb` / `func.bounds.ub` in the task prompt
- A second example beyond RandomSearch to reinforce the pattern

### 7.2 Error-Specific Feedback (targets: interface mismatch)

When the smoke test fails with a known signature error, replace the raw exception message with a targeted correction:

| Current feedback | Proposed feedback |
|-----------------|-------------------|
| `"__init__() got an unexpected keyword argument 'budget'"` | `"ERROR: Your __init__ must accept budget and dim as keyword arguments: def __init__(self, budget, dim). Your code defined __init__ without these parameters."` |
| `"__call__() missing 1 required positional argument: 'bounds'"` | `"ERROR: Your __call__ must accept only func: def __call__(self, func). Access bounds via func.bounds.lb and func.bounds.ub, not as a separate argument."` |

### 7.3 Code Repair / Validation Step (targets: code generation)

Before evaluation, add a lightweight check:
- If extraction found a `def` but no `class`: wrap the function in a class template with `__init__` and `__call__`.
- If `__init__` doesn't accept `budget`/`dim`: add them to the signature.
- This is an invasive change and may introduce its own issues — should be tested carefully.

### 7.4 Retry on Extraction Failure (targets: no code block / pseudocode)

If the extraction regex finds no code block, re-prompt the LLM with: "Your response must contain Python code in a ```python ... ``` code block." This adds one extra LLM call per failure but could recover 18-43 candidates.

### 7.5 Estimated Impact

| Fix | Effort | Targets (count) | Estimated saves |
|-----|--------|-----------------|-----------------|
| Prompt engineering | Low | init_kwarg (303), function_not_class (224), call_signature (107) | 150-300 |
| Error-specific feedback | Low | init_kwarg (303), call_signature (107) | 50-100 additional |
| Code repair | Medium | function_not_class (224) | 100-150 additional |
| Retry on extraction failure | Low | no_code_block (18), pseudocode (43) | 10-30 |

**Conservative estimate**: prompt engineering alone could reduce the failure rate from ~77% to ~50-55%. Combined with error-specific feedback and retry, potentially to ~35-45%.

## 8. Data Sources

All failure data was extracted from `results/<condition>/run-*/log.jsonl`. Each entry contains:
- `fitness`: `"-inf"` for failed candidates (stored as string by BLADE's JSON serializer)
- `name`: empty string when code extraction found no class/function
- `code`: the full extracted code (or the raw exception message for extraction failures)
- `feedback`: error message from the stage that failed
- `error`: secondary error field (mostly empty; primary error info is in `feedback`)
