"""Re-classify Stage 4 failures by re-running the BLADE compile + smoke pipeline.

The original Stage 4 logs do not preserve the per-failure error string. We
recover the failure category by replaying the candidate's source code through
the same gate the framework used at runtime.

Categories:
  - code_generation:    no class is produced or the file does not parse.
  - import_violation:   the code imports a disallowed package.
  - interface_mismatch: __init__(budget, dim) or __call__(func) cannot be
                        invoked with the framework's call signature.
  - runtime_error:      compiles and instantiates, but raises during the
                        first BBOB-level call.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Tuple

_THESIS_ROOT = Path(__file__).resolve().parents[2]


def _load_llamea_utils():
    spec = importlib.util.spec_from_file_location(
        'llamea.utils',
        os.path.join(str(_THESIS_ROOT), 'LLaMEA', 'llamea', 'utils.py'),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ALLOWED_IMPORTS = ['numpy']  # mirrors experiments/phase1_config.py:ALLOWED_IMPORTS


def classify_failure(code: str) -> Tuple[str | None, str]:
    """Return (label, detail). label is None for code that runs cleanly.

    The smoke test instantiates the candidate with budget=100, dim=2 and
    invokes it on BBOB function 11 (Discus), instance 1, dim 2.
    """
    if not code or 'class ' not in code:
        return 'code_generation', 'no class definition found'

    llamea_utils = _load_llamea_utils()

    # Stage 1 — pre-flight import gate.
    try:
        global_ns, possible_issue = llamea_utils.prepare_namespace(
            code, allowed=_ALLOWED_IMPORTS,
        )
    except SyntaxError as e:
        return 'code_generation', str(e)
    except Exception as e:
        return 'code_generation', f'{type(e).__name__}: {e}'
    if possible_issue:
        return 'import_violation', possible_issue

    # Stage 2 — exec.
    local_ns: dict = {}
    try:
        exec(code, global_ns, local_ns)
    except SyntaxError as e:
        return 'code_generation', str(e)
    except ImportError as e:
        return 'import_violation', str(e)
    except Exception as e:
        return 'code_generation', f'{type(e).__name__}: {e}'

    # Pull the first class from the module.
    classes = [v for k, v in local_ns.items() if isinstance(v, type)]
    if not classes:
        return 'code_generation', 'class not found in compiled namespace'
    cls = classes[-1]

    # Stage 3 — instantiate.
    try:
        algo = cls(budget=100, dim=2)
    except TypeError as e:
        return 'interface_mismatch', f'__init__ failed: {e}'
    except Exception as e:
        return 'runtime_error', f'__init__ raised {type(e).__name__}: {e}'

    # Stage 4 — smoke-test on BBOB function 11.
    try:
        import ioh
        from ioh import logger as ioh_logger
        from iohblade.utils import aoc_logger, OverBudgetException
    except ImportError as e:
        raise RuntimeError(f'cannot import ioh/iohblade for smoke test: {e}') from e

    try:
        l_tmp = aoc_logger(100, upper=1e2, triggers=[ioh_logger.trigger.ALWAYS])
        prob = ioh.get_problem(11, 1, 2)
        prob.attach_logger(l_tmp)
        algo(prob)
    except OverBudgetException:
        return None, 'ok'
    except TypeError as e:
        return 'interface_mismatch', f'__call__ failed: {e}'
    except Exception as e:
        return 'runtime_error', f'{type(e).__name__}: {e}'

    return None, 'ok'
