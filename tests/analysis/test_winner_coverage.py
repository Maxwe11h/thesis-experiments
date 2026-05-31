from pathlib import Path

import numpy as np

from analysis.winner_coverage import (
    load_traced_class,
    map_missing_to_ast,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "winner_with_deadcode.py"


def test_load_traced_class_returns_last_class_with_real_filename():
    cls = load_traced_class(FIXTURE)
    assert cls.__name__ == "FixtureWinner"
    # __call__'s code object must carry the real file path (needed for coverage).
    assert cls.__call__.__code__.co_filename == str(FIXTURE)


def test_map_missing_to_ast_labels_constructs():
    src = FIXTURE.read_text()
    # Lines of the dead helper body and the except body (1-indexed) from the fixture.
    lines = src.splitlines()
    helper_line = next(i + 1 for i, l in enumerate(lines) if "return np.zeros(self.dim)" in l)
    except_body = next(i + 1 for i, l in enumerate(lines) if "y = 1e30" in l)
    constructs = map_missing_to_ast(FIXTURE, [helper_line, except_body])
    kinds = {c["line"]: c["construct"] for c in constructs}
    assert "FunctionDef:_never_called_helper" in kinds[helper_line]
    assert "ExceptHandler" in kinds[except_body]
