#!/usr/bin/env python3
"""Dead-code / branch-coverage measurement for the four Stage-4 winners (section 5.4.5).

Drives each winner through the full-suite evaluation harness under coverage.py
(branch=True), then maps every unreached line back to its AST construct. Reports
per winner: % of statements/branches executed, never-executed constructs, and
how often each bare `except` handler actually fired.

Run locally (thesis env):  python analysis/winner_coverage.py
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
WINNERS = {
    "vanilla":  REPO_ROOT / "docs/stage4_winners/vanilla_winner.py",
    "neutral":  REPO_ROOT / "docs/stage4_winners/neutral_winner.py",
    "sage":     REPO_ROOT / "docs/stage4_winners/sage_winner.py",
    "combined": REPO_ROOT / "docs/stage4_winners/combined_neutral_winner.py",
}


def load_traced_class(winner_path: Path) -> type:
    """Compile+exec the winner with its REAL filename so coverage can attribute
    executed lines to it, returning the last-defined class (harness convention)."""
    src = Path(winner_path).read_text()
    code = compile(src, str(winner_path), "exec")
    ns: dict = {}
    exec(code, ns, ns)
    classes = [v for v in ns.values() if isinstance(v, type)]
    if not classes:
        raise RuntimeError(f"no class defined in {winner_path}")
    return classes[-1]


def _enclosing_label(tree: ast.AST, line: int) -> str:
    """Best (innermost) AST construct label covering `line`."""
    best = ("Module", -1, 1 << 30)  # (label, depth, span)
    for node in ast.walk(tree):
        if not hasattr(node, "lineno"):
            continue
        start = node.lineno
        end = getattr(node, "end_lineno", start)
        if start <= line <= end:
            span = end - start
            if isinstance(node, ast.FunctionDef):
                label = f"FunctionDef:{node.name}"
            elif isinstance(node, ast.ExceptHandler):
                label = "ExceptHandler"
            elif isinstance(node, ast.If):
                label = "If"
            elif isinstance(node, ast.While):
                label = "While"
            elif isinstance(node, ast.For):
                label = "For"
            else:
                continue
            # prefer the tightest (smallest-span) matching construct
            if span < best[2]:
                best = (label, 0, span)
    return best[0]


def map_missing_to_ast(winner_path: Path, missing_lines: list[int]) -> list[dict]:
    """For each unreached line, attach the innermost interesting AST construct."""
    tree = ast.parse(Path(winner_path).read_text())
    out = []
    for ln in sorted(set(missing_lines)):
        out.append({"line": ln, "construct": _enclosing_label(tree, ln)})
    return out


def _except_handler_lines(winner_path: Path) -> list[list[int]]:
    """Return, per except handler, the set of body line numbers."""
    tree = ast.parse(Path(winner_path).read_text())
    handlers = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            body_lines = []
            for stmt in node.body:
                body_lines.extend(
                    range(stmt.lineno, getattr(stmt, "end_lineno", stmt.lineno) + 1)
                )
            handlers.append(sorted(set(body_lines)))
    return handlers


def measure_coverage(winner_path: Path, driver: Callable[[type], None]) -> dict:
    """Run `driver(loaded_class)` under coverage.py(branch=True) and summarise.

    Coverage is started BEFORE the winner is exec'd so that definition-time
    lines (imports, class/def headers) are traced too; otherwise they would be
    miscounted as dead.
    """
    import coverage

    winner_path = Path(winner_path).resolve()
    cov = coverage.Coverage(branch=True, include=[str(winner_path)])
    cov.start()
    try:
        cls = load_traced_class(winner_path)   # exec'd while coverage is active
        driver(cls)
    finally:
        cov.stop()

    # analysis2 -> (filename, statements, excluded, missing, missing_formatted)
    _, statements, _excluded, missing, _ = cov.analysis2(str(winner_path))
    n_stmt = len(statements)
    n_missing = len(missing)
    pct_lines = 100.0 * (n_stmt - n_missing) / n_stmt if n_stmt else 100.0

    # Branch totals from the JSON report summary (public, stable API).
    import json
    import os
    import tempfile

    fd, tmp = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        cov.json_report(outfile=tmp)
        data = json.loads(Path(tmp).read_text())
    finally:
        os.unlink(tmp)
    fkey = next((k for k in data["files"] if Path(k).name == winner_path.name), None)
    summ = data["files"][fkey]["summary"] if fkey else {}
    n_branches = summ.get("num_branches", 0)
    covered_branches = summ.get("covered_branches", 0)
    pct_branches = (100.0 * covered_branches / n_branches) if n_branches else 100.0

    # Which except handlers never fired (no body line executed).
    handlers = _except_handler_lines(winner_path)
    missing_set = set(missing)
    triggered = sum(1 for body in handlers if any(ln not in missing_set for ln in body))

    return {
        "winner": winner_path.stem,
        "n_statements": n_stmt,
        "n_missing": n_missing,
        "pct_lines": round(pct_lines, 1),
        "n_branches": n_branches,
        "pct_branches": round(pct_branches, 1),
        "dead_lines": sorted(missing),
        "dead_constructs": map_missing_to_ast(winner_path, list(missing)),
        "except_handlers_total": len(handlers),
        "except_handlers_triggered": triggered,
    }
