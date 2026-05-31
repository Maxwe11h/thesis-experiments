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
