"""Unit tests for analysis.phase4.failure_modes."""
from pathlib import Path

import pytest

from analysis.phase4.failure_modes import classify_failure


def _read(fixture_dir: Path, name: str) -> str:
    return (fixture_dir / name).read_text()


@pytest.mark.parametrize('fname,expected', [
    ('code_syntax_error.py', 'code_generation'),
    ('code_no_class.py', 'code_generation'),
    ('code_bad_init.py', 'interface_mismatch'),
    ('code_bad_call.py', 'interface_mismatch'),
    ('code_runtime_error.py', 'runtime_error'),
    ('code_disallowed_import.py', 'import_violation'),
])
def test_classify_known_failures(fixture_dir: Path, fname: str, expected: str):
    code = _read(fixture_dir, fname)
    label, _detail = classify_failure(code)
    assert label == expected


def test_valid_code_returns_none(fixture_dir: Path):
    """Code that passes the compile + smoke pipeline classifies as None."""
    code = _read(fixture_dir, 'code_valid.py')
    label, _ = classify_failure(code)
    assert label is None


def test_classify_returns_detail_string(fixture_dir: Path):
    code = _read(fixture_dir, 'code_runtime_error.py')
    label, detail = classify_failure(code)
    assert label == 'runtime_error'
    assert 'division' in detail.lower() or 'zero' in detail.lower()
