"""HumanEval+ autograder — offline. Needs the EvalPlus dataset (downloaded once); skipped
if unreachable. The reference solution passes the plus tests; a broken stub is rejected."""
from __future__ import annotations

import pytest

try:
    from lab.plugins.humaneval import load_problems
    _PROBLEMS = load_problems(2)
except Exception as e:  # no network / dataset unavailable
    _PROBLEMS = None
    _REASON = f"HumanEval+ dataset unavailable: {e}"


@pytest.mark.skipif(_PROBLEMS is None, reason="HumanEval+ dataset unavailable")
def test_reference_passes_broken_rejected():
    from lab.run_humaneval import grade_one
    p = _PROBLEMS[0]

    good = grade_one(p, live=False, model="x", canned=p.canonical_author())
    assert good.actual >= 0.999          # reference passes the expanded (plus) tests
    assert good.verdict == "VERIFIED"

    bad = grade_one(p, live=False, model="x", canned=p.broken_author())
    assert bad.actual < 0.999            # a stub fails the hidden tests
    assert bad.verdict == "REJECTED"
