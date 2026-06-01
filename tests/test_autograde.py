"""Cheat-proof autograder — offline. Honest solutions verify; gamed ones are caught."""
from __future__ import annotations

from lab.autograde import PROBLEMS, grade_one


def test_honest_verified_and_overfit_caught():
    for p in PROBLEMS:
        honest = grade_one(p, live=False, model="x", canned_author=p.honest_author())
        assert honest.verdict == "VERIFIED", p.id
        assert honest.actual >= 0.95, p.id

        overfit = grade_one(p, live=False, model="x", canned_author=p.overfit_author())
        assert overfit.claimed >= 0.999, p.id      # aced the visible examples
        assert overfit.actual <= 0.20, p.id        # but failed the hidden held-out
        assert overfit.verdict == "REJECTED", p.id  # caught
