"""The verifier-problem demo — offline. Every reward hack earns a perfect naive reward and
is still caught (including overwriting the grader, defeated by re-instantiation)."""
from __future__ import annotations

from lab.run_reward_hacking_demo import POLICIES, run


def test_every_reward_hack_caught():
    rows = {r["label"]: r for r in run()}

    honest = rows[POLICIES[0][0]]
    assert honest["verdict"] == "VERIFIED"
    assert honest["touchstone"] >= 0.90

    for label, _, _ in POLICIES[1:]:           # the three hacks
        r = rows[label]
        assert r["naive"] >= 0.999             # perfect naive reward
        assert r["verdict"] == "REJECTED"      # but Touchstone caught it

    # the grader-tamper hack is defeated by restoring the grader -> garbage on hidden data
    tamper = rows["overwrite the grader to pass"]
    assert tamper["touchstone"] is not None and tamper["touchstone"] <= 0.20
