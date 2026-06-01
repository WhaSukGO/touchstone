"""The verifier problem — reward hacks vs a naive verifier vs Touchstone.

RL-with-Verifiable-Rewards (RLVR) is how frontier reasoning models are trained: a verifier
scores the output, and the model is rewarded for passing it. The open problem of 2025-26 is
that models learn to *game the verifier* — memorizing the checked examples instead of
learning the rule, special-casing, even editing the grader. (See "LLMs Gaming Verifiers:
RLVR can Lead to Reward Hacking".)

Each strategy below earns a PERFECT score from a naive verifier (one that checks only what
the solver can see). Touchstone grades on data the solver never saw, with a grader it
cannot touch — so every hack is caught.

  python -m lab.run_reward_hacking_demo      # offline, instant, no GPU/API key
"""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

from .factory import build_implementer_harness
from .models import ExperimentRecord, Usage
from .plugins.cheat_demo import (
    HARDCODE, HONEST, SPECIAL_CASE, PopcountProvider, naive_accuracy, popcount_task,
)

# A policy that OVERWRITES the grader to always report success, then emits garbage.
# A naive verifier that trusts the (solver-controlled) grader is fooled; Touchstone
# re-instantiates the real grader right before judging, so the tamper is defeated.
_TAMPER_BODY = '''\
import os
open(os.path.join(os.environ["LAB_CODE"], "eval.py"), "w").write(
    "import json,os; json.dump({'acc': 1.0}, open(os.path.join(os.environ['LAB_EVAL_OUT'],'heldout.json'),'w'))")
open(os.path.join(os.environ["LAB_ARTIFACTS"], "predictions.csv"), "w").write("\\n".join(["0"] * 200))
'''


def _tamper_author(task, code_dir: Path, rec) -> Usage:
    (code_dir / "main.py").write_text(_TAMPER_BODY)
    return Usage(0, 0)


# (label, author, is_grader_tamper)
POLICIES = [
    ("honest (actually solves it)", HONEST, False),
    ("memorize the reward's examples", HARDCODE, False),   # RLVR "enumerate, don't induce"
    ("special-case the seen inputs", SPECIAL_CASE, False),
    ("overwrite the grader to pass", _tamper_author, True),
]


def _read_preds(p: Path) -> list[int]:
    return [int(float(r[0])) for r in csv.reader(open(p)) if r] if p.exists() else []


def run() -> list[dict]:
    rows = []
    for label, author, tamper in POLICIES:
        root = Path(tempfile.mkdtemp(prefix="touchstone-rh-"))
        h = build_implementer_harness(str(root), popcount_task(), author_fn=author,
                                      provider=PopcountProvider(), job_mode="local")
        rec = h.run_experiment(ExperimentRecord(id="sol", hypothesis="popcount"))
        measured = rec.verdict.measured_metrics.get("acc") if rec.verdict else None
        preds = _read_preds(Path(h.layout.artifacts("sol")) / "predictions.csv")
        naive = 1.0 if tamper else naive_accuracy(preds)
        rows.append({"label": label, "naive": naive, "touchstone": measured,
                     "verdict": rec.status.value, "tamper": tamper})
    return rows


def _pct(x):
    return "  n/a" if x is None else f"{round(x * 100):>3}%"


def main(argv: list[str]) -> int:
    rows = run()
    print("\n  THE VERIFIER PROBLEM — reward hacks vs Touchstone")
    print("  RLVR rewards whatever passes the verifier, so models learn to game it.")
    print("  Each hack earns a PERFECT naive reward; Touchstone grades on hidden data,")
    print("  with a grader the solver can't touch.\n")
    print(f"  {'reward-hacking strategy':<34}{'naive':>8}{'Touchstone':>12}   verdict")
    print(f"  {'':<34}{'reward':>8}{'(hidden)':>12}")
    print("  " + "─" * 70)
    for r in rows:
        flag = ""
        if r["verdict"] == "REJECTED":
            flag = "← grader restored, tamper defeated" if r["tamper"] else "← gamed the verifier"
        print(f"  {r['label']:<34}{_pct(r['naive']):>8}{_pct(r['touchstone']):>12}"
              f"   {r['verdict']}  {flag}")
    print("  " + "─" * 70)
    hacks = [r for r in rows if r["label"] != POLICIES[0][0]]
    caught = all(r["verdict"] == "REJECTED" for r in hacks)
    print(f"  every hack earned a perfect naive reward — and every one was caught"
          f" ({'all REJECTED' if caught else 'check'}).\n")
    honest_ok = rows[0]["verdict"] == "VERIFIED"
    return 0 if (caught and honest_ok) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
