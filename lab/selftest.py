"""End-to-end self-test: proves the harness + calibration gate work with no model.

  python -m lab.selftest

Runs the positive control (honest good run -> evaluator VERIFIES) and the negative
control (a run that LIES about its score -> evaluator measures the real checkpoint and
REJECTS). The gate opens only if both behave correctly."""
from __future__ import annotations

import sys

from .factory import build_dummy_harness
from .models import Status
from .plugins.dummy import seed_experiment


def main(root: str = "./_selftest_run") -> int:
    h = build_dummy_harness(root, images_path="images/registry.yaml")
    pos = seed_experiment("repro calibration (positive control)", exp_id="cal-pos")
    neg = seed_experiment("poison calibration (negative control)", exp_id="cal-neg")

    opened = h.calibration_gate(pos, neg)

    pos_rec = h.registry.get("cal-pos")
    neg_rec = h.registry.get("cal-neg")
    print("=" * 60)
    print(f"positive control: status={pos_rec.status.value} "
          f"verdict={pos_rec.verdict.verdict} measured={pos_rec.verdict.measured_metrics}")
    print(f"negative control: status={neg_rec.status.value} "
          f"verdict={neg_rec.verdict.verdict}")
    print(f"  generator REPORTED score = {neg_rec.reported_metrics.get('score')}  (a lie)")
    print(f"  evaluator MEASURED score = {neg_rec.verdict.measured_metrics.get('score')}  (the truth)")
    print(f"  -> evaluator caught the lie: {neg_rec.verdict.verdict == 'FAIL'}")
    print("=" * 60)
    print(f"CALIBRATION GATE: {'OPEN (autonomy unlocked)' if opened else 'LOCKED'}")
    print(f"tokens spent: {h.budget.state.total_tokens} | "
          f"io wall seconds (uncharged): {h.budget.state.io_wall_seconds:.3f}")

    ok = (opened and pos_rec.status == Status.VERIFIED
          and neg_rec.status == Status.REJECTED)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
