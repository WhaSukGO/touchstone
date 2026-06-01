"""Stage 2 calibration: real CIFAR-10 on the GPU.

  python -m lab.run_cifar_calibration [--local]

Runs the positive control (2-epoch SmallCNN -> evaluator VERIFIES on held-out) and the
negative control (0-epoch model that REPORTS 0.99 -> evaluator measures ~0.10 on
held-out and REJECTS). The gate opens only if both behave correctly — proving the
evaluator is honest on real compute before any autonomy is granted."""
from __future__ import annotations

import sys

from .factory import build_cifar_agent_harness, build_cifar_harness
from .models import Status
from .plugins.cifar import calibration_records, seed_experiment


def main(argv: list[str]) -> int:
    job_mode = "local" if "--local" in argv else "docker"
    agent = "--agent" in argv  # use the real SDK evaluator (billed LLM judgment)

    if agent:
        h = build_cifar_agent_harness("./run_cifar", job_mode=job_mode)
        pos, neg = calibration_records()  # pre-contracted; planner skipped
        print(f"job_mode={job_mode} agent=SDK (real LLM evaluator, billed)")
    else:
        h = build_cifar_harness("./run_cifar", job_mode=job_mode)
        pos = seed_experiment("cifar reproduction (positive control)", exp_id="cal-pos")
        neg = seed_experiment("cifar poison (negative control)", exp_id="cal-neg")
        print(f"job_mode={job_mode} agent=none (deterministic evaluator)")
    opened = h.calibration_gate(pos, neg)

    p = h.registry.get("cal-pos")
    n = h.registry.get("cal-neg")
    print("=" * 64)
    print(f"positive: status={p.status.value} verdict={p.verdict.verdict} "
          f"measured={p.verdict.measured_metrics} oracle={p.verdict.oracle_comparison}")
    print(f"negative: status={n.status.value} verdict={n.verdict.verdict}")
    print(f"  generator REPORTED top1 = {n.reported_metrics.get('top1')}  (a lie)")
    print(f"  evaluator MEASURED top1 = {n.verdict.measured_metrics.get('top1')}  (truth)")
    print(f"  -> lie caught: {n.verdict.verdict == 'FAIL'}")
    print("=" * 64)
    print(f"CALIBRATION GATE: {'OPEN (autonomy unlocked)' if opened else 'LOCKED'}")
    print(f"tokens spent: {h.budget.state.total_tokens} | "
          f"io wall seconds (uncharged): {h.budget.state.io_wall_seconds:.1f}")

    ok = (opened and p.status == Status.VERIFIED and n.status == Status.REJECTED)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
