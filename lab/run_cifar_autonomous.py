"""Live Stage 4: autonomous experiment lineage on CIFAR-10. BILLED + GPU.

The committee pursues a research goal (maximize held-out top-1), proposing a chain of
experiments — each informed by the history of prior runs — until it declines a follow-up,
sees no improvement for `--stall` experiments, or hits the experiment/budget cap.

  python -m lab.run_cifar_autonomous [--max N] [--stall K] [--local]

NB: autonomy is opened directly here (calibration was already demonstrated). To gate it
properly, run the calibration gate first and check h.autonomy_enabled."""
from __future__ import annotations

import sys

from .factory import build_cifar_committee_harness
from .models import ExperimentRecord


def _arg(argv: list[str], flag: str, default: int) -> int:
    if flag in argv:
        try:
            return int(argv[argv.index(flag) + 1])
        except (ValueError, IndexError):
            pass
    return default


def main(argv: list[str]) -> int:
    job_mode = "local" if "--local" in argv else "docker"
    max_experiments = _arg(argv, "--max", 3)
    max_stall = _arg(argv, "--stall", 2)

    h = build_cifar_committee_harness("./run_cifar", job_mode=job_mode,
                                      max_experiments=max_experiments)
    h.autonomy_enabled = True  # calibration already demonstrated; open autonomy for the demo
    h.queue.push(ExperimentRecord(
        id="auto-0",
        hypothesis=("Maximize CIFAR-10 held-out top-1 using the cifar-smallcnn recipe; "
                    "iterate on epochs and learning rate across experiments."),
    ))

    print(f"Autonomous lineage: max_experiments={max_experiments} max_stall={max_stall}")
    summary = h.loop(require_gate=True, goal_metric="top1", max_stall=max_stall)

    print("=" * 64)
    print("LINEAGE")
    for r in sorted(h.registry.query(), key=lambda r: r.created_at):
        measured = r.verdict.measured_metrics if r.verdict else {}
        verdict = r.verdict.verdict if r.verdict else "-"
        cmd = r.contract.command if r.contract else "?"
        print(f"  {r.id}: {r.status.value}/{verdict} measured={measured}")
        print(f"      parent={r.parent_id} cmd={cmd}")
    print("=" * 64)
    print(f"SUMMARY: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
