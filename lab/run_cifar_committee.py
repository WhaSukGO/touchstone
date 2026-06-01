"""Live Stage 3 demo: an expert committee proposes ONE CIFAR experiment (constrained to
the vetted menu), the harness trains it on the GPU, and the skeptical SDK evaluator judges
it on the held-out split. BILLED (real LLM calls). Needs ANTHROPIC_API_KEY + the GPU.

  python -m lab.run_cifar_committee            # docker/GPU
  python -m lab.run_cifar_committee --local    # local job mode (needs torch installed)
"""
from __future__ import annotations

import sys

from .factory import build_cifar_committee_harness
from .models import ExperimentRecord, Status


def main(argv: list[str]) -> int:
    job_mode = "local" if "--local" in argv else "docker"
    h = build_cifar_committee_harness("./run_cifar", job_mode=job_mode)
    rec = ExperimentRecord(
        id="committee-exp-1",
        hypothesis=("Reproduce a CIFAR-10 SmallCNN baseline that clears the 0.45 held-out "
                    "top-1 bar; choose reasonable epochs and learning rate."),
    )

    print("Convening committee (PI + Modeling + Data)...")
    out = h.run_experiment(rec)

    mtg = getattr(h.planner, "last_meeting", {})
    print("=" * 64)
    print("COMMITTEE MEETING")
    print("  recipe :", mtg.get("recipe_id"))
    print("  command:", mtg.get("final_command"))
    for o in mtg.get("opinions", []):
        print(f"  [{o['role']}] overrides={o['overrides']} approve={o['approve']} "
              f"concerns={o['concerns']}")
    print("=" * 64)
    print("RESULT")
    print("  status   :", out.status.value)
    print("  reported :", out.reported_metrics)
    if out.verdict:
        print("  measured :", out.verdict.measured_metrics, "verdict:", out.verdict.verdict)
        print("  notes    :", out.verdict.evaluator_notes[-300:])
    print(f"  tokens   : {h.budget.state.total_tokens} | "
          f"io_wall_s: {h.budget.state.io_wall_seconds:.1f}")
    print("=" * 64)
    return 0 if out.status == Status.VERIFIED else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
