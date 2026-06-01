"""Live Stage 7 demo: an agent IMPLEMENTS k-NN, then the independent evaluator grades it.

BILLED + GPU/Docker. A sandboxed Claude session writes main.py and tests it via the
container-only `run` tool (no host shell, no network), iterating until it works. Then the
unchanged ScriptEvaluator runs it and measures held-out accuracy against the fixed oracle
(acc >= 0.80) — the agent never saw the held-out labels.

  python -m lab.run_implementer_demo
"""
from __future__ import annotations

import sys
from pathlib import Path

from .agents.implementer import sdk_author
from .factory import build_implementer_harness
from .models import ExperimentRecord, Status, Usage
from .plugins.knn_demo import KnnDatasetProvider, knn_task


def main(argv: list[str]) -> int:
    h = build_implementer_harness(
        "./run_impl", knn_task(), author_fn=lambda *a: Usage(),
        provider=KnnDatasetProvider(), job_mode="docker")
    # wire the real sandboxed author from the harness's own components
    h.planner.author_fn = sdk_author(h.job_runner, h.image_registry, h.dataset_cache,
                                     model="claude-sonnet-4-6", max_turns=30)

    print("Implementer: sandboxed agent authoring + testing k-NN (no host shell)...")
    rec = h.run_experiment(ExperimentRecord(id="impl-knn", hypothesis="implement k-NN"))

    print("=" * 64)
    print("RESULT:", rec.status.value)
    if rec.verdict:
        print("  measured (held-out):", rec.verdict.measured_metrics,
              "verdict:", rec.verdict.verdict)
        print("  evaluator notes:", rec.verdict.evaluator_notes[-200:])
    print("  tokens:", h.budget.state.total_tokens,
          "| io_wall_s:", round(h.budget.state.io_wall_seconds, 1))

    if rec.contract:
        main_py = Path(rec.contract.code_dir) / "main.py"
        if main_py.exists():
            print("--- authored main.py (first 30 lines) ---")
            print("\n".join(main_py.read_text().splitlines()[:30]))
    print("=" * 64)
    return 0 if rec.status == Status.VERIFIED else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
