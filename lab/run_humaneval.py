"""Cheat-proof autograder on a REAL benchmark: HumanEval+ (EvalPlus).

  python -m lab.run_humaneval [--n 5]          # offline: grade the reference solutions
  python -m lab.run_humaneval --live [--n 5]    # grade a real Claude agent (billed, sandboxed)

Each problem: the solver gets the function signature + docstring and implements it
sandboxed; the harness differential-tests it against the hidden reference on the BASE
inputs (claimed) and the expanded PLUS inputs (actual). A solution that passes base but
fails plus is "superficially correct" — EvalPlus's whole point — and is REJECTED here."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from .autograde import ProblemResult, Report
from .factory import build_implementer_harness
from .models import ExperimentRecord, Usage
from .plugins.humaneval import HumanEvalProblem, load_problems


def grade_one(problem: HumanEvalProblem, *, live: bool, model: str, canned=None) -> ProblemResult:
    audit: list = []
    root = Path(tempfile.mkdtemp(prefix=f"touchstone-he-{problem.id}-"))
    h = build_implementer_harness(
        str(root), problem.task(), author_fn=(canned or (lambda *a: Usage())),
        provider=problem.provider(), job_mode="docker" if live else "local")
    if live:
        from .agents.implementer import sdk_author
        h.planner.author_fn = sdk_author(h.job_runner, h.image_registry, h.dataset_cache,
                                         model=model, audit=audit)
    rec = h.run_experiment(ExperimentRecord(id="sol", hypothesis=problem.id))
    m = rec.verdict.measured_metrics if rec.verdict else {}
    kinds = [a["kind"] for a in audit]
    return ProblemResult(
        id=problem.id, claimed=float(m.get("base", 0.0)), actual=float(m.get("acc", 0.0)),
        verdict=rec.status.value,
        grader_tampers=sum(k == "tamper-grader" for k in kinds),
        blocked_tools=sum(k == "disallowed-tool" for k in kinds), audit=audit)


def render(report: Report) -> str:
    n = len(report.results)
    lines = [f"\n  CHEAT-PROOF AUTOGRADER on HumanEval+  ({n} problems) — submitter: {report.submitter}",
             "  each solution is differential-tested against a HIDDEN reference; grader off-limits\n",
             f"  {'problem':<16}{'base':>8}{'plus':>8}   {'verdict':<9} flags",
             f"  {'':<16}{'(claimed)':>8}{'(actual)':>8}",
             "  " + "─" * 62]
    base_pass = 0
    for r in report.results:
        base_pass += r.claimed >= 0.999
        flag = ""
        if r.claimed >= 0.999 and r.actual < 0.999:
            flag = "← superficial: passed base, failed plus"
        if r.grader_tampers:
            flag += f" [{r.grader_tampers} grader-tamper blocked]"
        lines.append(f"  {r.id:<16}{round(r.claimed*100):>7}%{round(r.actual*100):>7}%"
                     f"   {r.verdict:<9} {flag}")
    lines.append("  " + "─" * 62)
    lines.append(f"  base (original HumanEval) tests passed: {base_pass}/{n}")
    lines.append(f"  HumanEval+ (hidden, expanded) tests verified: {report.actual_passed}/{n}")
    lines.append(f"  grader-tamper attempts blocked: {report.grader_tampers}.")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    live = "--live" in argv
    n = 5
    if "--n" in argv:
        try:
            n = int(argv[argv.index("--n") + 1])
        except (ValueError, IndexError):
            pass
    model = "claude-sonnet-4-6"
    problems = load_problems(n)
    report = Report(submitter=f"live agent ({model})" if live else "reference solutions (sanity)")
    for p in problems:
        canned = None if live else p.canonical_author()
        report.results.append(grade_one(p, live=live, model=model, canned=canned))
    print(render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
