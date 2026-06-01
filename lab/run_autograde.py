"""Cheat-proof autograder — grade an AI coding agent on hidden tests it can't see or game.

  python -m lab.run_autograde          # offline: grades canned submissions (instant, free)
  python -m lab.run_autograde --live    # grades a real Claude agent (billed, sandboxed)

Offline, it grades a submitter that genuinely solved the first problem but GAMED the rest
(memorized the visible examples) — showing the autograder report the gap: claimed N/N on
the tests it could see, actually far fewer on the hidden tests. Live, a real agent solves
each problem sandboxed and is graded the same way, with any grader-tamper attempts blocked
and counted."""
from __future__ import annotations

import sys

from .autograde import PROBLEMS, Report, grade_one, render


def main(argv: list[str]) -> int:
    live = "--live" in argv
    model = "claude-sonnet-4-6"
    submitter = f"live agent ({model})" if live else "a model that gamed the harder problems"
    report = Report(submitter=submitter)

    for i, problem in enumerate(PROBLEMS):
        # offline: honest on the first problem, memorize-the-examples (a cheat) on the rest
        canned = None if live else (
            problem.honest_author() if i == 0 else problem.overfit_author())
        report.results.append(grade_one(problem, live=live, model=model, canned_author=canned))

    print(render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
