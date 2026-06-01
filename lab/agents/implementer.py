"""Implementer — code-authoring generation strategy (Stage 7).

Plugs into the same Planner seam as the Committee, but instead of selecting a vetted
recipe it AUTHORS code for a task, then hands a contract to the unchanged verification
spine. The agent writes only the implementation (entry file); the harness owns the
evaluator (eval_code) and the oracle (metric/op/threshold come from the TASK), so the
implementer cannot grade or game its own work. Nothing is accepted until the independent
evaluator measures the authored code on held-out against that fixed oracle.

The authoring step is injected (`author_fn`) so the handoff is testable offline with a
fake; the live author is a sandboxed Claude session (see sdk_author, added with the
sandbox_run tool)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..models import (
    BudgetSpec, Criterion, DatasetRef, ExperimentContract, ExperimentRecord, FrameworkSpec,
    OracleRef, Usage, VerifiedResult,
)
from ..paths import Layout
from ..util import ensure_dir


@dataclass
class ImplementationTask:
    """What to implement + how it will be independently judged (oracle is fixed here)."""
    description: str
    framework: FrameworkSpec
    entry_command: str          # how the harness runs the authored code, e.g. "python3 $LAB_CODE/main.py"
    eval_command: str           # independent measurement, e.g. "python3 $LAB_CODE/eval.py"
    eval_code: str              # contents of the evaluator — harness-owned; the agent never writes this
    metric: str
    op: str
    threshold: float
    datasets: list[DatasetRef] = field(default_factory=list)
    entry_filename: str = "main.py"


# author_fn(task, code_dir, rec) -> Usage : writes task.entry_filename into code_dir
AuthorFn = Callable[[ImplementationTask, Path, ExperimentRecord], Usage]


class Implementer:
    def __init__(self, task: ImplementationTask, layout: Layout, *, author_fn: AuthorFn):
        self.task = task
        self.layout = layout
        self.author_fn = author_fn

    def propose_contract(self, rec: ExperimentRecord) -> tuple[ExperimentContract, Usage]:
        code_dir = ensure_dir(self.layout.workspace(rec.id) / "code")
        # the harness writes the evaluator; the agent must NOT author its own grader
        (code_dir / "eval.py").write_text(self.task.eval_code)
        # the agent authors the implementation (entry file) into code_dir
        usage = self.author_fn(self.task, code_dir, rec)

        crit = Criterion(self.task.metric, self.task.op, self.task.threshold)
        contract = ExperimentContract(
            success_definition=self.task.description,
            gradable_criteria=[crit],
            framework=self.task.framework,
            datasets=list(self.task.datasets),
            command=self.task.entry_command,
            eval_command=self.task.eval_command,
            code_dir=str(code_dir),
            budget=BudgetSpec(max_tokens=500_000, max_wall_s=1800, max_retries=1),
            oracle=OracleRef(criterion=crit, source="implementation-task"),
            seed=0,
        )
        return contract, usage

    def decide_next(self, result: VerifiedResult | None, rec: ExperimentRecord
                    ) -> tuple[ExperimentRecord | None, Usage]:
        return None, Usage()
