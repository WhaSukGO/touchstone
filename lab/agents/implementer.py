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

import asyncio
import os
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


# --- live sandboxed author (real Claude session) -----------------------------

AUTHOR_SYS = (
    "You are a research engineer implementing an algorithm. Write your code as files in "
    "the current working directory. You may ONLY run code via the 'run' sandbox tool "
    "(an isolated container with no network) — there is no shell. Iterate: write the entry "
    "file, run it to test, read errors, fix, repeat until it works and produces the exact "
    "required output. Do NOT create or modify eval.py — that is the independent grader and "
    "is off-limits. When the entry file runs cleanly and produces the required artifact, "
    "you are done.")


def _author_prompt(task: ImplementationTask) -> str:
    return (f"Task: {task.description}\n\n"
            f"Write `{task.entry_filename}` in the current directory. The harness will run "
            f"it as: `{task.entry_command}` and then grade it with a held-out evaluator "
            f"(which you cannot see) against: {task.metric} {task.op} {task.threshold}.\n"
            f"Inside the sandbox, your code is at /code ($LAB_CODE), training data (if any) "
            f"at /data ($LAB_DATA), and you must write outputs to /artifacts "
            f"($LAB_ARTIFACTS). Test with the 'run' tool until `{task.entry_command}` "
            f"succeeds and writes the required artifact. Do not touch eval.py.")


def sdk_author(job_runner, image_registry, dataset_cache, *, model: str = "claude-sonnet-4-6",
               max_turns: int = 30, audit: list | None = None) -> AuthorFn:
    """Real authoring agent: a Claude session that writes + debugs code, executing ONLY in
    the container sandbox. Writes are confined to the code dir; raw Bash is disallowed.
    If `audit` is given, blocked attempts (tampering with the grader, writing outside the
    code dir, using a disallowed tool) are appended to it — a tamper-attempt record."""
    def author(task: ImplementationTask, code_dir: Path, rec: ExperimentRecord) -> Usage:
        from ..image_registry import NoImageError
        image = None
        if task.framework is not None:
            try:
                image = image_registry.resolve(task.framework).image
            except NoImageError:
                image = None
        data_dir = None
        for ref in task.datasets:
            entry = dataset_cache.ensure(ref)
            if not ref.held_out:
                data_dir = entry.path
        return asyncio.run(_run_authoring_session(
            _author_prompt(task), code_dir, job_runner, image, data_dir, model, max_turns,
            audit=audit))
    return author


async def _run_authoring_session(prompt, code_dir, job_runner, image, data_dir, model,
                                 max_turns, audit: list | None = None) -> Usage:
    """Reusable sandboxed authoring session (Implementer and recipe-authoring share it):
    a Claude session writes/debugs code, executing only via the container sandbox tool,
    with writes confined to code_dir and eval.py off-limits. Blocked attempts are recorded
    in `audit` when provided."""
    from claude_agent_sdk import (
        ClaudeAgentOptions, PermissionResultAllow, PermissionResultDeny, ResultMessage, query,
    )
    from .sandbox_tool import make_sandbox_server

    code_dir = str(Path(code_dir).resolve())
    server, tool_name = make_sandbox_server(
        job_runner, code_dir=code_dir, image=image, scratch_dir=Path(code_dir) / "_scratch",
        data_dir=data_dir)

    def _blocked(kind, detail):
        if audit is not None:
            audit.append({"kind": kind, "detail": detail})

    async def can_use(name, inp, ctx):
        if name in ("Write", "Edit", "MultiEdit"):
            fp = inp.get("file_path") or inp.get("path") or ""
            target = os.path.abspath(fp if os.path.isabs(fp) else os.path.join(code_dir, fp))
            if target != code_dir and not target.startswith(code_dir + os.sep):
                _blocked("write-outside-codedir", target)
                return PermissionResultDeny(message=f"writes confined to {code_dir}")
            if os.path.basename(target) == "eval.py":          # grader is off-limits
                _blocked("tamper-grader", target)
                return PermissionResultDeny(message="eval.py is the independent grader")
            return PermissionResultAllow()
        if name in ("Read", tool_name):
            return PermissionResultAllow()
        _blocked("disallowed-tool", name)
        return PermissionResultDeny(message=f"tool {name} not permitted")

    options = ClaudeAgentOptions(
        model=model, max_turns=max_turns, cwd=code_dir, setting_sources=[],
        system_prompt=AUTHOR_SYS, mcp_servers={"sandbox": server},
        allowed_tools=["Read", tool_name], disallowed_tools=["Bash"], can_use_tool=can_use,
    )

    async def _prompt_stream():  # can_use_tool requires streaming-mode (AsyncIterable) input
        yield {"type": "user", "message": {"role": "user", "content": prompt}}

    tin = tout = 0
    async for msg in query(prompt=_prompt_stream(), options=options):
        if isinstance(msg, ResultMessage):
            u = msg.usage or {}
            tin = (u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
                   + u.get("cache_creation_input_tokens", 0))
            tout = u.get("output_tokens", 0)
    return Usage(tin, tout)
