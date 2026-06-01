"""SDK-backed planner (the reasoning/generator seat).

Implements the Planner protocol with a real Claude session. Returns a validated
ExperimentContract and the token Usage (so the harness budgets correctly). The SDK call
is injected (run_fn) so the glue is testable without API spend.

NOTE: this is the general capability. Constraining proposals to a vetted menu of
frameworks/datasets (so the autonomous loop can't go off the rails) is the Stage-3 safety
layer; for now the system prompt documents the harness's execution contract and the
harness still fails safely if an image/dataset is unavailable."""
from __future__ import annotations

from ..models import ExperimentContract, ExperimentRecord, Usage, VerifiedResult
from ..serde import from_dict
from .schemas import CONTRACT_SCHEMA, DECIDE_SCHEMA
from .sdk import DEFAULT_MODEL, RunFn, run_agent

PLANNER_SYS = """You are the principal investigator of an autonomous computer-vision
research lab. You design ONE experiment at a time as a precise, gradable contract.

Execution contract (the harness runs your experiment in a Docker container):
- Commands run via `bash -c`. These env vars/paths are provided inside the container:
  LAB_CODE (=/code, read-only reference code), LAB_DATA (=/data, read-only dataset),
  LAB_ARTIFACTS (=/artifacts, write your checkpoint + metrics.json here),
  LAB_EVAL_OUT (=/eval_out, the evaluator writes here).
- `framework` must be a pre-built image the lab supports (e.g. torch 2.4 / cuda 12.1).
- Provide BOTH a `command` (train/produce) and an `eval_command` (independent held-out
  measurement). Mark the held-out dataset with held_out=true; the generator never sees it.
- `gradable_criteria` and `oracle.criterion` make success measurable, not vibes. A result
  that merely 'runs' is NOT success.
Return only the structured contract."""

EVAL_METRIC_NOTE = ("Success must be a measurable held-out metric, not 'it ran'.")


def _propose_prompt(rec: ExperimentRecord) -> str:
    return (f"Design an experiment contract for this research hypothesis:\n\n"
            f"  {rec.hypothesis}\n\n{EVAL_METRIC_NOTE}")


def _decide_prompt(result: VerifiedResult, rec: ExperimentRecord) -> str:
    return (f"The experiment '{rec.id}' ({rec.hypothesis}) finished with verdict "
            f"{result.verdict}. Measured: {result.measured_metrics}. Oracle: "
            f"{result.oracle_comparison}. Decide whether to propose ONE follow-up "
            f"experiment that would advance the research, and if so give a new id and "
            f"hypothesis.")


class SdkPlanner:
    def __init__(self, *, model: str = DEFAULT_MODEL, run_fn: RunFn = run_agent,
                 max_turns: int = 6):
        self.model = model
        self.run_fn = run_fn
        self.max_turns = max_turns

    def propose_contract(self, rec: ExperimentRecord) -> tuple[ExperimentContract, Usage]:
        res = self.run_fn(_propose_prompt(rec), system_prompt=PLANNER_SYS,
                          schema=CONTRACT_SCHEMA, model=self.model, max_turns=self.max_turns)
        contract = from_dict(ExperimentContract, res.data)
        return contract, res.usage

    def decide_next(self, result: VerifiedResult, rec: ExperimentRecord
                    ) -> tuple[ExperimentRecord | None, Usage]:
        res = self.run_fn(_decide_prompt(result, rec), system_prompt=PLANNER_SYS,
                          schema=DECIDE_SCHEMA, model=self.model, max_turns=self.max_turns)
        d = res.data or {}
        if not d.get("propose_followup"):
            return None, res.usage
        nxt = ExperimentRecord(
            id=d.get("next_id") or f"{rec.id}-next",
            hypothesis=d.get("hypothesis", ""),
            parent_id=rec.id, priority=rec.priority,
        )
        return nxt, res.usage
