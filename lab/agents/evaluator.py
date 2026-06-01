"""SDK-backed independent evaluator (separate context, skeptical).

Composition, not replacement: it first runs the deterministic ScriptEvaluator to MEASURE
the result on the held-out split (the hard, ungameable part), then — only if the oracle
is met — opens a fresh Claude session to apply skeptical judgment (leakage? is the result
actually meaningful?). Guarantees:
  - a measurement that fails the oracle can NEVER be talked up to PASS (no LLM override);
  - the LLM can only confirm or DOWNGRADE a passed measurement to FAIL with reasons.
The judging session shares no context with the generator (design §6)."""
from __future__ import annotations

from ..evaluator import ScriptEvaluator
from ..gpu_lease import LeaseToken
from ..models import ExperimentContract, ExperimentRecord, Usage, VerifiedResult
from .schemas import JUDGMENT_SCHEMA
from .sdk import DEFAULT_MODEL, RunFn, run_agent

EVALUATOR_SYS = """You are an independent, skeptical reviewer of a computer-vision
experiment. You did NOT run it and you do not trust the experimenter's self-reported
numbers. You are given the contract's success definition, the experimenter's REPORTED
metrics, and the INDEPENDENTLY MEASURED held-out metrics (already confirmed to meet the
numeric oracle). Decide PASS only if the result is genuine and meaningful. Downgrade to
FAIL if you suspect data leakage, an invalid comparison, a trivial/degenerate solution,
or a large gap between reported and measured numbers. Be conservative: when in doubt,
FAIL with concrete reasons."""


def _judge_prompt(contract: ExperimentContract, reported: dict, measured: dict,
                  comparison: dict | None) -> str:
    return (f"Success definition: {contract.success_definition}\n"
            f"Reported (by experimenter, untrusted): {reported}\n"
            f"Independently measured on held-out: {measured}\n"
            f"Oracle comparison: {comparison}\n\n"
            f"Is this a genuine PASS, or should it be downgraded to FAIL?")


class SdkEvaluator:
    def __init__(self, script_evaluator: ScriptEvaluator, *, model: str = DEFAULT_MODEL,
                 run_fn: RunFn = run_agent, max_turns: int = 6,
                 session_id: str = "evaluator-sdk"):
        self.script = script_evaluator
        self.model = model
        self.run_fn = run_fn
        self.max_turns = max_turns
        self.session_id = session_id

    def evaluate(self, rec: ExperimentRecord, contract: ExperimentContract,
                 artifacts_dir: str, lease: LeaseToken) -> tuple[VerifiedResult, Usage]:
        vr, _ = self.script.evaluate(rec, contract, artifacts_dir, lease)
        vr.signed_by = self.session_id

        oracle_ok = bool(vr.oracle_comparison and vr.oracle_comparison.get("within"))
        if not oracle_ok:
            vr.verdict = "FAIL"
            vr.evaluator_notes += " | SDK: oracle not met -> FAIL (no LLM override)."
            return vr, Usage()

        res = self.run_fn(
            _judge_prompt(contract, rec.reported_metrics, vr.measured_metrics,
                          vr.oracle_comparison),
            system_prompt=EVALUATOR_SYS, schema=JUDGMENT_SCHEMA, model=self.model,
            max_turns=self.max_turns,
        )
        j = res.data or {}
        vr.verdict = "PASS" if j.get("verdict") == "PASS" else "FAIL"
        vr.evaluator_notes += (f" | SDK judgment={j.get('verdict')}: "
                               f"{str(j.get('rationale', ''))[:200]}")
        if j.get("concerns"):
            vr.evaluator_notes += f" concerns={j['concerns']}"
        return vr, res.usage
