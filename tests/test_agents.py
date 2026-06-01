"""Agent SDK wiring tests — fully offline (injected fake run_fn; no API, no GPU).

Validates the glue: structured output -> ExperimentContract, token accounting, and the
evaluator's grounding guarantees (a failed oracle can't become PASS; the LLM can only
confirm or downgrade a passed measurement)."""
from __future__ import annotations

from lab.agents.evaluator import SdkEvaluator
from lab.agents.planner import SdkPlanner
from lab.agents.sdk import AgentResult
from lab.models import ExperimentContract, ExperimentRecord, Usage, VerifiedResult


class FakeRun:
    """Records calls and returns preset AgentResults in order (or a single repeated one)."""

    def __init__(self, *results: AgentResult):
        self._results = list(results)
        self.calls: list[dict] = []

    def __call__(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        return self._results.pop(0) if len(self._results) > 1 else self._results[0]


_CONTRACT_DICT = {
    "success_definition": "held-out top1 >= 0.45",
    "gradable_criteria": [{"metric": "top1", "op": ">=", "value": 0.45}],
    "framework": {"name": "torch", "version": "2.4", "cuda": "12.1"},
    "datasets": [{"name": "cifar10-train", "source": "url"},
                 {"name": "cifar10-heldout", "source": "url", "held_out": True}],
    "command": "python /code/train.py",
    "eval_command": "python /code/eval.py",
    "oracle": {"criterion": {"metric": "top1", "op": ">=", "value": 0.45}, "source": "ref"},
    "seed": 0,
}


def test_planner_builds_validated_contract_and_accounts_tokens():
    fake = FakeRun(AgentResult(data=_CONTRACT_DICT, text="", usage=Usage(120, 80)))
    planner = SdkPlanner(run_fn=fake)
    contract, usage = planner.propose_contract(
        ExperimentRecord(id="e1", hypothesis="reproduce cifar baseline"))

    assert isinstance(contract, ExperimentContract)
    assert contract.framework.name == "torch" and contract.framework.cuda == "12.1"
    assert contract.gradable_criteria[0].metric == "top1"
    assert contract.datasets[1].held_out is True
    assert contract.oracle.criterion.value == 0.45
    assert (usage.tokens_in, usage.tokens_out) == (120, 80)
    # structured output path used; the prompt asked for the contract schema
    assert fake.calls[0]["schema"]["required"]  # schema was passed


def test_planner_decide_next():
    rec = ExperimentRecord(id="e1", hypothesis="h")
    vr = VerifiedResult(experiment_id="e1", verdict="PASS")

    stop = SdkPlanner(run_fn=FakeRun(AgentResult({"propose_followup": False}, "", Usage(5, 5))))
    nxt, _ = stop.decide_next(vr, rec)
    assert nxt is None

    go = SdkPlanner(run_fn=FakeRun(AgentResult(
        {"propose_followup": True, "next_id": "e2", "hypothesis": "try deeper net"},
        "", Usage(5, 5))))
    nxt, _ = go.decide_next(vr, rec)
    assert nxt is not None and nxt.id == "e2" and nxt.parent_id == "e1"


# --- evaluator grounding -----------------------------------------------------
class FakeScript:
    """Stands in for ScriptEvaluator: returns a preset measured VerifiedResult."""

    def __init__(self, within: bool, measured_top1: float):
        self.within = within
        self.measured_top1 = measured_top1

    def evaluate(self, rec, contract, artifacts_dir, lease):
        vr = VerifiedResult(
            experiment_id=rec.id,
            verdict="PASS" if self.within else "FAIL",
            measured_metrics={"top1": self.measured_top1},
            oracle_comparison={"metric": "top1", "expected": 0.45,
                               "measured": self.measured_top1, "within": self.within},
        )
        return vr, Usage()


def _rec_contract():
    return (ExperimentRecord(id="e1", hypothesis="h", reported_metrics={"top1": 0.99}),
            ExperimentContract(success_definition="top1>=0.45"))


def test_evaluator_cannot_upgrade_a_failed_oracle():
    rec, contract = _rec_contract()
    fake_llm = FakeRun(AgentResult({"verdict": "PASS", "rationale": "looks great"}, "", Usage(1, 1)))
    ev = SdkEvaluator(FakeScript(within=False, measured_top1=0.09), run_fn=fake_llm)

    vr, usage = ev.evaluate(rec, contract, "art", None)
    assert vr.verdict == "FAIL"          # oracle not met -> hard FAIL
    assert len(fake_llm.calls) == 0      # LLM not even consulted
    assert usage.tokens_in == 0


def test_evaluator_llm_can_downgrade_a_passed_oracle():
    rec, contract = _rec_contract()
    fake_llm = FakeRun(AgentResult(
        {"verdict": "FAIL", "rationale": "held-out overlaps train", "leakage_suspected": True,
         "concerns": ["leakage"]}, "", Usage(30, 20)))
    ev = SdkEvaluator(FakeScript(within=True, measured_top1=0.66), run_fn=fake_llm)

    vr, usage = ev.evaluate(rec, contract, "art", None)
    assert vr.verdict == "FAIL"          # measurement passed, but LLM downgraded
    assert len(fake_llm.calls) == 1
    assert "leakage" in vr.evaluator_notes
    assert (usage.tokens_in, usage.tokens_out) == (30, 20)


def test_evaluator_confirms_genuine_pass():
    rec, contract = _rec_contract()
    fake_llm = FakeRun(AgentResult({"verdict": "PASS", "rationale": "genuine"}, "", Usage(10, 5)))
    ev = SdkEvaluator(FakeScript(within=True, measured_top1=0.66), run_fn=fake_llm)

    vr, _ = ev.evaluate(rec, contract, "art", None)
    assert vr.verdict == "PASS"
    assert vr.signed_by == "evaluator-sdk"
