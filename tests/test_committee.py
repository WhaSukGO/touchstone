"""Stage 3 tests — capability menu guardrails + expert committee. Fully offline."""
from __future__ import annotations

import pytest

from lab.agents.committee import Committee
from lab.agents.sdk import AgentResult
from lab.menu import Menu, MenuError, ParamSpec, Proposal, Recipe
from lab.models import DatasetRef, ExperimentRecord, FrameworkSpec, Usage, VerifiedResult


def _recipe() -> Recipe:
    return Recipe(
        id="r1", description="train smallcnn", framework=FrameworkSpec("torch", "2.4", "12.1"),
        code_dir="/code",
        datasets=[DatasetRef("train", "u"), DatasetRef("ho", "u", held_out=True)],
        train_template="LAB_EPOCHS={epochs} LAB_LR={lr} LAB_SEED={seed} python /code/train.py",
        eval_command="python /code/eval.py", metric="top1", threshold=0.45,
        params=[ParamSpec("epochs", "int", low=1, high=20, default=2),
                ParamSpec("lr", "float", low=1e-4, high=1e-2, default=1e-3)],
    )


def _menu() -> Menu:
    return Menu([_recipe()])


# ---- menu guardrails --------------------------------------------------------
def test_paramspec_clamps_and_defaults():
    e = ParamSpec("epochs", "int", low=1, high=20, default=2)
    assert e.coerce(9999) == 20
    assert e.coerce(-5) == 1
    assert e.coerce("garbage") == 2
    assert e.coerce(3.7) == 4
    lr = ParamSpec("lr", "float", low=1e-4, high=1e-2, default=1e-3)
    assert lr.coerce(5.0) == 1e-2
    assert lr.coerce(0) == 1e-4


def test_validate_params_drops_unknown():
    p = _recipe().validate_params({"epochs": 5, "evil": "rm -rf /", "lr": 0.002})
    assert p == {"epochs": 5, "lr": 0.002}     # 'evil' is not a declared param -> dropped


def test_build_contract_command_is_safe_and_bar_is_fixed():
    c = _recipe().build_contract({"epochs": 9999, "evil": "; rm -rf /"}, hypothesis="h")
    assert "LAB_EPOCHS=20" in c.command         # clamped
    assert "rm -rf" not in c.command            # injected key never reaches the shell
    assert c.oracle.criterion.value == 0.45     # fixed oracle bar, not proposer-chosen


def test_menu_rejects_unknown_recipe():
    with pytest.raises(MenuError):
        _menu().build(Proposal(recipe_id="ghost", params={}))


# ---- committee --------------------------------------------------------------
class FakeRun:
    def __init__(self, *results: AgentResult):
        self._r = list(results)
        self.calls: list[dict] = []

    def __call__(self, prompt, **kw):
        self.calls.append({"prompt": prompt, **kw})
        return self._r.pop(0) if len(self._r) > 1 else self._r[0]


def test_committee_builds_menu_constrained_contract_and_sums_tokens():
    fake = FakeRun(
        AgentResult({"recipe_id": "r1", "params": {"epochs": 3, "lr": 0.001},
                     "hypothesis": "baseline"}, "", Usage(100, 50)),               # PI
        AgentResult({"param_overrides": {"epochs": 8}, "concerns": [], "approve": True},
                    "", Usage(40, 20)),                                            # Modeling
        AgentResult({"param_overrides": {}, "concerns": ["verify no leakage"],
                     "approve": True}, "", Usage(30, 10)),                         # Data
    )
    c = Committee(_menu(), run_fn=fake)
    contract, usage = c.propose_contract(ExperimentRecord(id="e1", hypothesis="reproduce"))

    assert "LAB_EPOCHS=8" in contract.command                  # modeling override applied
    assert (usage.tokens_in, usage.tokens_out) == (170, 80)    # summed across the meeting
    assert c.last_meeting["recipe_id"] == "r1"
    assert "verify no leakage" in c.last_meeting["concerns"]
    assert len(fake.calls) == 3                                # PI + 2 experts


def test_committee_clamps_hallucinated_params():
    fake = FakeRun(
        AgentResult({"recipe_id": "r1", "params": {"epochs": 9999}, "hypothesis": "h"},
                    "", Usage(1, 1)),
        AgentResult({"param_overrides": {"epochs": -3, "evil": "; rm -rf /"}, "approve": True},
                    "", Usage(1, 1)),
        AgentResult({"param_overrides": {}, "approve": True}, "", Usage(1, 1)),
    )
    contract, _ = Committee(_menu(), run_fn=fake).propose_contract(
        ExperimentRecord(id="e1", hypothesis="h"))
    assert "LAB_EPOCHS=1" in contract.command   # -3 clamped to low; 9999 overridden then clamped
    assert "rm -rf" not in contract.command


def test_committee_rejects_hallucinated_recipe():
    fake = FakeRun(
        AgentResult({"recipe_id": "ghost", "params": {}}, "", Usage(1, 1)),
        AgentResult({"approve": True}, "", Usage(1, 1)),
        AgentResult({"approve": True}, "", Usage(1, 1)),
    )
    with pytest.raises(MenuError):              # fails safe; harness records FAILED
        Committee(_menu(), run_fn=fake).propose_contract(
            ExperimentRecord(id="e1", hypothesis="h"))


def test_committee_decide_next():
    fake = FakeRun(AgentResult({"propose_followup": True, "next_id": "e2",
                                "hypothesis": "more epochs"}, "", Usage(5, 5)))
    nxt, _ = Committee(_menu(), run_fn=fake).decide_next(
        VerifiedResult(experiment_id="e1", verdict="PASS"),
        ExperimentRecord(id="e1", hypothesis="h"))
    assert nxt is not None and nxt.id == "e2" and nxt.parent_id == "e1"
