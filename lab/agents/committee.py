"""Expert committee — the "meeting" (Stage 3).

Replaces the single SdkPlanner with a small panel that negotiates the experiment BEFORE
it runs (cf. the harness-design blog's pre-run 'sprint contract'): the PI drafts a
menu-constrained proposal, each expert reviews it (suggesting parameter overrides and
raising concerns), then a deterministic synthesis builds the contract through the Menu —
which validates the recipe and clamps every parameter. The agents only ever select and
parameterize; they cannot author commands, so a bad suggestion can't escape the menu.

Implements the Planner protocol, so it drops into the existing harness unchanged."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..menu import Menu, Proposal
from ..models import ExperimentRecord, Usage, VerifiedResult
from ..notebook import Notebook
from .schemas import DECIDE_SCHEMA, EXPERT_OPINION_SCHEMA, PROPOSAL_SCHEMA
from .sdk import DEFAULT_MODEL, RunFn, run_agent


@dataclass
class Expert:
    role: str
    system_prompt: str


PI = Expert("PI", (
    "You are the principal investigator of a computer-vision research lab. You design ONE "
    "experiment by choosing exactly one recipe from the lab's vetted menu and setting only "
    "its declared parameters. You NEVER invent commands, datasets, code, or new parameters "
    "— only select and parameterize. Frame a clear, testable hypothesis."))

MODELING = Expert("Modeling", (
    "You are the modeling expert. Given a draft proposal, suggest parameter overrides "
    "(only the recipe's declared params, within their stated ranges) that you believe will "
    "best meet the success metric. Raise concerns if the draft looks weak. Approve when the "
    "configuration is sound."))

DATA = Expert("Data", (
    "You are the data expert. Review the dataset choice and held-out split for leakage or "
    "validity problems. You rarely change hyperparameters; your job is to raise concrete "
    "data concerns and approve or reject the plan."))

DEFAULT_EXPERTS = [MODELING, DATA]


def _draft_prompt(menu: Menu, rec: ExperimentRecord) -> str:
    return (f"{menu.describe()}\n\nResearch goal:\n  {rec.hypothesis}\n\n"
            f"Propose an experiment: choose a recipe_id, set its params, and state a "
            f"testable hypothesis. Success must be the recipe's fixed held-out metric.")


def _review_prompt(menu: Menu, recipe_id: str, params: dict, hypothesis: str,
                   expert: Expert) -> str:
    return (f"{menu.describe()}\n\nDraft proposal:\n  recipe_id={recipe_id}\n  "
            f"params={params}\n  hypothesis={hypothesis}\n\nAs the {expert.role} expert, "
            f"give param_overrides (only declared params, within range), list concerns, "
            f"and approve true/false.")


def _decide_prompt(menu: Menu, result: VerifiedResult, rec: ExperimentRecord) -> str:
    return (f"{menu.describe()}\n\nExperiment '{rec.id}' ({rec.hypothesis}) finished with "
            f"verdict {result.verdict}; measured {result.measured_metrics}, oracle "
            f"{result.oracle_comparison}. Propose ONE follow-up experiment (new id + "
            f"hypothesis) if it would advance the research, else decline.")


class Committee:
    """Planner protocol via a menu-constrained multi-expert meeting."""

    def __init__(self, menu: Menu, *, model: str = DEFAULT_MODEL, run_fn: RunFn = run_agent,
                 experts: list[Expert] | None = None, pi: Expert = PI, max_turns: int = 6,
                 notebook: Notebook | None = None):
        self.menu = menu
        self.model = model
        self.run_fn = run_fn
        self.pi = pi
        self.experts = DEFAULT_EXPERTS if experts is None else experts
        self.max_turns = max_turns
        self.notebook = notebook
        self.last_meeting: dict = {}

    def propose_contract(self, rec: ExperimentRecord):
        tin = tout = 0

        draft_res = self.run_fn(_draft_prompt(self.menu, rec), system_prompt=self.pi.system_prompt,
                                schema=PROPOSAL_SCHEMA, model=self.model, max_turns=self.max_turns)
        tin += draft_res.usage.tokens_in
        tout += draft_res.usage.tokens_out
        draft = draft_res.data or {}
        recipe_id = draft.get("recipe_id") or self.menu.ids()[0]
        params = dict(draft.get("params") or {})
        hypothesis = draft.get("hypothesis") or rec.hypothesis

        opinions = []
        for ex in self.experts:
            r = self.run_fn(_review_prompt(self.menu, recipe_id, params, hypothesis, ex),
                            system_prompt=ex.system_prompt, schema=EXPERT_OPINION_SCHEMA,
                            model=self.model, max_turns=self.max_turns)
            tin += r.usage.tokens_in
            tout += r.usage.tokens_out
            op = r.data or {}
            overrides = op.get("param_overrides") or {}
            params.update(overrides)  # raw values; the recipe validates/clamps at build time
            opinions.append({"role": ex.role, "overrides": overrides,
                             "concerns": op.get("concerns", []), "approve": op.get("approve"),
                             "rationale": op.get("rationale", "")})

        proposal = Proposal(recipe_id=recipe_id, params=params, hypothesis=hypothesis,
                            rationale="committee meeting")
        contract = self.menu.build(proposal, seed=0)  # Menu validates recipe + clamps params

        all_concerns = [c for o in opinions for c in o["concerns"]]
        self.last_meeting = {"recipe_id": recipe_id, "draft_params": draft.get("params"),
                             "final_command": contract.command, "opinions": opinions,
                             "concerns": all_concerns}
        if self.notebook:
            self.notebook.log_event(rec, f"committee chose {recipe_id}; "
                                         f"cmd={contract.command}; concerns={all_concerns}")
        return contract, Usage(tin, tout)

    def decide_next(self, result: VerifiedResult, rec: ExperimentRecord):
        res = self.run_fn(_decide_prompt(self.menu, result, rec),
                          system_prompt=self.pi.system_prompt, schema=DECIDE_SCHEMA,
                          model=self.model, max_turns=self.max_turns)
        d = res.data or {}
        if not d.get("propose_followup"):
            return None, res.usage
        nxt = ExperimentRecord(id=d.get("next_id") or f"{rec.id}-next",
                               hypothesis=d.get("hypothesis", ""), parent_id=rec.id,
                               priority=rec.priority)
        return nxt, res.usage
