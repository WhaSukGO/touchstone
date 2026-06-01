"""Capability menu (Stage 3 guardrail).

The prior ralph-loop failure mode was an agent inventing arbitrary commands/datasets and
going off the rails. Here the agents can ONLY pick a vetted Recipe and set its declared
parameters; the Recipe owns the (fixed, reviewed) command template and the oracle
threshold. Parameters are type-checked and clamped, and unknown keys are dropped — so no
raw model-authored string ever reaches the shell, and the success bar can't be gamed
down by the proposer."""
from __future__ import annotations

from dataclasses import dataclass, field

from .models import (
    BudgetSpec, Criterion, DatasetRef, ExperimentContract, FrameworkSpec, OracleRef,
)


class MenuError(ValueError):
    pass


@dataclass
class ParamSpec:
    name: str
    type: str                       # "int" | "float" | "choice"
    low: float | None = None
    high: float | None = None
    choices: list | None = None
    default: object = None

    def coerce(self, value: object) -> object:
        """Return a safe value: clamp numerics to [low, high]; unknown choice -> default."""
        if value is None:
            return self.default
        if self.type == "int":
            try:
                v = int(round(float(value)))
            except (TypeError, ValueError):
                return self.default
            if self.low is not None:
                v = max(int(self.low), v)
            if self.high is not None:
                v = min(int(self.high), v)
            return v
        if self.type == "float":
            try:
                v = float(value)
            except (TypeError, ValueError):
                return self.default
            if self.low is not None:
                v = max(float(self.low), v)
            if self.high is not None:
                v = min(float(self.high), v)
            return v
        if self.type == "choice":
            return value if value in (self.choices or []) else self.default
        return self.default

    def describe(self) -> str:
        if self.type == "choice":
            return f"{self.name} (choice of {self.choices}, default {self.default})"
        rng = f"[{self.low}, {self.high}]"
        return f"{self.name} ({self.type} in {rng}, default {self.default})"


@dataclass
class Recipe:
    id: str
    description: str
    framework: FrameworkSpec
    code_dir: str
    datasets: list[DatasetRef]
    train_template: str             # vetted; interpolates only declared params + {seed}
    eval_command: str
    metric: str
    threshold: float                # fixed oracle bar — NOT proposer-chosen
    params: list[ParamSpec] = field(default_factory=list)
    max_wall_s: float = 3600.0

    def validate_params(self, params: dict | None) -> dict:
        """Keep only declared params, coerced/clamped. Drops anything the model invented."""
        params = params or {}
        return {ps.name: ps.coerce(params.get(ps.name)) for ps in self.params}

    def build_contract(self, params: dict | None, *, hypothesis: str = "",
                       seed: int = 0) -> ExperimentContract:
        safe = self.validate_params(params)
        command = self.train_template.format(seed=seed, **safe)
        crit = Criterion(metric=self.metric, op=">=", value=self.threshold)
        return ExperimentContract(
            success_definition=(f"recipe {self.id}: held-out {self.metric} >= "
                                f"{self.threshold}"
                                + (f" | {hypothesis}" if hypothesis else "")),
            gradable_criteria=[crit],
            framework=self.framework,
            datasets=list(self.datasets),
            command=command,
            eval_command=self.eval_command,
            code_dir=self.code_dir,
            budget=BudgetSpec(max_tokens=300_000, max_wall_s=self.max_wall_s, max_retries=1),
            oracle=OracleRef(criterion=crit, source=f"recipe:{self.id}"),
            seed=seed,
        )

    def describe(self) -> str:
        ps = "; ".join(p.describe() for p in self.params) or "(none)"
        ds = ", ".join(d.name + ("[held-out]" if d.held_out else "") for d in self.datasets)
        return (f"- {self.id}: {self.description}\n"
                f"    framework={self.framework.name} {self.framework.version}/"
                f"cuda{self.framework.cuda} | datasets={ds}\n"
                f"    success: held-out {self.metric} >= {self.threshold} (fixed)\n"
                f"    params: {ps}")


@dataclass
class Proposal:
    recipe_id: str
    params: dict = field(default_factory=dict)
    hypothesis: str = ""
    rationale: str = ""


class Menu:
    """The lab's vetted catalogue. Proposals must resolve to a recipe here."""

    def __init__(self, recipes: list[Recipe]):
        self._recipes = {r.id: r for r in recipes}

    def ids(self) -> list[str]:
        return list(self._recipes)

    def recipe(self, recipe_id: str) -> Recipe:
        r = self._recipes.get(recipe_id)
        if r is None:
            raise MenuError(f"unknown recipe {recipe_id!r}; choose from {self.ids()}")
        return r

    def describe(self) -> str:
        return "Available recipes (pick exactly one; set only its listed params):\n" + \
            "\n".join(r.describe() for r in self._recipes.values())

    def build(self, proposal: Proposal, *, seed: int = 0) -> ExperimentContract:
        return self.recipe(proposal.recipe_id).build_contract(
            proposal.params, hypothesis=proposal.hypothesis, seed=seed)
