"""Autonomous recipe-authoring + admission (Stage 6).

The committee can only pick from the vetted Menu (Stage 3). Stage 6 lets the lab GROW that
menu: an agent authors a new reusable Recipe (parameterized reference code), and it is
admitted ONLY if it passes an admission gate — the same calibration principle that protects
single experiments, lifted to the level of capabilities:

  positive control: the recipe's default config must VERIFY (it works, oracle reachable);
  negative control: with the produced artifact CORRUPTED, the recipe's evaluator must
                    REJECT it (proving the evaluator actually measures — not a rubber stamp).

Only then does the recipe enter the Menu, where the committee + autonomous loop can use it.
This reuses the Implementer's sandboxed authoring, the harness run, and the independent
evaluator — the only new piece is the admission gate."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..gpu_lease import hold
from ..menu import Menu, ParamSpec, Recipe
from ..models import DatasetRef, ExperimentRecord, FrameworkSpec, Status, Usage
from ..paths import Layout
from ..util import ensure_dir


@dataclass
class RecipeRequest:
    """A request to author a new recipe. The eval_code (grader) and oracle are fixed by the
    requester; the agent only authors the training/entry code that reads the params."""
    recipe_id: str
    description: str
    framework: FrameworkSpec
    datasets: list[DatasetRef]
    metric: str
    threshold: float
    eval_code: str
    params: list[ParamSpec]
    entry_filename: str = "train.py"


# author_fn(request, code_dir) -> Usage : writes entry_filename (reads LAB_<PARAM> env vars)
RecipeAuthorFn = Callable[["RecipeRequest", Path], Usage]


@dataclass
class RecipeCandidate:
    recipe: Recipe
    usage: Usage


def author_recipe(request: RecipeRequest, layout: Layout, *,
                  author_fn: RecipeAuthorFn) -> RecipeCandidate:
    """Author the recipe's code into a stable per-recipe dir and assemble the Recipe spec.
    The agent writes only the entry file; the harness owns eval.py."""
    code_dir = ensure_dir(Path(layout.root) / "recipes" / request.recipe_id)
    (code_dir / "eval.py").write_text(request.eval_code)
    usage = author_fn(request, code_dir)

    # each param is passed to the entry code as LAB_<NAME> (the agent's code reads these)
    env_prefix = " ".join(f"LAB_{p.name.upper()}={{{p.name}}}" for p in request.params)
    template = (f"{env_prefix} python3 $LAB_CODE/{request.entry_filename}").strip()
    recipe = Recipe(
        id=request.recipe_id, description=request.description, framework=request.framework,
        code_dir=str(code_dir), datasets=list(request.datasets), train_template=template,
        eval_command="python3 $LAB_CODE/eval.py", metric=request.metric,
        threshold=request.threshold, params=list(request.params),
    )
    return RecipeCandidate(recipe=recipe, usage=usage)


def _recipe_prompt(request: RecipeRequest) -> str:
    params = "; ".join(
        f"{p.name} = int($LAB_{p.name.upper()}) ({p.type} in [{p.low}, {p.high}], "
        f"default {p.default}; read with os.environ.get and a sensible default)"
        for p in request.params)
    return (
        f"Implement `{request.entry_filename}` for a REUSABLE recipe (it will be run many "
        f"times with different hyperparameters).\n\n{request.description}\n\n"
        f"Hyperparameters to read from the environment: {params}.\n"
        f"Your code is at /code ($LAB_CODE), data (if any) read-only at /data ($LAB_DATA); "
        f"write outputs to /artifacts ($LAB_ARTIFACTS). A hidden held-out evaluator will "
        f"grade it on {request.metric} >= {request.threshold}.\n"
        f"Test via the 'run' tool, setting the hyperparameters yourself, e.g. "
        f"`LAB_{request.params[0].name.upper() if request.params else 'X'}="
        f"{request.params[0].default if request.params else 0} python3 /code/"
        f"{request.entry_filename}`. Iterate until it runs and writes the required "
        f"artifact. Do NOT create or modify eval.py.")


def sdk_recipe_author(job_runner, image_registry, dataset_cache, *,
                      model: str = "claude-sonnet-4-6", max_turns: int = 30) -> RecipeAuthorFn:
    """Live recipe author: a sandboxed Claude session writes the recipe's parameterized
    entry code (reusing the Implementer's authoring session)."""
    def author(request: RecipeRequest, code_dir: Path) -> Usage:
        from ..image_registry import NoImageError
        from .implementer import _run_authoring_session
        image = None
        if request.framework is not None:
            try:
                image = image_registry.resolve(request.framework).image
            except NoImageError:
                image = None
        data_dir = None
        for ref in request.datasets:
            entry = dataset_cache.ensure(ref)
            if not ref.held_out:
                data_dir = entry.path
        return asyncio.run(_run_authoring_session(
            _recipe_prompt(request), str(code_dir), job_runner, image, data_dir,
            model, max_turns))
    return author


@dataclass
class AdmissionReport:
    recipe_id: str
    admitted: bool
    reason: str
    positive_status: str = ""
    negative_verdict: str = ""


def _corrupt_dir(d: Path) -> None:
    for p in sorted(Path(d).rglob("*")):
        if p.is_file():
            p.write_bytes(b"corrupted")


def admit_recipe(harness, candidate: RecipeCandidate, menu: Menu, *,
                 prefix: str = "admit") -> AdmissionReport:
    """Run the admission gate; on pass, add the recipe to `menu`."""
    recipe = candidate.recipe

    # positive control: the recommended (default) config must verify
    pos = ExperimentRecord(
        id=f"{prefix}-{recipe.id}-pos", hypothesis=f"admit {recipe.id}: positive control",
        contract=recipe.build_contract({}, hypothesis="admission positive control"))
    pos = harness.run_experiment(pos)
    if pos.status != Status.VERIFIED:
        return AdmissionReport(recipe.id, False,
                               f"positive control did not verify ({pos.status.value})",
                               positive_status=pos.status.value)

    # negative control: corrupt the produced artifact; the evaluator must now reject it
    artifacts = harness.layout.artifacts(pos.id)
    _corrupt_dir(artifacts)
    neg = ExperimentRecord(id=f"{prefix}-{recipe.id}-neg",
                           hypothesis=f"admit {recipe.id}: negative control")
    with hold(harness.gpu_lease, f"admit:{recipe.id}",
              timeout_s=harness.lease_timeout_s) as token:
        verdict, _ = harness.evaluator.evaluate(neg, pos.contract, str(artifacts), token)
    if verdict.verdict != "FAIL":
        return AdmissionReport(
            recipe.id, False, "evaluator rubber-stamped a corrupted artifact (not trustworthy)",
            positive_status=pos.status.value, negative_verdict=verdict.verdict)

    menu.admit(recipe)
    return AdmissionReport(recipe.id, True, "positive verified + corrupted artifact rejected",
                           positive_status=pos.status.value, negative_verdict=verdict.verdict)
