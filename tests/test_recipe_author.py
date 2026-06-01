"""Stage 6 tests — recipe authoring + admission gate. Offline (fake author, local mode).

Proves the admission principle: a new recipe enters the menu only if its default config
verifies AND its evaluator rejects a corrupted artifact. A recipe whose evaluator
rubber-stamps (ignores the artifact) is refused — the lab won't grow an untrustworthy
capability."""
from __future__ import annotations

from pathlib import Path

from lab.agents.recipe_author import RecipeRequest, admit_recipe, author_recipe
from lab.factory import build_recipe_admission_harness
from lab.menu import Menu, ParamSpec, Proposal
from lab.models import FrameworkSpec, Usage

# A real evaluator: the held-out score IS the produced artifact's value.
_GOOD_EVAL = (
    "import json, os\n"
    "s = float(open(os.path.join(os.environ['LAB_ARTIFACTS'], 'ckpt.txt')).read())\n"
    "json.dump({'q': s}, open(os.path.join(os.environ['LAB_EVAL_OUT'], 'heldout.json'), 'w'))\n"
)
# A rubber-stamp evaluator: ignores the artifact and always reports a pass.
_RUBBER_EVAL = (
    "import json, os\n"
    "json.dump({'q': 0.99}, open(os.path.join(os.environ['LAB_EVAL_OUT'], 'heldout.json'), 'w'))\n"
)


def _author(request, code_dir: Path) -> Usage:
    # the agent writes parameterized training code that reads LAB_LEVEL
    (code_dir / "train.py").write_text(
        "import os\n"
        "lvl = int(os.environ['LAB_LEVEL'])\n"
        "open(os.path.join(os.environ['LAB_ARTIFACTS'], 'ckpt.txt'), 'w').write(str(lvl / 10))\n")
    return Usage(10, 5)


def _request(eval_code: str) -> RecipeRequest:
    return RecipeRequest(
        recipe_id="authored-clf", description="train a toy classifier; tune level",
        framework=FrameworkSpec("dummy", "0", ""), datasets=[],
        metric="q", threshold=0.85, eval_code=eval_code,
        params=[ParamSpec("level", "int", low=1, high=10, default=9)])


def test_recipe_admitted_when_evaluator_measures(tmp_path):
    h = build_recipe_admission_harness(tmp_path / "lab")
    menu = Menu([])
    candidate = author_recipe(_request(_GOOD_EVAL), h.layout, author_fn=_author)

    report = admit_recipe(h, candidate, menu)

    assert report.admitted is True
    assert report.positive_status == "VERIFIED"
    assert report.negative_verdict == "FAIL"        # corrupted artifact correctly rejected
    assert "authored-clf" in menu.ids()             # the menu grew
    # the admitted recipe is immediately usable by the committee:
    contract = menu.build(Proposal("authored-clf", {}))
    assert "LAB_LEVEL=9" in contract.command        # default param wired through


def test_recipe_refused_when_evaluator_rubber_stamps(tmp_path):
    h = build_recipe_admission_harness(tmp_path / "lab2")
    menu = Menu([])
    candidate = author_recipe(_request(_RUBBER_EVAL), h.layout, author_fn=_author)

    report = admit_recipe(h, candidate, menu)

    assert report.admitted is False                 # positive passes, but corrupted not rejected
    assert report.positive_status == "VERIFIED"
    assert report.negative_verdict == "PASS"        # the rubber stamp is what gets it refused
    assert "authored-clf" not in menu.ids()         # the untrustworthy recipe is kept out
