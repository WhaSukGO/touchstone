"""Live Stage 6 demo: the lab AUTHORS a new recipe and admits it. BILLED + GPU/Docker.

A sandboxed Claude session writes a parameterized k-NN recipe (tunable K), then the
admission gate runs: the default config must VERIFY, and with the produced predictions
CORRUPTED the recipe's evaluator must REJECT — proving the new capability works and its
evaluator actually measures. Only then is it admitted to the menu.

  python -m lab.run_recipe_author_demo
"""
from __future__ import annotations

import sys
from pathlib import Path

from .agents.recipe_author import (
    RecipeRequest, admit_recipe, author_recipe, sdk_recipe_author,
)
from .factory import build_recipe_admission_harness
from .menu import Menu, ParamSpec
from .models import DatasetRef, FrameworkSpec
from .plugins.knn_demo import _EVAL_CODE, KnnDatasetProvider


def main(argv: list[str]) -> int:
    h = build_recipe_admission_harness("./run_recipe", job_mode="docker",
                                       provider=KnnDatasetProvider())
    request = RecipeRequest(
        recipe_id="authored-knn",
        description=("Implement a k-nearest-neighbours classifier (NumPy). Read the labelled "
                     "training set $LAB_DATA/train.csv (x1,x2,label) and the unlabelled "
                     "$LAB_DATA/test_features.csv (x1,x2). Use K neighbours. Write the "
                     "predicted label for each test row, one integer per line, to "
                     "$LAB_ARTIFACTS/predictions.csv (same order as test_features.csv)."),
        framework=FrameworkSpec("torch", "2.4", "12.1"),
        datasets=[DatasetRef("knn-train", "synthetic"),
                  DatasetRef("knn-heldout", "synthetic", held_out=True)],
        metric="acc", threshold=0.80, eval_code=_EVAL_CODE,
        params=[ParamSpec("k", "int", low=1, high=15, default=5)],
    )

    author = sdk_recipe_author(h.job_runner, h.image_registry, h.dataset_cache, max_turns=30)
    print("Agent authoring a new parameterized recipe (sandboxed)...")
    candidate = author_recipe(request, h.layout, author_fn=author)

    menu = Menu([])
    print("Running admission gate (positive control + corrupted-artifact control)...")
    report = admit_recipe(h, candidate, menu)

    print("=" * 64)
    print(f"ADMITTED: {report.admitted}")
    print(f"  positive control: {report.positive_status}")
    print(f"  corrupted-artifact control verdict: {report.negative_verdict}  "
          f"(must be FAIL to admit)")
    print(f"  reason: {report.reason}")
    print(f"  menu now: {menu.ids()}")
    print(f"  author tokens: in={candidate.usage.tokens_in} out={candidate.usage.tokens_out}")
    train_py = Path(candidate.recipe.code_dir) / "train.py"
    if train_py.exists():
        print("--- authored train.py (first 25 lines) ---")
        print("\n".join(train_py.read_text().splitlines()[:25]))
    print("=" * 64)
    return 0 if report.admitted else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
