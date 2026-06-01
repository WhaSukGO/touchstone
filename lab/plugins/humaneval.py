"""Real benchmark slice: HumanEval+ (EvalPlus), wired into the cheat-proof autograder.

The agent gets a function signature + docstring and must implement the function (written
to solution.py, sandboxed). The harness-owned grader differential-tests the agent's
function against the reference solution (which the agent never sees) on the EvalPlus
inputs: the BASE inputs (the original HumanEval tests = 'claimed') and the expanded PLUS
inputs ('actual'). EvalPlus's whole finding is that many solutions pass base but fail plus
— superficially correct. Here that shows up as claimed-vs-actual, on real, recognizable
problems, with the tests hidden and the grader off-limits.

Dataset is fetched once from the official EvalPlus release (no heavy deps)."""
from __future__ import annotations

import gzip
import json
import urllib.request
from pathlib import Path

from ..agents.implementer import ImplementationTask
from ..models import DatasetRef, FrameworkSpec, Usage

_URL = ("https://github.com/evalplus/humanevalplus_release/releases/download/"
        "v0.1.10/HumanEvalPlus.jsonl.gz")
_CACHE = Path.home() / ".cache" / "touchstone" / "humanevalplus-v0.1.10.jsonl"

# harness-owned differential grader. Builds the reference (prompt + canonical) and the
# candidate (the agent's solution.py), runs both on base then plus inputs, all-or-nothing.
_EVAL = r'''
import json, math, os, signal, copy
d = json.load(open(os.path.join(os.environ["LAB_DATA"], "problem.json")))
out = os.path.join(os.environ["LAB_EVAL_OUT"], "heldout.json")


def _build(src):
    ns = {}
    exec(src, ns)
    return ns[d["entry_point"]]


def _eq(a, b, atol):
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-6, abs_tol=max(atol or 0, 1e-9))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_eq(x, y, atol) for x, y in zip(a, b))
    return a == b


signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(TimeoutError()))

try:
    ref = _build(d["prompt"] + d["canonical_solution"])
    cand = _build(open(os.path.join(os.environ["LAB_CODE"], "solution.py")).read())
except Exception as e:
    json.dump({"acc": 0.0, "base": 0.0, "error": str(e)[:120]}, open(out, "w"))
    raise SystemExit


def passes(inputs):
    for args in inputs:
        try:
            signal.alarm(8)
            r = ref(*copy.deepcopy(args))
            c = cand(*copy.deepcopy(args))
            signal.alarm(0)
            if not _eq(r, c, d.get("atol", 0)):
                return 0.0
        except Exception:
            signal.alarm(0)
            return 0.0
    return 1.0


json.dump({"acc": passes(d["plus_input"]), "base": passes(d["base_input"])}, open(out, "w"))
'''


def _ensure_dataset() -> Path:
    if not _CACHE.exists():
        _CACHE.parent.mkdir(parents=True, exist_ok=True)
        raw = urllib.request.urlopen(_URL, timeout=120).read()
        _CACHE.write_text(gzip.decompress(raw).decode())
    return _CACHE


class HumanEvalProblem:
    def __init__(self, rec: dict):
        self.rec = rec
        self.id = rec["task_id"].replace("/", "-")           # e.g. HumanEval-0
        self.entry_point = rec["entry_point"]

    def task(self) -> ImplementationTask:
        return ImplementationTask(
            description=(f"Complete this Python function. Write the COMPLETE function "
                        f"(signature + body) to `solution.py` in your current directory — "
                        f"no tests, no main block, no extra prints.\n\n"
                        f"```python\n{self.rec['prompt']}```\n"
                        f"The function name is `{self.entry_point}`."),
            framework=FrameworkSpec("torch", "2.4", "12.1"),
            entry_command=("python3 -c \"import ast,os; ast.parse(open(os.path.join("
                           "os.environ['LAB_CODE'],'solution.py')).read())\""),
            eval_command="python3 $LAB_CODE/eval.py", eval_code=_EVAL,
            metric="acc", op=">=", threshold=0.999,
            datasets=[DatasetRef(f"{self.id}-problem", "evalplus", held_out=True)],
            entry_filename="solution.py")

    def provider(self):
        return _ProblemProvider(self.rec)

    def canonical_author(self):   # the reference solution (passes base + plus)
        return _writer(self.rec["prompt"] + self.rec["canonical_solution"])

    def broken_author(self):      # a stub that returns nothing (fails)
        return _writer(self.rec["prompt"] + "    return None\n")


class _ProblemProvider:
    """Writes the held-out problem spec (reference solution + hidden test inputs) — the
    evaluator-only split the agent never sees."""
    def __init__(self, rec: dict):
        self.rec = rec

    def fetch(self, ref: DatasetRef, dest: Path) -> None:
        keep = {k: self.rec[k] for k in
                ("prompt", "canonical_solution", "entry_point", "base_input", "plus_input")}
        keep["atol"] = self.rec.get("atol", 0)
        (dest / "problem.json").write_text(json.dumps(keep))


def _writer(src: str):
    def author(task, code_dir: Path, rec) -> Usage:
        (code_dir / "solution.py").write_text(src)
        return Usage(0, 0)
    return author


def load_problems(n: int = 5) -> list[HumanEvalProblem]:
    lines = _ensure_dataset().read_text().splitlines()
    return [HumanEvalProblem(json.loads(line)) for line in lines[:n]]
