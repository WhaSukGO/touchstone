# Stage 7 — The Implementer Loop (code-authoring on top of the verification spine)

- Date: 2026-06-01
- Branch: `stage7-implementer`
- Goal: add the missing half — agents that **implement algorithms** (write + debug code) —
  on top of the existing verification spine, so the lab can *research and implement*, not
  just select+parameterize vetted recipes.
- Non-goal: replacing the verification spine. Nothing the implementer writes is "done"
  until the **independent evaluator** measures it on held-out against an oracle. That gate
  is what makes letting an agent write code safe instead of a repeat of the original
  ralph-loop failure.

## 1. Where it plugs in (reuse, don't rebuild)

The Implementer is a new **generation strategy** behind the existing `Planner` seam:

```
Committee.propose_contract   -> selects+parameterizes a vetted Recipe   (Stage 3)
Implementer.propose_contract -> WRITES + debugs code, returns a contract (Stage 7)
        |                                   |
        +---------- both feed --------------+
                            v
        run_experiment -> ScriptEvaluator/SdkEvaluator (held-out, oracle)  [UNCHANGED]
```

`propose_contract(rec)` for the Implementer:
1. runs a build-test-fix loop that authors code into an isolated `code/` dir,
2. returns an `ExperimentContract` whose `command`/`eval_command` run that code, with
   `code_dir` pointing at it.

Everything downstream (registry, GPU lease, dataset cache, budget, independent evaluator,
calibration gate, history, peer review) is reused as-is.

## 2. The build-test-fix loop

The Anthropic Agent SDK *is* a build-test-fix harness: give a session tools + a goal and
it iterates. So the Implementer is a real Claude session with:
- `Write` / `Read` / `Edit` — to author code in an **isolated workspace** (`cwd`),
- a **custom `sandbox_run` tool** — executes a command in a Docker container (via the
  existing `JobRunner`) and returns stdout/exit. This is the ONLY way it can run code.
- **raw `Bash` / host execution DISALLOWED** (`allowed_tools` excludes Bash; only Write/
  Read/Edit + the sandbox tool).

Loop (driven by the SDK's own agentic loop, bounded by `max_turns` + token budget):
author → `sandbox_run` smoke test → read errors → fix → repeat → emit a final manifest
(entry command, eval command) as structured output.

## 3. Safety model (this is the crux)

Letting an agent write+run code is the highest-risk capability. Defenses, in depth:
1. **Authoring is confined** to an isolated workspace dir (no writes outside `cwd`).
2. **Execution is container-only** via `sandbox_run` → `JobRunner` docker mode: `--user`
   (host uid, not root), read-only dataset/code mounts, `--network none` (add to runner),
   resource/time caps. No raw host shell.
3. **Budget caps**: token budget + max build-fix rounds + wall caps.
4. **The verification gate is the backstop**: the implementer's own smoke tests do NOT
   make a result accepted. Only the independent evaluator, measuring on a held-out split
   the implementer never sees, against a fixed oracle, can VERIFY. Garbage code simply
   never passes — exactly the property the whole project is built on.
5. **Human sign-off (optional)**: first execution of agent-authored code can require
   approval, like recipe admission in the Stage-6 note.

Add `network: "none"` support to `JobRunner` for the sandbox tool (default deny).

## 4. Handoff contract

Implementer emits a structured `Implementation`:
```
{ entry_command: str,        # how to run it (uses LAB_* env, e.g. "python /code/main.py")
  eval_command: str,         # independent measurement (held-out)
  metric: str, op: str, threshold: float,   # the oracle to grade against
  datasets: [DatasetRef], framework: FrameworkSpec, notes: str }
```
-> mapped to an `ExperimentContract` (code_dir = the authored workspace). Then the normal
loop runs + grades it. The oracle/threshold should come from the TASK (given), not be
chosen by the implementer, to prevent gaming (same rule as recipes).

## 5. MVP demo (small, cheap, genuinely "implement an algorithm")

Task: *"Implement a k-nearest-neighbours classifier (NumPy only) for a small tabular
dataset; eval measures held-out accuracy; oracle acc >= 0.85."*
- Implementer writes `knn.py`, smoke-tests via `sandbox_run`, fixes until it runs.
- run_experiment runs it; the independent evaluator measures held-out accuracy.
- Negative control: a task whose oracle is unreachable (or tamper) -> evaluator REJECTS.
CPU-only, tiny data, one cheap LLM session — proves real code authoring + sandbox + the
verification gate end to end.

## 6. Offline testability

Inject the agent (like prior stages): a fake "implementer" that writes a canned good
script (and a canned bad one) instead of calling Claude. Tests then cover: the
`sandbox_run` tool execs in the container and returns output; the Implementation→contract
mapping; the verification handoff (good code -> VERIFIED, bad code -> REJECTED) — all
without API spend. The live path uses the real SDK session with tools.

## 7. Build order
1. `JobRunner`: add `network` control (`--network none`) + a thin `exec(command)->stdout`
   used by the sandbox tool.
2. `sandbox_run` custom SDK tool (in-container exec, the only run path).
3. `Implementer` (Planner seam): SDK session with Write/Read/Edit + sandbox_run; emits
   `Implementation`; maps to a contract. Agent call injectable for tests.
4. Offline tests (fake implementer: good -> VERIFIED, bad -> REJECTED).
5. MVP runner + live demo (k-NN task).
6. (Later) wire into the autonomous loop + Stage-6 recipe admission.

## Open questions
- Multi-file / package projects (start single-file).
- How much the implementer self-tests vs leaving it to the evaluator (keep self-test as a
  smoke gate only; the evaluator remains the source of truth).
- Network-none may break pip installs mid-task — pre-bake deps into the image (recipe
  ethos: vetted images, no runtime installs).
