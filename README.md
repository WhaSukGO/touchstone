# CV Research Lab — Harness Skeleton (Stage 1)

A deterministic harness for an autonomous Computer Vision research lab built on the
Anthropic Agent SDK. This is **Stage 1**: the LLM-free core that owns scheduling,
resources, state, and the independent-evaluator boundary. It runs and self-tests with
**no GPU, Docker, or model**.

Design docs:
- `claudedocs/research_cv_lab_harness_design_2026-06-01.md` — why (lessons from
  Anthropic's *harness design* and *C compiler* posts).
- `claudedocs/design_cv_lab_harness_skeleton_2026-06-01.md` — the full spec this implements.

## Why this exists (the three failures it prevents)

1. **Turn waste on IO.** Downloads/builds/training run in the harness, not the agent
   loop. Budget is measured in **tokens + experiment count**, never turns. IO wait is
   recorded but never charged (`Budget.note_io`).
2. **"It runs" ≠ success.** Success is a gradable `ExperimentContract`, verified by an
   **independent evaluator** that measures on a held-out split and distrusts reported
   metrics.
3. **Self-evaluation bias.** Generator and evaluator are separate contexts. Autonomy is
   locked behind a **calibration gate** (a positive *and* a negative control).

## Quick start

```bash
pip install -r requirements.txt
python -m lab.selftest      # proves the calibration gate with no model/GPU/Docker
pytest -q                   # 10 tests, all local

# Stage 2: real CIFAR-10 calibration on the GPU (needs Docker + nvidia runtime)
docker pull pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
python -m lab.run_cifar_calibration
```

Self-test output shows the key property — the negative control reports a fake score of
0.99, the evaluator independently measures the real checkpoint (0.10) and **rejects it**:

```
negative control: status=REJECTED verdict=FAIL
  generator REPORTED score = 0.99  (a lie)
  evaluator MEASURED score = 0.1  (the truth)
  -> evaluator caught the lie: True
CALIBRATION GATE: OPEN (autonomy unlocked)
```

## Architecture (Stage 1)

```
HARNESS (deterministic Python, 0 LLM)         lab/loop.py
  Queue ─ Registry(SQLite) ─ Budget            queue.py registry.py budget.py
  GpuLease(flock, single mutex)                gpu_lease.py
  ImageRegistry(prebuilt matrix)               image_registry.py
  DatasetCache(download once)                  dataset_cache.py
  JobRunner(local | docker)                    job_runner.py
  Notebook + failed-approaches log             notebook.py

AGENT SEAMS (Protocols; dummy now, Agent SDK in Stage 2)   lab/plugins/base.py
  Planner   — propose_contract / decide_next
  Evaluator — independent, separate context, measures on held-out
```

Experiment lifecycle (each transition committed to the registry → crash-resumable):
`PROPOSED → CONTRACTED → ENV_READY → DATA_READY → RUNNING → ARTIFACTS_READY →
EVALUATING → VERIFIED | REJECTED` (`FAILED` from any step).

The lab's only public output is a `VerifiedResult` **signed by the evaluator**, carrying
provenance (config hash, image, dataset hashes, seed) so future labs can trust it
(multi-lab collaboration).

## Stage 2 status

**Done — real CIFAR-10 calibration on the GPU** (`lab/plugins/cifar.py`, `cifar_ref/`,
`lab/evaluator.py`). Verified on an RTX 3080 via Docker + torch 2.4/cu121:
- positive control: 2-epoch SmallCNN → evaluator measured **65.7%** held-out → VERIFIED
- negative control: 0-epoch model **reporting 0.99** → evaluator measured **8.75%** → REJECTED
- gate OPEN; **0 tokens** (calibration is LLM-free, by design — it validates the
  evaluator against a known answer).

This also validated `docker` job mode end-to-end (GPU passthrough, read-only cache mounts,
`/code` reference-code mount, held-out isolation).

**Agent SDK wired** (`lab/agents/`, on `claude-agent-sdk`):
- `SdkPlanner` proposes `ExperimentContract`s via structured output (real Claude session).
- `SdkEvaluator` is independent + skeptical: it composes the deterministic `ScriptEvaluator`
  (held-out measurement) then opens a **separate** session for judgment. Guarantees a
  measurement that fails the oracle can never be talked up to PASS; the LLM may only
  confirm or **downgrade** a passed measurement to FAIL (e.g. suspected leakage).
- `query()` is independent by default (no shared context); `setting_sources=[]` keeps
  sessions clean; token usage read from `ResultMessage` (budget stays token-based).
- The SDK call is injectable (`run_fn`), so all glue is covered by offline tests (no API,
  no GPU). Live check: `python -m lab.agents.smoke` (billed). Factory:
  `build_cifar_agent_harness`.

**Combined live run — Stage 2 closed.** `python -m lab.run_cifar_calibration --agent`
(real GPU + real LLM evaluator) on an RTX 3080:
- positive: trained 2 epochs → measured 0.684 held-out → oracle passed → LLM judged PASS
  (with genuine skeptical notes about leakage risk) → VERIFIED
- negative: reported 0.99 → measured 0.088 → oracle failed → hard FAIL with **no LLM call**
- gate OPEN; 31k tokens charged (LLM judgment), IO uncharged. ~46s wall.

## Stage 3 status — expert committee + vetted menu

**Done** (`lab/menu.py`, `lab/agents/committee.py`):
- **Capability menu (the guardrail).** Agents can only pick a vetted `Recipe` and set its
  declared params; the recipe owns the (reviewed) command template and a **fixed** oracle
  bar. Params are type-checked and **clamped**, unknown keys are **dropped** — so no raw
  model-authored string ever reaches the shell, and the success bar can't be gamed down.
- **Committee meeting.** PI drafts a menu-constrained proposal → Modeling + Data experts
  review (param overrides + concerns) → deterministic synthesis → `Menu.build` validates.
  Implements the Planner protocol, so it drops into the harness (`build_cifar_committee_harness`).
- Demonstrated offline: a PI proposing `epochs=500` is tuned to `4` by the Modeling expert
  and would be clamped to `20` regardless; an injected `"; rm -rf /"` param is dropped; a
  hallucinated recipe id fails safe (`MenuError` → experiment FAILED, loop continues).

**Still to do (Stage 4+)**
- Live committee run on GPU (needs GPU + `ANTHROPIC_API_KEY`).
- Scale up (CIFAR → ImageNet / a real research domain + oracle; more recipes in the menu).
- Stage 4: `decide_next`-driven autonomous experiment lineage + failed-approaches feedback.

## Layout

```
lab/            harness package (see module docstrings)
  plugins/      base.py (interfaces) + dummy.py (LLM-free self-test plugin)
images/registry.yaml   CUDA image matrix
config.example.yaml    example config
tests/          local test suite
state/ cache/ logs/ workspaces/   runtime (gitignored)
```
