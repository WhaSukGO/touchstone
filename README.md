# Touchstone

*An autonomous computer-vision research lab that trusts no claim it didn't independently measure.*

**A harness-engineered, self-directing computer-vision research lab where a committee of
Claude agents proposes, implements, and runs experiments on GPU — and an independent,
skeptical evaluator verifies every result on held-out data against a fixed oracle, so
nothing counts as "done" until something independent confirms it actually works.**

**See it in 10 seconds** (no GPU, no API key): `python -m lab.run_cheat_demo` — watch it
catch an agent cheating its own tests. (Or `python -m lab.demo` for the full tour;
[DEMO.md](DEMO.md).)

Built on the Anthropic Agent SDK. Inspired by Anthropic's engineering posts on
[harness design for long-running agents](https://www.anthropic.com/engineering/harness-design-long-running-apps)
and [building a C compiler](https://www.anthropic.com/engineering/building-c-compiler).

---

## The idea in one picture

```
 a research direction
        │
        ▼
 ┌──────────────┐   propose      ┌──────────────┐   run on GPU    ┌──────────────┐
 │  COMMITTEE   │──────────────▶ │   HARNESS    │───────────────▶ │  experiment  │
 │ PI · Modeling│  (or WRITE     │ (deterministic│  (container,    │  artifacts   │
 │ · Data       │   the code)    │  no LLM)     │   cached data)  │              │
 └──────────────┘                └──────────────┘                 └──────┬───────┘
        ▲                                                                 │
        │ decide next (memory of past runs)                              ▼
        │                                              ┌────────────────────────────┐
        └───────────────── VERIFIED only ─────────────│  INDEPENDENT EVALUATOR      │
                                                       │  separate context; measures │
                                                       │  on HELD-OUT vs the oracle; │
                                                       │  distrusts reported numbers │
                                                       └────────────────────────────┘
```

The agent that does the work and the agent that judges it are **different, isolated
sessions**. A second lab can **peer-review** any result by independently re-measuring it.

## Why it works this way (the failures it prevents)

This started from a failed agent loop. The fixes are the architecture:

1. **"It runs" ≠ success.** Success is a gradable contract, checked by an **independent
   evaluator** that measures on a **held-out split** the producer never saw and distrusts
   self-reported numbers. (In testing it caught a run reporting `0.99` that was actually
   `0.088`.)
2. **Self-evaluation bias.** The generator and evaluator are **separate contexts**.
   Autonomy stays locked behind a **calibration gate** — it won't trust the evaluator
   until the evaluator both confirms a known-good run *and* rejects a deliberately
   poisoned one.
3. **Wasted budget on I/O.** Downloads, builds, and training run in the harness, not the
   agent loop. Budget is measured in **tokens + experiments**, never "turns"; I/O waits
   are free.
4. **Going off the rails.** Agents pick from a **vetted menu** of recipes (or write code
   in a **sandbox with no host shell**); they can't author arbitrary commands, and nothing
   they produce is accepted until the verifier confirms it.

## What it can do (verified on a real GPU)

| Capability | Try it | Proven result |
|---|---|---|
| **Catches reward hacking** — agents can't cheat a hidden, untouchable grader | `python -m lab.run_cheat_demo` | hardcode/special-case cheats score 100% on visible tests → **REJECTED** on hidden |
| Independent verification + **calibration gate** | `python -m lab.run_cifar_calibration` | trained model VERIFIED; a run lying `0.99` (real `0.088`) REJECTED |
| **Expert committee** proposes a menu-constrained experiment | `python -m lab.run_cifar_committee` | chose epochs/lr; evaluator passed it but flagged a reported-vs-held-out inflation |
| **Autonomous lineage** — a chain of experiments toward a goal, with memory | `python -m lab.run_cifar_autonomous` | iterates; stops on stall / budget |
| **Cross-lab peer review** | `python -m lab.run_collab [--tamper]` | a 2nd lab CONFIRMS a genuine result; DISPUTES a tampered checkpoint |
| **Implementer** — a sandboxed agent writes code, then it's independently graded | `python -m lab.run_implementer_demo` | agent wrote NumPy k-NN; held-out acc **0.9933** → VERIFIED |
| **Recipe-authoring** — the lab grows its own menu, gated by admission | `python -m lab.run_recipe_author_demo` | agent authored a parameterized k-NN recipe; admitted (default VERIFIED, corrupted artifact REJECTED) |

## Quick start

```bash
pip install -r requirements.txt

# 1) No GPU, Docker, or API key needed — proves the whole loop + calibration gate offline:
python -m lab.selftest
pytest -q                      # 37 local tests (+1 opt-in GPU test)

# 2) Real GPU runs need Docker + the NVIDIA runtime, and a torch image:
docker pull pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
python -m lab.run_cifar_calibration          # GPU only, no API
python -m lab.run_collab                      # GPU only, no API

# 3) Agent runs need ANTHROPIC_API_KEY (billed) in addition to a GPU:
export ANTHROPIC_API_KEY=...                  # or put it in .env
python -m lab.run_cifar_committee
python -m lab.run_implementer_demo
```

The offline self-test prints the core property — the evaluator catching a lie:

```
negative control: status=REJECTED verdict=FAIL
  generator REPORTED score = 0.99  (a lie)
  evaluator MEASURED score = 0.1   (the truth)
CALIBRATION GATE: OPEN (autonomy unlocked)
```

## How it's built

**Deterministic harness (`lab/`, no LLM)** — owns everything mechanical so the agents only
ever spend tokens on reasoning:
- `loop.py` experiment state machine + autonomous lineage; `registry.py` crash-resumable
  SQLite store; `budget.py` token/experiment caps.
- `gpu_lease.py` single-GPU mutex; `image_registry.py` prebuilt CUDA-image matrix (no
  runtime builds); `dataset_cache.py` download-once cache; `job_runner.py` runs jobs in
  Docker (or locally) as the host user with read-only mounts.

**Agent seams (`lab/agents/`)** — real Claude sessions behind clean Protocols:
- `committee.py` PI + Modeling + Data experts negotiate a **menu-constrained** proposal
  (`menu.py` — recipes with clamped parameters and a fixed oracle).
- `evaluator.py` the independent, skeptical evaluator (measures on held-out, can confirm or
  downgrade but never upgrade a failed measurement).
- `implementer.py` + `sandbox_tool.py` an agent that **writes code**, executing only in a
  container (no host shell, no network, writes confined to its code dir).
- `history.py` feeds prior results back into planning so the lab builds on itself.

**Composable labs (`lab/lab.py`, `lab/exchange.py`)** — a lab's only export is a
**signed `VerifiedResult`** (with provenance: config hash, image digest, dataset hash,
seed). Other labs consume it only through a trust gate, or re-verify it via peer review.

The pluggable bits are the **domain**: a `Recipe` = dataset provider + reference code +
metric + oracle + CUDA image. Add one and the whole machinery (committee, autonomy,
verification, peer review) applies. Today there's a CIFAR-10 recipe and a k-NN implementer
task.

## What it does *not* do (honest limits)

- It does **not invent novel methods or chase SOTA**. Autonomy means exploring the
  parameter space of vetted recipes, or implementing a *specified* task — not inventing.
- The lab can now **author its own recipes** and admit them via a calibration gate
  (Stage 6), but wiring a genuinely new *domain* (its dataset provider + CUDA image) is
  still human setup. One CV recipe (CIFAR-10) and one implementer task ship today.
- **Single local GPU, sequential**; labs run **in one process** (collaboration isn't
  networked).
- Verification needs a **known oracle**, so it's strongest for reproduction-style work.

## Layout

```
lab/                 harness core + agents/ + plugins/
  agents/            committee, evaluator, implementer, sandbox tool, SDK wrapper
  plugins/           cifar (recipe + reference code), knn_demo, dummy (offline)
  run_*.py           runnable demos (calibration, committee, autonomous, collab, implementer)
images/registry.yaml CUDA image matrix
tests/               37 local tests + 1 opt-in GPU test
claudedocs/          design + research notes (the "why" and the staged build)
```

See `claudedocs/research_cv_lab_harness_design_2026-06-01.md` (the rationale) and the
`design_*` / `stage*` notes for the full design history.
