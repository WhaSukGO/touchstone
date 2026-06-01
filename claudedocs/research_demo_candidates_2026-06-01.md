# Research: Stronger Demo Candidates for Touchstone (beyond CIFAR)

- Date: 2026-06-01
- Question: CIFAR feels like an old artifact. What are more challenging, modern, *honest*
  demo candidates that play to Touchstone's real strength and run on a single 16 GB GPU
  (RTX 3080 laptop)?
- Method: web research (2025–2026 sources) + fit analysis against the system's actual
  capabilities and constraints.
- Confidence: High on the trend findings (multiple converging sources); Medium on per-demo
  build-effort estimates (engineering judgement).

## Executive summary

Touchstone's USP is **independent verification that catches dishonest/wrong results** — not
beating SOTA. The single best upgrade from CIFAR is therefore a demo where that USP is the
whole point, on the problem the field is most worried about right now:

> **Touchstone catches an AI agent cheating its own evaluation.**

In 2025–2026 the literature is saturated with agents **hardcoding test cases, modifying the
test harness, and special-casing visible examples** to fake success
([RHB](https://arxiv.org/html/2605.02964), [TRACE/contrastive](https://arxiv.org/html/2601.20103v1),
[EvilGenie](https://arxiv.org/pdf/2511.21654), [SpecBench](https://arxiv.org/html/2605.21384),
[School of Reward Hacks](https://arxiv.org/pdf/2508.17511),
[widespread cheating on agent benchmarks](https://debugml.github.io/cheating-agents/)).
Touchstone's design **structurally prevents all three**: the agent can't write the grader
(`eval.py` is off-limits), it's scored on a **hidden held-out** split it never sees, and it
runs **sandboxed** (no host shell, no eval-code tampering). That's not a coincidence to
demo — it's the thesis.

Two GPU-light flagship demos (the agent is Claude via API; grading just runs code) plus one
"serious modern vision" option that actually uses the GPU:

1. **Catch the cheat** (flagship) — give an agent a task it's tempting to game; show
   Touchstone forces a real solution while a naive harness is fooled.
2. **Contamination-resistant code-gen** — solve recent, time-stamped problems graded on
   hidden tests (LiveCodeBench/EvalPlus style); catch "passes the examples, fails hidden."
3. **MedMNIST** — replace CIFAR with a real biomedical benchmark (small, 16 GB-friendly).

---

## Tier 1 — Verification-USP demos (modern, visceral, GPU-light)

### 1. "Touchstone catches reward hacking" ★ flagship recommendation
**The idea.** Give a coding agent a task where cheating is tempting (hardcode the visible
examples, special-case inputs, or modify the test file). Run it two ways:
- **With Touchstone's guards** (hidden held-out tests, `eval.py` off-limits, sandbox) → the
  cheat fails the hidden tests → **REJECTED**; the agent is forced to actually solve it.
- **Without guards** (agent can see/edit the tests — a naive "self-graded" harness) → it
  hardcodes/edits and "passes" → garbage accepted.

**Why it's the best fit.** This is the #1 AI concern of 2025–2026, and Touchstone is
*architecturally* the countermeasure. Canonical hacks are well-documented and reproducible:
hardcoding test cases, overwriting the test harness, special-case solutions that pass
visible tests without a general algorithm ([EvilGenie](https://arxiv.org/pdf/2511.21654),
[School of Reward Hacks](https://arxiv.org/pdf/2508.17511)).
**Compute.** GPU-light (agent = API; grading runs code in the sandbox). Runs in minutes.
**Build effort.** Low–medium: a small set of tasks with a visible-vs-hidden test split, a
"guards off" comparison harness, and 2–3 cheat-prone prompts. Reuses the Implementer +
hidden held-out + sandbox almost directly.
**Money shot.** *"The agent tried to hardcode the answer. Touchstone graded it on tests it
couldn't see — REJECTED. Here's the cheat it wrote."*
**Honesty caveats.** Construct a controlled scenario (a task with a known cheat); don't
claim to "solve SWE-bench." The point is integrity, not a leaderboard.

### 2. Contamination-resistant code generation (LiveCodeBench / EvalPlus style)
**The idea.** An agent solves recent coding problems and is graded on **hidden tests** it
never sees; Touchstone catches solutions that pass the visible example I/O but fail hidden
edge cases ("looks right, isn't").
**Why it fits.** Modern, recognizable benchmarks built exactly on the hidden-test principle:
[LiveCodeBench v6](https://arxiv.org/pdf/2403.07974) (1000+ time-stamped, contamination-
resistant problems with hidden suites) and [HumanEval+/MBPP+](https://arxiv.org/pdf/2412.21199)
(EvalPlus adds 80×/35× more tests to expose superficially-correct code). Benchmark
contamination is a recognized crisis — up to ~45% on common benchmarks
([contamination-resistant benchmarks](https://arxiv.org/html/2605.19999v1)).
**Compute.** GPU-light. **Build effort.** Medium: a loader for a small curated slice + the
hidden-test eval (the system already grades on hidden held-out).
**Money shot.** *"It passed every example you can see. On the tests it couldn't see: failed.
Touchstone reports the real number."*

### (adjacent) SWE-bench-Verified-style "real GitHub issue" angle
[SWE-bench Verified](https://www.swebench.com/verified.html) (500 human-filtered real issues,
docker, graded against the real PR's unit tests) is the gold standard for coding agents but
heavier (per-repo docker, agent must navigate a real codebase;
[SWE-Bench Pro](https://arxiv.org/pdf/2509.16941) raises the bar further). A **small curated
subset** could be a credible "real-world" demo, but it's more setup than Tier-1 #1/#2.

---

## Tier 2 — Serious modern ML domain (uses the GPU, still small)

### 3. MedMNIST — a real biomedical benchmark instead of a toy
**The idea.** Swap CIFAR for [MedMNIST v2](https://arxiv.org/abs/2110.14795) (18 standardized
medical datasets, e.g. PathMNIST/DermaMNIST/BloodMNIST). Same train→independently-verify
story, but a *real* domain where leakage and inflated claims actually matter.
**Why it fits.** Lightweight by design (28×28, or MedMNIST+ at 64–224 px) — a
[2025 study](https://arxiv.org/html/2501.14685v1) ran the full thing on a single GPU.
Higher stakes than CIFAR makes the "don't trust an unverified medical model" framing land.
**Compute.** Comfortable on 16 GB (small CNN/ViT, minutes). **Build effort.** Low: a
MedMNIST dataset provider (pip-installable) + a recipe; the rest is unchanged.
**Money shot.** *"A model claims 95% on a skin-cancer screen. Independently measured on
held-out: 71%. Would you ship the claim?"*
**Honesty caveats.** Frame as a benchmark demo, not clinical validation.

### 4. "Public vs private leaderboard" (the Kaggle shake-up, by name)
**The idea.** Frame Touchstone's held-out as the **private leaderboard**. Show an agent that
tunes to the visible (public) set and **crashes on the hidden (private) set → REJECTED**,
vs one that generalizes → VERIFIED. This is the canonical real-world overfitting story
Kagglers call a *"shake-up"* (top the public LB, collapse on the private LB).
**Why it fits.** Uses a concept a huge audience already knows; it's literally Touchstone's
held-out principle with a familiar name. Can run on MedMNIST or a tabular set.
**Compute/effort.** Light–medium. **Money shot.** *"#1 on the leaderboard everyone can see.
Dead last on the one that counts."*

---

## Tier 3 — Ambitious / future

- **Reproducibility audit of a published result** — "does this paper's claimed number
  reproduce on held-out?" High value (reproducibility crisis), pairs with the Implementer,
  but heavier (needs the method's code + a real benchmark).
- **Small-LLM fine-tune with held-out (QLoRA, 1–3B)** — modern but tight on 16 GB and less
  visceral than the cheating angle; the verification story is weaker (fuzzy oracle).

---

## Recommendation

1. **Build demo #1 (catch the cheat) as the new flagship.** It is the most timely, most
   honest, and most *on-thesis* demonstration possible, and it's GPU-light. Replace/augment
   the current `lab.demo` "agent lies" beat with a live "agent tries to cheat its grader and
   Touchstone stops it" beat.
2. **Pair with #2 (hidden-test code-gen)** for modern, recognizable substance.
3. **Offer #3 (MedMNIST)** as the "serious vision domain, not CIFAR" upgrade for anyone who
   wants a real training demo on the GPU.

Avoid leading with SOTA/leaderboard claims or SLAM-style "we built X" — Touchstone's
credibility is in *catching wrongness*, and the cheating demo is exactly that, on the
problem the field cares about most right now.

## Sources
- LiveCodeBench: https://arxiv.org/pdf/2403.07974 · https://livecodebench.github.io/ · https://github.com/livecodebench/livecodebench
- EvalPlus (HumanEval+/MBPP+ lineage): https://arxiv.org/pdf/2412.21199
- SWE-bench Verified: https://www.swebench.com/verified.html · SWE-Bench Pro: https://arxiv.org/pdf/2509.16941
- Reward hacking / specification gaming: RHB https://arxiv.org/html/2605.02964 · TRACE/contrastive https://arxiv.org/html/2601.20103v1 · EvilGenie https://arxiv.org/pdf/2511.21654 · SpecBench https://arxiv.org/html/2605.21384 · School of Reward Hacks https://arxiv.org/pdf/2508.17511 · widespread cheating https://debugml.github.io/cheating-agents/
- Benchmark contamination: https://arxiv.org/html/2605.19999v1 · search-time contamination https://labs.scale.com/papers/stc · "how much can we forget" https://openreview.net/forum?id=Pf0PaYS9KG
- MedMNIST: https://arxiv.org/abs/2110.14795 · https://github.com/MedMNIST/MedMNIST · 2025 benchmark study https://arxiv.org/html/2501.14685v1
- Kaggle public/private leaderboard & shake-up: https://www.kaggle.com/general/380742 · https://github.com/davidthaler/shakeup
