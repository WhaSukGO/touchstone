# Writeup draft — Touchstone

Three formats below: **Show HN blurb** (short), **blog post** (long), **tweet/X thread**.
All numbers are real and reproducible from the repo. Keep the honesty — it's the whole point.

---

## Title options
- **Touchstone: a harness that makes it impossible for an AI to cheat its own eval**
- **Your agent benchmark is gameable. Here's one an agent can't cheat.**
- **Show HN: I built a cheat-proof autograder for AI coding agents**

---

## Show HN blurb (~120 words)

**Show HN: Touchstone — a cheat-proof autograder for AI coding agents**

AI agents are notorious for gaming their own evals — hardcoding the visible tests,
special-casing, even editing the grader. Touchstone makes that structurally impossible:
the solver is graded on **hidden** tests it never sees, runs in a **sandbox** (no host
shell, no network), and **cannot touch the grader**. Nothing is accepted unless an
independent evaluator confirms it on held-out data.

Run it in 10 seconds, no GPU or API key:

```
pip install -r requirements.txt
python -m lab.run_cheat_demo
```

You'll watch two reward hacks score 100% on the tests they can see — and get REJECTED on
the ones they can't. There's also a live mode that grades a real Claude agent on HumanEval+.

Repo: https://github.com/WhaSukGO/touchstone

---

## Blog post (~750 words)

### AI agents cheat their evals. I built a harness that makes it impossible.

If you've watched coding agents long enough, you've seen it: the agent that "solves" a task
by hardcoding the example outputs, special-casing the visible inputs, or — my favorite —
editing the test file so everything passes. The 2025–26 literature is full of it (EvilGenie,
TRACE, SpecBench, "widespread cheating on agent benchmarks"). The uncomfortable truth is
that **most evals trust the thing they're evaluating** — they grade on tests the model can
see, or take the agent's self-reported success at face value.

Touchstone takes the opposite stance: **prevention by construction.** A solver literally
cannot benefit from cheating, because of three structural rules:

1. **It's graded on hidden held-out tests it never sees.** Memorizing the visible examples
   gets you nothing.
2. **It can't touch the grader.** The evaluator is owned by the harness; the agent is
   denied write access to it.
3. **It runs in a sandbox** — a container with no host shell and no network. It can only
   run code through one tool, and only its own code dir is writable.

Nothing is "done" until an **independent** evaluator — a separate context that distrusts
the agent's self-report — measures the result on held-out data against a fixed bar.

#### Watch it catch a cheat (10 seconds, no GPU, no API key)

```
python -m lab.run_cheat_demo
```

```
  solution                           self-graded  Touchstone   verdict
                                       (visible)    (hidden)
  honest implementation                     100%        100%   VERIFIED
  hardcode the visible tests                100%          8%   REJECTED  ← fools naive
  special-case the visible range            100%          8%   REJECTED  ← fools naive
```

Both cheats ace the tests they can see — a naive self-graded harness would pass them — and
both are rejected on the tests they can't. That's the whole product in one screen.

#### On a real benchmark

The same machinery grades a live Claude agent on **HumanEval+** (EvalPlus). The agent gets
a function signature + docstring, implements it sandboxed, and is differential-tested
against a hidden reference on EvalPlus's expanded inputs:

```
  HumanEval+ (hidden, expanded) tests verified: 5/5
  grader-tamper attempts blocked: 0.
```

A strong model passes cleanly — that's the honest result. The *catching* shows up when a
solution is superficial or a model is weaker; the cheat demo and the autograder's gamed
submitter ("claims 3/3, verified 1/3") show that side.

#### The honesty story (this is the part I'm proud of)

The first time I ran the live autograder, it reported **"3 grader-tamper attempts blocked"** —
a dramatic, shareable headline: *the AI tried to tamper with the grader!* Before I shipped
it, I inspected the actual blocked attempt:

```
{'kind': 'write-outside-codedir', 'detail': '/code/main.py'}
```

The agent had simply written its solution to the container path `/code` (which my own prompt
told it about); the host-side write was correctly denied and the agent retried successfully.
**Path confusion, not tampering.** Reporting it as cheating would have been exactly the kind
of overclaiming this project exists to stop. So I fixed the prompt and the classification,
and the honest number is **0**.

That's the discipline, applied to itself: a verifier that will call out *its own* false
positives is the only kind worth trusting.

#### What it is and isn't

Touchstone is a **research-lab harness with verification wired through every step** — a
committee of agents proposes experiments, an implementer writes code, and an independent
evaluator + calibration gate keep everyone honest, all the way to cross-lab peer review.
It does **not** beat SOTA or invent novel methods; it *verifies*. Coding benchmarks here
are API + CPU (the GPU is for training demos). It's an opinionated reference implementation,
not a platform.

Where it sits: eval platforms (Braintrust, Inspect, Promptfoo) **trust the metric you
write**; reward-hack tools like RewardHackWatch **detect** hacking after the fact.
Touchstone makes hacking **not pay off in the first place**. Different stance, complementary
to both.

Try it: `python -m lab.run_cheat_demo` · Repo: https://github.com/WhaSukGO/touchstone

---

## Tweet / X thread (5 posts)

1/ AI agents cheat their evals — hardcoding tests, special-casing, editing the grader.
I built Touchstone: a harness where cheating is *structurally impossible*. 10s, no GPU/API:
`python -m lab.run_cheat_demo` 🧵

2/ The trick: the solver is graded on HIDDEN tests it never sees, in a sandbox (no shell,
no net), and it can't touch the grader. Two reward hacks score 100% on the visible tests —
and get REJECTED on the hidden ones. [demo screenshot]

3/ On a real benchmark: it grades a live Claude agent on HumanEval+ — differential-tested
vs a hidden reference. 5/5 on the expanded (plus) tests, 0 tamper attempts. [screenshot]

4/ The honesty bit: my first live run reported "3 tamper attempts!" — turned out the agent
just used the container path /code; a benign write, denied. I caught my own false positive
and fixed it. A verifier that flags its own overclaims is the only trustworthy kind.

5/ It's not a SOTA model or a platform — it's a verification spine: independent evaluator +
calibration gate + sandbox, prevention-by-construction. Complements detectors/eval tools.
https://github.com/WhaSukGO/touchstone
