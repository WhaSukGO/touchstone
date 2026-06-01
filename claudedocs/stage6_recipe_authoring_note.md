# Stage 6 (future) — Autonomous Recipe-Authoring

- Date noted: 2026-06-01
- Status: NOT STARTED (beyond the original 5-stage roadmap)
- Context: Stages 1–5 complete. Today agents can only *select and parameterize* a
  human-authored `Recipe` from the vetted `Menu` (the Stage-3 guardrail). They cannot
  invent a new architecture/dataset/eval and run it. Stage 6 closes that gap *without*
  reopening the "agent goes off the rails" failure.

## The gap
- **Current:** autonomy = explore the *parameter space* of existing recipes (epochs, lr…).
- **Missing:** autonomy to expand into *new methods* (new model code, new datasets, new
  metrics) — i.e. the lab proposing genuinely new recipes by itself.

## Core idea: new capabilities must pass calibration before the autonomous loop can use them
A recipe-authoring loop that mirrors the trust model already in place:

1. **Propose.** A "Recipe Engineer" agent (new committee role) drafts a *candidate recipe*:
   reference train/eval code, dataset refs, metric, a parameter schema, and a CUDA image
   from the existing matrix. Output is structured, like the committee's `Proposal`.
2. **Review.** Static guards: code runs only in the existing sandboxed container (ro
   mounts, `--user`), no network beyond the dataset cache, parameter schema validated.
   A skeptical reviewer agent checks the eval for leakage / triviality.
3. **Calibrate (the gate).** Before admission, the candidate recipe must pass a
   calibration like §7: a positive control (a known-good config reproduces a known
   number) AND a negative control (a deliberately broken config is rejected by its own
   eval). Reuses `calibration_gate` machinery.
4. **Admit.** Only on passing both controls is the recipe added to the `Menu`. From then
   on the normal committee + autonomous loop can use it.

## Why this is the right shape
- Reuses everything: `Menu`/`Recipe`, `calibration_gate`, `ScriptEvaluator`, the sandbox,
  peer review. The only new pieces are a recipe-proposing agent + a recipe-admission gate.
- Keeps the guardrail: a new capability can't enter the autonomous loop until it has
  *proven itself against a known answer* — the same principle that protects single
  experiments, lifted to the level of capabilities.
- Natural cross-lab extension: a recipe authored by lab A can be peer-reviewed (calibrated
  independently) by lab B before the network trusts it.

## Risks / open questions
- Letting an agent author *code* that then runs is the highest-risk capability in the
  system. Mitigations: container isolation (already), no-network, resource caps, and the
  mandatory calibration gate. Consider human sign-off on first admission of any recipe.
- Defining a *general* calibration target for an arbitrary new recipe is hard (for CIFAR
  the oracle is known; for a novel method the "known answer" may not exist). Likely start
  with reproduction-style recipes (known papers) where an oracle exists.
- Code-generation quality + flaky training make the positive control noisy; need loose,
  robust bars (as the CIFAR calibration uses ≥0.45).

## Smallest first step
Add a `RecipeCandidate` schema + a `propose_recipe` committee role that emits a candidate,
then route it through a `recipe_calibration_gate` that admits it to the Menu only on
pass. Demonstrate by having the lab author a *second* CIFAR recipe (e.g. a different
architecture) and admit it autonomously.
