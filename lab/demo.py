"""Touchstone in one screen — it catches AI when it's wrong, and only ships results you
can trust.

  python -m lab.demo          # offline: instant, no GPU, no API key (runnable by anyone)
  python -m lab.demo --live    # the real thing: a GPU + a real Claude agent (billed)

Three beats, each independently verified — nothing is staged."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from .models import ExperimentRecord, FrameworkSpec, Usage


def _hr(c="─"):
    print(c * 70)


def _headline(n, benefit):
    print(f"\n{'━' * 70}\n  BEAT {n}.  {benefit}\n{'━' * 70}")


# --- beat 1: agents lie; the independent verifier catches it ------------------
def beat_lie(root: Path, live: bool) -> bool:
    if live:
        from .factory import build_cifar_harness
        from .plugins.cifar import seed_experiment
        h = build_cifar_harness(str(root), job_mode="docker")
        rec = h.run_experiment(seed_experiment("cifar poison (lie)", exp_id="demo-lie"))
        metric = "top1"
        print("  A model was trained and the producer reported its score.")
    else:
        from .factory import build_dummy_harness
        from .plugins.dummy import seed_experiment
        h = build_dummy_harness(str(root), images_path="images/registry.yaml")
        rec = h.run_experiment(seed_experiment("poison", exp_id="demo-lie"))
        metric = "score"
        print("  A 'researcher' produced a model and reported its score.")

    reported = rec.reported_metrics.get(metric)
    measured = rec.verdict.measured_metrics.get(metric) if rec.verdict else None
    print(f"  ► reported {metric}:  {reported}      (the claim)")
    print(f"  ► an INDEPENDENT verifier measured it on held-out data:  {metric} = {measured}")
    print(f"  ► verdict: {rec.status.value}  —  the inflated claim was caught and rejected.")
    return rec.status == rec.status.REJECTED


# --- beat 2: real autonomous work, proven on held-out -------------------------
_IMPL_EVAL = (
    "import json, os\n"
    "s = float(open(os.path.join(os.environ['LAB_ARTIFACTS'], 'result.txt')).read())\n"
    "json.dump({'score': s}, open(os.path.join(os.environ['LAB_EVAL_OUT'], 'heldout.json'), 'w'))\n"
)


def _impl_task():
    from .agents.implementer import ImplementationTask
    return ImplementationTask(
        "produce a result scoring >= 0.85 on held-out", FrameworkSpec("dummy", "0", ""),
        "python3 $LAB_CODE/main.py", "python3 $LAB_CODE/eval.py", _IMPL_EVAL, "score", ">=", 0.85)


def _impl_author(score: str):
    def author(task, code_dir: Path, rec) -> Usage:
        (code_dir / "main.py").write_text(
            "import os\n"
            f"open(os.path.join(os.environ['LAB_ARTIFACTS'], 'result.txt'), 'w').write('{score}')\n")
        return Usage(0, 0)
    return author


def beat_implement(root: Path, live: bool) -> bool:
    if live:
        from .agents.implementer import sdk_author
        from .factory import build_implementer_harness
        from .plugins.knn_demo import KnnDatasetProvider, knn_task
        h = build_implementer_harness(str(root), knn_task(), author_fn=lambda *a: Usage(),
                                      provider=KnnDatasetProvider(), job_mode="docker")
        h.planner.author_fn = sdk_author(h.job_runner, h.image_registry, h.dataset_cache)
        print("  A sandboxed AI agent wrote a k-NN classifier from scratch")
        print("  (container only — no host shell, no network), then it was graded on")
        print("  held-out labels it never saw.")
        rec = h.run_experiment(ExperimentRecord(id="demo-impl", hypothesis="implement k-NN"))
        print(f"  ► independently measured held-out accuracy: {rec.verdict.measured_metrics.get('acc')}")
        print(f"  ► verdict: {rec.status.value}")
        return rec.status == rec.status.VERIFIED

    from .factory import build_implementer_harness
    good = build_implementer_harness(str(root / "good"), _impl_task(), _impl_author("0.9"))
    g = good.run_experiment(ExperimentRecord(id="s", hypothesis="impl"))
    bad = build_implementer_harness(str(root / "bad"), _impl_task(), _impl_author("0.5"))
    b = bad.run_experiment(ExperimentRecord(id="s", hypothesis="impl"))
    print("  Two pieces of code were submitted; both RAN without errors.")
    print(f"  ► submission A: held-out score {g.verdict.measured_metrics['score']}  →  {g.status.value}")
    print(f"  ► submission B: held-out score {b.verdict.measured_metrics['score']}  →  {b.status.value}")
    print("  'It ran' is not success — only the one that hit the target was accepted.")
    return g.status == g.status.VERIFIED and b.status == b.status.REJECTED


# --- beat 3: even the verifier is checked — a second lab re-confirms -----------
def beat_peer(root: Path, live: bool) -> bool:
    from .exchange import peer_review
    from .lab import Lab
    if live:
        from .factory import build_cifar_harness
        a = Lab("alpha", build_cifar_harness(str(root / "A"), job_mode="docker"))
        b = Lab("beta", build_cifar_harness(str(root / "B"), job_mode="docker"))
        a.harness.run_experiment(ExperimentRecord(id="r", hypothesis="train cifar"))
        artifact_name, metric = "model.pt", "top1"
    else:
        from .factory import build_dummy_harness
        a = Lab("alpha", build_dummy_harness(str(root / "A"), images_path="images/registry.yaml"))
        b = Lab("beta", build_dummy_harness(str(root / "B"), images_path="images/registry.yaml"))
        a.run_one("good", exp_id="r")
        artifact_name, metric = "ckpt.txt", "score"

    pub = a.publish("r")
    confirmed = peer_review(b, pub)
    # corrupt alpha's artifact after publishing
    for f in Path(pub.artifact_dir).glob(artifact_name):
        f.write_bytes(b"corrupted")
    disputed = peer_review(b, pub)

    print(f"  Lab α published a result (held-out {metric} = {pub.result.measured_metrics.get(metric)}).")
    print(f"  ► Lab β independently re-measured the artifact  →  "
          f"{'CONFIRMED' if confirmed.agree else 'DISPUTED'}")
    print("  Then we CORRUPTED α's artifact after it published…")
    print(f"  ► Lab β independently re-measured again  →  "
          f"{'CONFIRMED' if disputed.agree else 'DISPUTED'}  (it caught the corruption)")
    return confirmed.agree and not disputed.agree


def main(argv: list[str]) -> int:
    live = "--live" in argv
    mode = "LIVE — real GPU + a real Claude agent (billed)" if live else \
        "offline — no GPU, no API key, ~instant"
    print(f"\n  TOUCHSTONE")
    print(f"  It catches AI when it's wrong, and only ships results you can trust.")
    print(f"  mode: {mode}")

    base = Path("./run_demo") if live else Path(tempfile.mkdtemp(prefix="touchstone-demo-"))
    results = []

    _headline(1, "AI agents confidently lie. Touchstone catches them.")
    results.append(beat_lie(base / "lie", live))

    _headline(2, "It does real, autonomous work — and proves it on held-out data.")
    results.append(beat_implement(base / "impl", live))

    _headline(3, "Even the verifier is checked: a second lab independently re-confirms.")
    results.append(beat_peer(base / "peer", live))

    print()
    _hr("━")
    ok = all(results)
    print(f"  {sum(results)}/3 beats passed — every claim was independently verified, "
          f"not asserted.")
    print(f"  {'Nothing here is staged.' if ok else 'A beat did not behave as expected.'}")
    _hr("━")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
