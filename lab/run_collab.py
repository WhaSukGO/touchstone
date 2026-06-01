"""Live Stage 5: two CIFAR labs collaborate (GPU-only, no API by default).

Lab 'alpha' trains and verifies a model, then publishes a SIGNED VerifiedResult to a
shared exchange. Lab 'beta' independently PEER-REVIEWS it — re-measuring alpha's artifact
on its own held-out split, in its own context — before the result is trusted. The
deterministic CIFAR harness is used (fixed recipe), so this costs GPU time but no tokens.

  python -m lab.run_collab            # docker/GPU
  python -m lab.run_collab --tamper   # corrupt the artifact after publish -> beta DISPUTES
"""
from __future__ import annotations

import os
import sys

from .exchange import ResultExchange, peer_review
from .factory import build_cifar_harness
from .lab import Lab
from .models import ExperimentRecord, Status


def main(argv: list[str]) -> int:
    job_mode = "local" if "--local" in argv else "docker"
    tamper = "--tamper" in argv

    alpha = Lab("alpha", build_cifar_harness("./run_collab_A", job_mode=job_mode))
    beta = Lab("beta", build_cifar_harness("./run_collab_B", job_mode=job_mode))

    print("Lab alpha: training + verifying...")
    rec = alpha.harness.run_experiment(ExperimentRecord(
        id="alpha-1", hypothesis="Train a CIFAR-10 SmallCNN clearing 0.45 held-out top-1."))
    if rec.status != Status.VERIFIED:
        print("alpha did not verify:", rec.status.value)
        return 1
    print(f"  alpha VERIFIED, measured={rec.verdict.measured_metrics}, "
          f"signed_by={rec.verdict.signed_by}")

    pub = alpha.publish("alpha-1")
    exchange = ResultExchange()
    exchange.publish(pub)
    print(f"  published '{pub.result.experiment_id}' to the exchange (provenance: "
          f"image={pub.result.provenance.image}, seed={pub.result.provenance.seed})")

    if tamper:
        ckpt = os.path.join(pub.artifact_dir, "model.pt")
        with open(ckpt, "wb") as f:           # corrupt the checkpoint after publishing
            f.write(b"corrupted")
        print("  [tampered] alpha's checkpoint was corrupted after publishing")

    print("Lab beta: independent peer review...")
    review = peer_review(beta, pub)
    print("=" * 60)
    print(f"PEER REVIEW: {'CONFIRMED' if review.agree else 'DISPUTED'}")
    print(f"  alpha claimed {review.metric}={review.claimed}; "
          f"beta measured {review.measured}")
    print(f"  note: {review.note}")
    print("=" * 60)
    return 0 if review.agree else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
