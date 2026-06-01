"""Stage 2 calibration test — real CIFAR-10 on GPU via Docker.

Opt-in (it needs the GPU, Docker, the torch image, and a ~170MB download):
    LAB_RUN_GPU_TESTS=1 pytest tests/test_cifar_calibration.py
Skipped by default so the standard suite stays fast and host-independent."""
from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from lab.factory import build_cifar_harness
from lab.models import Status
from lab.plugins.cifar import seed_experiment

IMAGE = "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime"


def _docker_image_ready() -> bool:
    if not shutil.which("docker"):
        return False
    out = subprocess.run(["docker", "images", "-q", IMAGE],
                         capture_output=True, text=True)
    return bool(out.stdout.strip())


pytestmark = pytest.mark.skipif(
    os.environ.get("LAB_RUN_GPU_TESTS") != "1" or not _docker_image_ready(),
    reason="GPU/Docker calibration (set LAB_RUN_GPU_TESTS=1 with torch image present)",
)


def test_cifar_calibration_gate_on_real_gpu(tmp_path):
    h = build_cifar_harness(tmp_path / "lab", job_mode="docker")
    pos = seed_experiment("cifar positive control", exp_id="cal-pos")
    neg = seed_experiment("cifar poison negative control", exp_id="cal-neg")

    assert h.calibration_gate(pos, neg) is True

    pos_rec = h.registry.get("cal-pos")
    neg_rec = h.registry.get("cal-neg")
    assert pos_rec.status == Status.VERIFIED
    assert pos_rec.verdict.measured_metrics["top1"] >= 0.45

    # negative control lied (reported 0.99); evaluator measured the real checkpoint low.
    assert neg_rec.status == Status.REJECTED
    assert neg_rec.reported_metrics["top1"] == 0.99
    assert neg_rec.verdict.measured_metrics["top1"] < 0.45
    assert h.budget.state.total_tokens == 0  # calibration is LLM-free
