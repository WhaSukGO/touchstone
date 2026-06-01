"""Stage 5 tests — composable labs + cross-lab peer review. Offline (dummy labs)."""
from __future__ import annotations

import os

import pytest

from lab.exchange import Publication, ResultExchange, is_trustworthy, peer_review
from lab.factory import build_dummy_harness
from lab.lab import Lab
from lab.models import ExperimentContract, Provenance, Status, VerifiedResult


def _dummy_lab(tmp_path, name) -> Lab:
    h = build_dummy_harness(tmp_path / name, images_path="images/registry.yaml")
    h.autonomy_enabled = True
    return Lab(name, h)


# ---- trust gate -------------------------------------------------------------
def test_trust_gate():
    prov = Provenance(config_hash="h", image=None)
    assert is_trustworthy(VerifiedResult("e", "PASS", signed_by="ev", provenance=prov))[0]
    assert not is_trustworthy(VerifiedResult("e", "FAIL", signed_by="ev", provenance=prov))[0]
    assert not is_trustworthy(VerifiedResult("e", "PASS", signed_by="", provenance=prov))[0]
    assert not is_trustworthy(VerifiedResult("e", "PASS", signed_by="ev", provenance=None))[0]


def test_exchange_refuses_untrustworthy():
    ex = ResultExchange()
    bad = Publication("A", VerifiedResult("e", "FAIL", signed_by="ev"),
                      ExperimentContract(success_definition="x"), "/tmp")
    with pytest.raises(ValueError):
        ex.publish(bad)


# ---- produce / publish / consume -------------------------------------------
def test_lab_produces_publishes_and_exchange_roundtrip(tmp_path):
    A = _dummy_lab(tmp_path, "A")
    rec = A.run_one("good", exp_id="a-1")
    assert rec.status == Status.VERIFIED

    pub = A.publish("a-1")
    assert pub.lab == "A" and pub.result.verdict == "PASS"

    ex = ResultExchange()
    ex.publish(pub)
    assert ex.get("a-1") is not None
    assert ex.best("score").result.experiment_id == "a-1"


def test_cannot_publish_a_rejected_experiment(tmp_path):
    A = _dummy_lab(tmp_path, "A")
    A.run_one("poison", exp_id="a-bad")           # -> REJECTED
    with pytest.raises(ValueError):
        A.publish("a-bad")


# ---- cross-lab peer review --------------------------------------------------
def test_peer_review_confirms_a_genuine_result(tmp_path):
    A = _dummy_lab(tmp_path, "A")
    B = _dummy_lab(tmp_path, "B")
    A.run_one("good", exp_id="a-1")
    pub = A.publish("a-1")

    review = peer_review(B, pub)
    assert review.agree is True
    assert review.measured == 0.95
    assert review.reviewer == "B" and review.original_lab == "A"


def test_peer_review_disputes_a_tampered_artifact(tmp_path):
    A = _dummy_lab(tmp_path, "A")
    B = _dummy_lab(tmp_path, "B")
    A.run_one("good", exp_id="a-1")
    pub = A.publish("a-1")                          # claim: score=0.95

    # corrupt the checkpoint after publishing — the claim still says 0.95
    with open(os.path.join(pub.artifact_dir, "ckpt.txt"), "w") as f:
        f.write("0.10")

    review = peer_review(B, pub)
    assert review.agree is False                   # B independently measures the real artifact
    assert review.measured == 0.10
    assert "DISPUTED" in review.note
