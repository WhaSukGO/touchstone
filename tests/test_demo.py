"""The offline demo doubles as an end-to-end smoke test: all three beats must pass."""
from __future__ import annotations

from lab.demo import main


def test_offline_demo_all_beats_pass(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "REJECTED" in out          # beat 1: the lie was caught
    assert "VERIFIED" in out          # beat 2: good work accepted
    assert "DISPUTED" in out          # beat 3: tampering caught
