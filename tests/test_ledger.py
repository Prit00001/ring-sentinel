"""Hash-chain integrity tests for the audit ledger (build spec section 11).

`make verify-ledger` walking this chain and failing on any break is the
audit log's entire reason for existing — these tests are the ledger's own
version of test_leakage.py's "prove the guard can fail" discipline.
"""

from __future__ import annotations

import json

from ring_sentinel.audit.ledger import LedgerEntry, append, verify


def _entry(case_id: str, decision: str) -> LedgerEntry:
    return LedgerEntry(
        ts="2026-01-01T00:00:00+00:00",
        case_id=case_id,
        actor="analyst",
        action="decide",
        decision=decision,
        expected_cost=100.0,
        prev_hash="",  # append() is the source of truth for this; see below
    )


def test_verify_true_for_a_ledger_that_does_not_exist_yet(tmp_path):
    """A fresh checkout with no decisions recorded must not crash."""
    assert verify(tmp_path / "no_such_ledger.jsonl") is True


def test_verify_true_for_an_empty_ledger_file(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text("")
    assert verify(path) is True


def test_append_sets_the_real_prev_hash_not_whatever_the_caller_passed(tmp_path):
    """The caller passes prev_hash="" (or anything) because it cannot know
    the true chain tail under concurrent writers — append() must overwrite
    it with the real value, not persist the caller's placeholder."""
    path = tmp_path / "ledger.jsonl"
    append(path, _entry("c_1", "block"))
    append(path, _entry("c_2", "allow"))

    lines = [json.loads(l) for l in path.read_text().splitlines()]
    assert lines[0]["prev_hash"] == "0" * 64
    assert lines[1]["prev_hash"] == lines[0]["hash"]
    assert lines[0]["prev_hash"] != ""
    assert lines[1]["prev_hash"] != ""


def test_chain_of_several_entries_verifies(tmp_path):
    path = tmp_path / "ledger.jsonl"
    for i in range(5):
        append(path, _entry(f"c_{i}", "review"))
    assert verify(path) is True


def test_verify_detects_tampering():
    """A guard that cannot fail is not a guard: corrupt one field in the
    middle of the chain and confirm verify() actually notices."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "ledger.jsonl"
        for i in range(4):
            append(path, _entry(f"c_{i}", "review"))
        assert verify(path) is True

        lines = path.read_text().splitlines()
        tampered = json.loads(lines[2])
        tampered["decision"] = "block"  # flip a recorded decision after the fact
        lines[2] = json.dumps(tampered)
        path.write_text("\n".join(lines) + "\n")

        assert verify(path) is False


def test_concurrent_appends_do_not_corrupt_the_chain(tmp_path):
    """append() must serialise concurrent writers via flock rather than let
    two requests both read the same prev_hash and both write from it."""
    import threading

    path = tmp_path / "ledger.jsonl"
    n_writers = 8

    def worker(i: int) -> None:
        append(path, _entry(f"c_{i}", "review"))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = path.read_text().splitlines()
    assert len(lines) == n_writers
    assert verify(path) is True
