"""Hash-chained append-only audit log for decisions.

Concurrency: append() takes an exclusive advisory lock (fcntl.flock) on the
ledger file for the read-prev-hash + write span, so two requests handled
concurrently by the ASGI server cannot both read the same prev_hash and
produce two entries that both claim to follow it — which would silently
break the chain that verify() exists to check.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path
from dataclasses import dataclass, asdict

GENESIS_HASH = "0" * 64


@dataclass
class LedgerEntry:
    ts: str
    case_id: str
    actor: str
    action: str
    decision: str
    expected_cost: float
    prev_hash: str
    model_version: str | None = None
    policy_version: str | None = None
    score: float | None = None
    hash: str = ""

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}


def sha256_entry(prev_hash: str, entry_dict: dict) -> str:
    canonical = json.dumps(entry_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256((prev_hash + canonical).encode()).hexdigest()


def _last_hash(f) -> str:
    prev_hash = GENESIS_HASH
    for line in f:
        line = line.strip()
        if not line:
            continue
        prev_hash = json.loads(line).get("hash", prev_hash)
    return prev_hash


def append(ledger_path: Path, entry: LedgerEntry) -> None:
    """Append an entry to the ledger with hash chaining.

    Opened in "a+" so the same file descriptor is used to read the existing
    chain and to write the new line, with a single flock held across both —
    a competing writer blocks until this entry (and its hash, which depends
    on having read the true current tail) is durably written.
    """
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            prev_hash = _last_hash(f)

            # The caller need not (and, given concurrent writers, cannot
            # reliably) know the current chain tail — append() is the single
            # source of truth for it. Overwrite whatever prev_hash the entry
            # was constructed with so the persisted record is truthful.
            entry.prev_hash = prev_hash
            entry_dict = entry.to_dict()
            entry_dict.pop("hash", None)
            entry.hash = sha256_entry(prev_hash, entry_dict)

            f.write(json.dumps(entry.to_dict()) + "\n")
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def verify(ledger_path: Path) -> bool:
    """Verify the entire hash chain.

    A missing or empty ledger is vacuously valid — there is nothing to
    tamper with yet, and `make verify-ledger` on a fresh checkout (before any
    console decision has been recorded) must not crash.
    """
    if not ledger_path.exists():
        return True
    prev_hash = GENESIS_HASH
    with ledger_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            recorded_hash = entry.pop("hash")
            computed = sha256_entry(prev_hash, entry)
            if computed != recorded_hash:
                return False
            prev_hash = recorded_hash
    return True
