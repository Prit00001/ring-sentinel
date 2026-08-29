"""Load the IEEE-CIS training files and join transaction to identity.

Only the `train_*` pair is used. The competition's `test_*` files have no
public labels, so a held-out period is carved out of the training file BY TIME
(see split.py). Nothing here shuffles anything.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_config

log = logging.getLogger(__name__)

TRANSACTION_FILE = "train_transaction.csv"
IDENTITY_FILE = "train_identity.csv"

# Identity columns are id_01..id_38 in the training file. Kaggle's *test*
# identity file uses hyphens (id-01); we only read the train pair, but we
# normalise defensively so a mirror download cannot silently produce an
# all-missing identity block.
def _normalise_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: c.replace("-", "_") for c in df.columns})


class DataNotDownloaded(RuntimeError):
    """Raised with an actionable message when the raw files are absent."""


def _require(path: Path) -> Path:
    if not path.exists():
        raise DataNotDownloaded(
            f"Missing raw data file: {path}\n\n"
            "The IEEE-CIS files are not redistributable and are not committed to\n"
            "this repo. To fetch them:\n\n"
            "  1. Create a Kaggle account and accept the competition rules at\n"
            "     https://www.kaggle.com/competitions/ieee-fraud-detection/rules\n"
            "     (the API returns 403 until you do)\n"
            "  2. Put kaggle.json in ~/.kaggle/ and chmod 600 it\n"
            "  3. Run: make data\n"
        )
    return path


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def write_checksums(raw_dir: Path, out: Path) -> None:
    """Record checksums so a reviewer can verify byte-identical inputs."""
    lines = []
    for p in sorted(raw_dir.rglob("*.csv")):
        lines.append(f"{sha256_file(p)}  {p.relative_to(raw_dir.parent.parent)}")
    out.write_text("\n".join(lines) + "\n")
    log.info("Wrote %d checksums to %s", len(lines), out)


def load_raw(raw_dir: Path | None = None, nrows: int | None = None) -> pd.DataFrame:
    """Join train_transaction with train_identity on TransactionID.

    Returns a frame sorted by TransactionDT with derived `day` and `hour`.
    """
    cfg = load_config()
    raw_dir = raw_dir or cfg.path("raw_ieee")

    tx_path = _require(raw_dir / TRANSACTION_FILE)
    id_path = _require(raw_dir / IDENTITY_FILE)

    log.info("Reading %s", tx_path)
    tx = pd.read_csv(tx_path, nrows=nrows)
    log.info("Reading %s", id_path)
    idf = _normalise_identity_columns(pd.read_csv(id_path))

    df = tx.merge(idf, on="TransactionID", how="left")
    log.info("Joined: %d rows x %d cols", len(df), df.shape[1])
    return prepare(df)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Derive time columns and sort. Shared by the real loader and the tests.

    TransactionDT is a timedelta in seconds from an unstated reference datetime.
    It is NOT a real timestamp and this code never claims a calendar date.
    """
    df = df.copy()
    df["day"] = (df["TransactionDT"] // 86400).astype("int64")
    df["hour"] = ((df["TransactionDT"] // 3600) % 24).astype("int16")
    df["day_of_week"] = (df["day"] % 7).astype("int8")

    # Sort by time. This ordering is load-bearing for every downstream stage and
    # is asserted in tests/test_leakage.py.
    df = df.sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)

    if "isFraud" in df.columns:
        df["isFraud"] = df["isFraud"].astype("int8")
    return df


def identity_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Coverage table per entity-bearing column.

    Published BEFORE any graph metric, because the identity block is sparse and
    a graph statistic computed over a 24%-covered column means something very
    different from one computed over a 100%-covered column.
    """
    cols = [
        "card1", "card2", "card3", "card4", "card5", "card6",
        "addr1", "addr2", "P_emaildomain", "R_emaildomain",
        "DeviceType", "DeviceInfo", "id_30", "id_31", "id_32", "id_33",
        "D1",
    ]
    rows = []
    n = len(df)
    for c in cols:
        if c not in df.columns:
            continue
        present = int(df[c].notna().sum())
        rows.append({
            "column": c,
            "present": present,
            "coverage_pct": round(100.0 * present / n, 2),
            "n_distinct": int(df[c].nunique(dropna=True)),
        })
    return pd.DataFrame(rows)


def base_rate(df: pd.DataFrame) -> float:
    return float(df["isFraud"].mean())


def summarise(df: pd.DataFrame) -> dict:
    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "n_positive": int(df["isFraud"].sum()),
        "base_rate": base_rate(df),
        "day_min": int(df["day"].min()),
        "day_max": int(df["day"].max()),
        "dt_min": int(df["TransactionDT"].min()),
        "dt_max": int(df["TransactionDT"].max()),
    }
