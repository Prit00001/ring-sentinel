"""TEST FIXTURE ONLY. NEVER USED FOR ANY REPORTED METRIC.

===========================================================================
WHAT THIS IS
  A small random table that carries the IEEE-CIS *column schema* so the
  pipeline's plumbing can be exercised by pytest and by `make smoke` on a
  machine that has no Kaggle credentials. Roughly 6,000 rows over ~40 synthetic
  days.

WHAT THIS IS NOT
  It is not a fraud simulator and it is not a traffic generator. It contains no
  fraud typology, no evasion logic, no attempt to imitate real attacker
  behaviour, and nothing that could be pointed at a detector. The label is a
  plain logistic function of two ordinary numeric columns plus noise — the same
  thing you would write to unit-test any binary classifier. It exists so that
  `test_shuffled_target_kills_signal` has *some* signal to destroy; without a
  learnable relationship that test would pass vacuously.

  NO NUMBER PRODUCED FROM THIS FIXTURE APPEARS IN README.md, MODEL_CARD.md, OR
  reports/results.md. Outputs generated from it land in reports/_smoke/ behind a
  loud banner. The build spec's rule stands: no custom or synthesised dataset is
  used for any reported metric.
===========================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_ROWS = 6000
N_DAYS = 40


def make_fixture(n_rows: int = N_ROWS, n_days: int = N_DAYS, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Timestamps: seconds, sorted, spread over n_days. Deliberately includes
    # duplicate timestamps so the tie-group handling in causal.py is exercised.
    dt = np.sort(rng.integers(0, n_days * 86400, size=n_rows))
    dt[: n_rows // 50] = dt[1]  # a block of exact ties

    n_cards = 400
    card1 = rng.integers(1000, 1000 + n_cards, size=n_rows)
    card2 = rng.integers(100, 600, size=n_rows).astype(float)
    card3 = np.full(n_rows, 150.0)
    card5 = rng.integers(100, 240, size=n_rows).astype(float)
    # addr1 needs enough distinct values that it does not become a bridging
    # hub across the whole window on its own: with only 60 distinct values
    # (the original range) and addr2 held constant, every row shares one of
    # 60 "addr" entity nodes, which — chained through the also-moderate
    # "device" entity — merges an entire 14-day window into one component
    # regardless of hub pruning. 3,000 distinct values against ~150 rows/day
    # keeps typical addr degree near the min_degree=2 floor, so the fixture
    # actually exercises hub pruning (via the deliberately skewed
    # P_emaildomain "gmail.com" hub below) instead of collapsing structurally.
    addr1 = rng.integers(200, 3200, size=n_rows).astype(float)
    addr2 = np.full(n_rows, 87.0)

    amt = np.round(np.exp(rng.normal(3.4, 1.1, size=n_rows)), 2)

    day = dt // 86400
    # D1 = "days since the card began". Give each card1 a stable origin so that
    # day - D1 reconstructs a usable uid, exercising resolve.add_uid.
    origin = rng.integers(0, 30, size=n_cards + 1000)
    d1 = np.clip(day - origin[card1 - 1000], 0, None).astype(float)
    d1[rng.random(n_rows) < 0.05] = np.nan

    # Same reasoning as addr1 above: DeviceInfo needs enough distinct raw
    # values that the normalised ent_device combo (DeviceInfo x id_30 x id_31
    # x id_32 x id_33) stays well below one occurrence per window row. With
    # only 28 raw device names the combined entity had ~206 surviving values
    # across a ~2,146-row window (avg degree ~10) — individually under the
    # max_entity_degree=200 cap, but dense enough that transitive union-find
    # chaining merged nearly the whole window into one component anyway. 2,000
    # distinct names pushes combos well past window size so most degrees fall
    # to 0-1 and get dropped by min_entity_degree, leaving only genuine repeat
    # devices as edges.
    devices = [f"SM-G{i}" for i in range(2000)] + ["Windows", "MacOS", "iOS Device"]
    dev = rng.choice(devices + [None] * len(devices), size=n_rows)

    emails = ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "aol.com", None]
    p_email = rng.choice(emails, size=n_rows, p=[0.45, 0.15, 0.12, 0.13, 0.05, 0.10])

    df = pd.DataFrame({
        "TransactionID": np.arange(1, n_rows + 1),
        "TransactionDT": dt,
        "TransactionAmt": amt,
        "ProductCD": rng.choice(list("WCHRS"), size=n_rows),
        "card1": card1, "card2": card2, "card3": card3,
        "card4": rng.choice(["visa", "mastercard", "discover", "american express"], size=n_rows),
        "card5": card5,
        "card6": rng.choice(["debit", "credit"], size=n_rows),
        "addr1": addr1, "addr2": addr2,
        "P_emaildomain": p_email,
        "R_emaildomain": rng.choice(emails, size=n_rows),
        "DeviceType": rng.choice(["desktop", "mobile", None], size=n_rows),
        "DeviceInfo": dev,
        "id_30": rng.choice(["Windows 10", "iOS 11.1.2", "Android 7.0", None], size=n_rows),
        "id_31": rng.choice(["chrome 63.0", "safari 11.0", "mobile safari 11.0", None], size=n_rows),
        "id_32": rng.choice([24.0, 32.0, np.nan], size=n_rows),
        "id_33": rng.choice(["1920x1080", "1334x750", "2208x1242", None], size=n_rows),
    })

    for i in range(1, 15):
        df[f"C{i}"] = rng.poisson(2.0, size=n_rows).astype(float)
    df["D1"] = d1
    for i in range(2, 16):
        v = rng.exponential(30.0, size=n_rows)
        v[rng.random(n_rows) < 0.3] = np.nan
        df[f"D{i}"] = v
    for i in range(1, 10):
        df[f"M{i}"] = rng.choice(["T", "F", None], size=n_rows)
    for i in range(1, 30):
        df[f"V{i}"] = rng.normal(size=n_rows)
    # A few deliberately collinear V columns so prune_v_block has work to do.
    df["V30"] = df["V1"] * 2.0 + rng.normal(0, 0.01, size=n_rows)
    df["V31"] = df["V2"] * -1.5 + rng.normal(0, 0.01, size=n_rows)
    for i in range(2, 39):
        col = f"id_{i:02d}"
        if col not in df.columns:
            df[col] = rng.normal(size=n_rows) if i % 2 else rng.choice(["A", "B", None], size=n_rows)

    # ---- label: an ordinary logistic function of two ordinary columns ----
    # No fraud typology is encoded. This exists solely so a model has something
    # to learn and the shuffle test has something to destroy.
    card_repeat = pd.Series(card1).map(pd.Series(card1).value_counts()).to_numpy()
    z = -3.6 + 0.35 * np.log1p(amt) + 0.06 * card_repeat + rng.normal(0, 0.6, size=n_rows)
    p = 1.0 / (1.0 + np.exp(-z))
    df["isFraud"] = (rng.random(n_rows) < p).astype(np.int8)

    return df


def write_fixture(path) -> None:
    make_fixture().to_csv(path, index=False)


if __name__ == "__main__":
    import sys
    write_fixture(sys.argv[1] if len(sys.argv) > 1 else "fixture.csv")
