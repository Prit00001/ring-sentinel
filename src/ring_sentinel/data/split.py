"""Temporal splitting. Never shuffle. TransactionDT ordering is sacred.

Returns index arrays only — the splitter never copies or reorders the frame,
so there is no path by which a caller can accidentally receive shuffled data.

An embargo window is dropped between consecutive splits. Without it, a
transaction one second before the train/val boundary and its ring partner one
second after sit in different splits while sharing a component, and the model
is quietly scored on a case it was partly trained on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import load_config


@dataclass(frozen=True)
class Splits:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    embargoed: np.ndarray
    boundaries: dict

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"train": self.train, "val": self.val, "test": self.test}

    def label_of(self, n_rows: int) -> np.ndarray:
        """Per-row split label; embargoed rows get 'embargo'."""
        out = np.full(n_rows, "embargo", dtype=object)
        out[self.train] = "train"
        out[self.val] = "val"
        out[self.test] = "test"
        return out

    def summary(self) -> dict:
        return {
            "n_train": len(self.train),
            "n_val": len(self.val),
            "n_test": len(self.test),
            "n_embargoed": len(self.embargoed),
            **self.boundaries,
        }


def temporal_split(df: pd.DataFrame, cfg=None) -> Splits:
    """Split by TransactionDT into train/val/test with an embargo gap.

    df must already be sorted by TransactionDT (data.load.prepare does this).
    """
    cfg = cfg or load_config()
    sp = cfg.base["split"]
    train_frac = float(sp["train_frac"])
    val_frac = float(sp["val_frac"])
    embargo_sec = int(sp["embargo_days"]) * 86400

    dt = df["TransactionDT"].to_numpy()
    if not np.all(np.diff(dt) >= 0):
        raise ValueError(
            "TransactionDT is not sorted ascending. The splitter refuses to run "
            "on an unsorted frame — call data.load.prepare first."
        )

    n = len(df)
    i1 = int(round(n * train_frac))
    i2 = int(round(n * (train_frac + val_frac)))

    # Boundary timestamps, taken as the first row of the *later* split.
    t_boundary_1 = dt[i1]
    t_boundary_2 = dt[i2]

    idx = np.arange(n)

    # Embargo: drop rows from the start of the later split that fall within
    # embargo_sec of the boundary. Dropping from the later side (rather than the
    # earlier) means the training set keeps its full history and the evaluation
    # set is the one that pays the cost — the conservative direction.
    val_start = int(np.searchsorted(dt, t_boundary_1 + embargo_sec, side="left"))
    test_start = int(np.searchsorted(dt, t_boundary_2 + embargo_sec, side="left"))

    train = idx[:i1]
    val = idx[val_start:i2]
    test = idx[test_start:]

    embargoed = np.concatenate([idx[i1:val_start], idx[i2:test_start]])

    boundaries = {
        "train_end_dt": int(dt[i1 - 1]) if i1 > 0 else None,
        "val_start_dt": int(dt[val_start]) if val_start < n else None,
        "val_end_dt": int(dt[i2 - 1]) if i2 > 0 else None,
        "test_start_dt": int(dt[test_start]) if test_start < n else None,
        "train_end_day": int(df["day"].iloc[i1 - 1]) if i1 > 0 else None,
        "val_start_day": int(df["day"].iloc[val_start]) if val_start < n else None,
        "test_start_day": int(df["day"].iloc[test_start]) if test_start < n else None,
        "embargo_days": int(sp["embargo_days"]),
    }

    return Splits(train=train, val=val, test=test, embargoed=embargoed, boundaries=boundaries)


def assert_disjoint_and_ordered(df: pd.DataFrame, s: Splits) -> None:
    """Hard invariants. Called by the pipeline and by the leakage tests."""
    dt = df["TransactionDT"].to_numpy()

    sets = [set(s.train.tolist()), set(s.val.tolist()), set(s.test.tolist())]
    assert not (sets[0] & sets[1]), "train/val index overlap"
    assert not (sets[1] & sets[2]), "val/test index overlap"
    assert not (sets[0] & sets[2]), "train/test index overlap"

    if len(s.train) and len(s.val):
        assert dt[s.train].max() < dt[s.val].min(), "train is not strictly before val"
    if len(s.val) and len(s.test):
        assert dt[s.val].max() < dt[s.test].min(), "val is not strictly before test"
    if len(s.train) and len(s.test):
        assert dt[s.train].max() < dt[s.test].min(), "train is not strictly before test"
