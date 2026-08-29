"""Slice metrics — recall @ 1% FPR broken out by subgroup (build spec 12).

A single headline recall number can hide a model that only works for
high-amount daytime transactions with full device coverage. Slicing is how
that gets caught before a judge (or production) catches it instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import recall_at_fpr

AMOUNT_BAND_LABELS = ["q1_lowest", "q2", "q3", "q4_highest"]
HOUR_BAND_LABELS = ["night_0_6", "morning_6_12", "afternoon_12_18", "evening_18_24"]


def amount_band(amount: pd.Series) -> pd.Series:
    """4 quantile buckets over the amount distribution (spec: "4 buckets")."""
    try:
        return pd.qcut(amount, q=4, labels=AMOUNT_BAND_LABELS, duplicates="drop")
    except ValueError:
        # Too few distinct values for 4 quantile bins (e.g. a tiny fixture) —
        # fall back to as many equal-frequency bins as the data supports.
        return pd.qcut(amount, q=min(4, amount.nunique()), duplicates="drop")


def hour_band(hour: pd.Series) -> pd.Series:
    bins = [-1, 6, 12, 18, 24]
    return pd.cut(hour, bins=bins, labels=HOUR_BAND_LABELS)


def _recall_by_group(
    y: np.ndarray, score: np.ndarray, group: pd.Series, target_fpr: float
) -> pd.DataFrame:
    rows = []
    for value, idx in group.groupby(group, observed=True).groups.items():
        pos = group.index.get_indexer(idx)
        y_g, s_g = y[pos], score[pos]
        rows.append({
            "slice": str(value),
            "n": int(len(pos)),
            "n_positive": int(y_g.sum()),
            f"recall_at_fpr_{target_fpr}": recall_at_fpr(y_g, s_g, target_fpr),
        })
    return pd.DataFrame(rows)


def build_slices(
    df: pd.DataFrame,
    y: np.ndarray,
    score: np.ndarray,
    target_fpr: float = 0.01,
) -> dict[str, pd.DataFrame]:
    """Slice tables keyed by dimension name.

    df must carry: TransactionAmt, hour, ent_device (may be all-null), and
    either uid_prior_count or an equivalent "is this uid new" signal.
    """
    y = np.asarray(y)
    score = np.asarray(score)
    out: dict[str, pd.DataFrame] = {}

    out["amount_band"] = _recall_by_group(y, score, amount_band(df["TransactionAmt"]), target_fpr)
    out["hour_band"] = _recall_by_group(y, score, hour_band(df["hour"]), target_fpr)

    if "ent_device" in df.columns:
        device_present = df["ent_device"].notna().map({True: "device_present", False: "device_absent"})
        out["device_coverage"] = _recall_by_group(y, score, device_present, target_fpr)

    if "uid_prior_count" in df.columns:
        is_new = (df["uid_prior_count"].fillna(0) <= 0).map({True: "new_uid", False: "returning_uid"})
        out["uid_freshness"] = _recall_by_group(y, score, is_new, target_fpr)

    return out
