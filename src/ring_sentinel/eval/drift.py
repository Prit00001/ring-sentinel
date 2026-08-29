"""Drift report — PSI on the top-N features between train and test windows.

Population Stability Index compares a feature's binned distribution in two
periods. It is the standard, cheap check for "did the world change under the
model between when it was trained and when it is scored" (build spec 12).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Conventional PSI interpretation: < 0.1 no meaningful shift, 0.1-0.25 modest
# shift worth watching, > 0.25 significant shift (the spec's flag threshold).
PSI_EPS = 1e-6


def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """PSI of `actual` relative to `expected`, using expected's own quantile
    bin edges so a distribution that hasn't moved scores exactly 0."""
    expected = pd.Series(expected).dropna().to_numpy(dtype=float)
    actual = pd.Series(actual).dropna().to_numpy(dtype=float)
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")

    quantiles = np.linspace(0, 1, buckets + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    if len(edges) < 2:
        return 0.0  # expected is constant — nothing to compare against
    edges[0], edges[-1] = -np.inf, np.inf

    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)

    exp_pct = exp_counts / max(exp_counts.sum(), 1) + PSI_EPS
    act_pct = act_counts / max(act_counts.sum(), 1) + PSI_EPS

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def psi_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    flag_threshold: float = 0.25,
    buckets: int = 10,
) -> pd.DataFrame:
    """One row per feature, sorted by PSI descending, with a flagged column."""
    rows = []
    for col in feature_cols:
        if col not in train_df.columns or col not in test_df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(train_df[col]):
            continue
        value = psi(train_df[col].to_numpy(), test_df[col].to_numpy(), buckets=buckets)
        rows.append({"feature": col, "psi": value, "flagged": bool(value > flag_threshold)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("psi", ascending=False).reset_index(drop=True)
    return out


def top_n_by_importance(feature_importance: dict[str, float], n: int) -> list[str]:
    """Feature names for the top-N by (e.g. LightGBM gain) importance."""
    ordered = sorted(feature_importance.items(), key=lambda kv: kv[1], reverse=True)
    return [name for name, _ in ordered[:n]]
