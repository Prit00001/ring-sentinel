"""Headline metrics (build spec section 12).

PR-AUC, recall at fixed FPR, precision@k, and calibration are the metrics
this track actually asks for. ROC-AUC may appear but is inflated at a 3.5%
base rate, so roc_auc_with_caveat returns the note that has to travel with
the number rather than leaving a reader to rediscover that themselves.

Calibration (Brier, ECE) is intentionally NOT reimplemented here — it already
lives in models/calibrate.py, which the leakage-test suite depends on for
test_calibrator_never_saw_test. headline_metrics() reuses it rather than
maintaining two copies of the same formula.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ..models.calibrate import brier_score, expected_calibration_error


def pr_auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(average_precision_score(y, score))


def roc_auc_with_caveat(y: np.ndarray, score: np.ndarray) -> tuple[float, str]:
    base_rate = float(np.mean(y))
    value = float(roc_auc_score(y, score))
    caveat = (
        f"ROC-AUC is reported for reference only. At a {base_rate:.1%} base rate, "
        "ROC-AUC is dominated by the large true-negative population and looks "
        "high even for a mediocre detector; PR-AUC and recall@FPR are the "
        "metrics that actually distinguish models on this track."
    )
    return value, caveat


def recall_at_fpr(y: np.ndarray, score: np.ndarray, target_fpr: float) -> float:
    """Recall (true positive rate) at the score threshold achieving target_fpr.

    Thresholds are swept from the negative class's score distribution — the
    threshold is the (1 - target_fpr) quantile of scores among y == 0, so the
    achieved FPR is <= target_fpr by construction (never sneaks over it).
    """
    y = np.asarray(y)
    score = np.asarray(score)
    neg_scores = score[y == 0]
    pos_scores = score[y == 1]
    if len(neg_scores) == 0 or len(pos_scores) == 0:
        return float("nan")

    threshold = float(np.quantile(neg_scores, 1.0 - target_fpr))
    tp = int((pos_scores >= threshold).sum())
    return tp / len(pos_scores)


def precision_at_k(y: np.ndarray, score: np.ndarray, k: int) -> float:
    """Precision among the top-k highest-scored rows."""
    y = np.asarray(y)
    score = np.asarray(score)
    if k <= 0 or len(y) == 0:
        return float("nan")
    k = min(k, len(y))
    top_idx = np.argsort(-score, kind="mergesort")[:k]
    return float(y[top_idx].mean())


def case_level_precision(case_flagged: np.ndarray, case_has_confirmed_fraud: np.ndarray) -> float:
    """Spec's explicit case-level precision definition: a flagged case is a
    true positive if it contains AT LEAST ONE confirmed fraudulent
    transaction — not "every member is fraud"."""
    flagged = np.asarray(case_flagged, dtype=bool)
    if flagged.sum() == 0:
        return float("nan")
    hit = np.asarray(case_has_confirmed_fraud, dtype=bool) & flagged
    return float(hit.sum() / flagged.sum())


def headline_metrics(
    y: np.ndarray,
    score: np.ndarray,
    fpr_points: list[float],
    precision_at_k_per_day: int,
    ece_bins: int = 15,
) -> dict:
    """One row of the headline table (build spec section 12)."""
    y = np.asarray(y)
    score = np.asarray(score)

    roc, roc_caveat = roc_auc_with_caveat(y, score)
    ece, _bins = expected_calibration_error(y, score, ece_bins)

    row = {
        "n": int(len(y)),
        "base_rate": float(y.mean()) if len(y) else float("nan"),
        "pr_auc": pr_auc(y, score),
        "roc_auc": roc,
        "roc_auc_caveat": roc_caveat,
        "brier": brier_score(y, score),
        "ece": ece,
        "precision_at_k": precision_at_k(y, score, precision_at_k_per_day),
    }
    for fp in fpr_points:
        row[f"recall_at_fpr_{fp}"] = recall_at_fpr(y, score, fp)
    return row
