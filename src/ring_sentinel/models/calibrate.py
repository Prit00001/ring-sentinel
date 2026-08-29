"""Isotonic calibration, fitted on the validation split only.

The calibrator records the exact index set it was fitted on. That is not
defensive programming for its own sake: `test_calibrator_never_saw_test`
asserts the recorded set is disjoint from test, which turns "we calibrated
properly" from a claim in a README into something a reviewer can execute.

Calibration matters here more than in most projects because the policy engine
consumes probabilities as rupees. An uncalibrated score of 0.9 that really
means 0.4 does not produce a slightly wrong decision — it produces a decision
optimised against a cost that was never on the table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass
class CalibrationReport:
    brier: float
    ece: float
    log_loss: float
    bins: list


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> tuple[float, list]:
    """Equal-width binned ECE, plus the per-bin table for the reliability plot."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    total = len(p)
    ece = 0.0
    rows = []
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            rows.append({"bin": b, "lo": edges[b], "hi": edges[b + 1], "n": 0,
                         "mean_pred": None, "frac_pos": None})
            continue
        mean_pred = float(p[m].mean())
        frac_pos = float(y[m].mean())
        ece += (n / total) * abs(mean_pred - frac_pos)
        rows.append({"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]),
                     "n": n, "mean_pred": mean_pred, "frac_pos": frac_pos})
    return float(ece), rows


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def safe_log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_report(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> CalibrationReport:
    ece, rows = expected_calibration_error(y, p, n_bins)
    return CalibrationReport(
        brier=brier_score(y, p), ece=ece, log_loss=safe_log_loss(y, p), bins=rows
    )


class IsotonicCalibrator:
    """Thin wrapper that remembers its fitting index set."""

    def __init__(self) -> None:
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.fit_index: np.ndarray | None = None
        self.fitted = False

    def fit(self, scores: np.ndarray, y: np.ndarray, fit_index: np.ndarray | None = None):
        if fit_index is None:
            raise ValueError(
                "fit_index is required. The calibrator must record which rows it "
                "saw so the leakage test can verify disjointness from test."
            )
        self.iso.fit(scores, y)
        self.fit_index = np.asarray(fit_index)
        self.fitted = True
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("calibrator not fitted")
        return np.clip(self.iso.predict(scores), 0.0, 1.0)

    __call__ = transform
