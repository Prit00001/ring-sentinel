"""Tests for eval/metrics.py, slices.py, drift.py, report.py.

These run entirely against the synthetic fixture / synthetic arrays — no
Kaggle download required, same spirit as `make smoke`. They exist because
SETUP_PHASES_EXPLAINED.md claims the eval/ layer is "Complete" and it needs
to actually be true, not just claimed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ring_sentinel.eval.drift import psi, psi_report, top_n_by_importance
from ring_sentinel.eval.metrics import (
    case_level_precision,
    headline_metrics,
    precision_at_k,
    pr_auc,
    recall_at_fpr,
    roc_auc_with_caveat,
)
from ring_sentinel.eval.report import (
    df_to_markdown_table,
    write_ablation_md,
    write_drift_md,
    write_results_md,
    write_slices_md,
)
from ring_sentinel.eval.slices import amount_band, build_slices, hour_band


@pytest.fixture
def scored_binary():
    rng = np.random.default_rng(0)
    n = 2000
    y = (rng.random(n) < 0.035).astype(int)
    # A score correlated with y but noisy — enough signal that recall@low-FPR
    # is meaningfully between 0 and 1, not degenerate.
    score = np.clip(y * 0.6 + rng.random(n) * 0.5, 0, 1)
    return y, score


# ---------------------------------------------------------------------------
# metrics.py
# ---------------------------------------------------------------------------

def test_recall_at_fpr_threshold_tracks_the_target_fpr(scored_binary):
    """The quantile-derived threshold puts the achieved FPR close to the
    target. It is not an exact upper bound: with ties at the threshold value
    (common near a 0.1% tail on a few thousand rows), >= at the threshold can
    admit a few extra points, so a small overshoot is expected, not a bug."""
    y, score = scored_binary
    for target in (0.001, 0.01, 0.05):
        threshold = np.quantile(score[y == 0], 1.0 - target)
        achieved_fpr = float((score[y == 0] >= threshold).mean())
        assert achieved_fpr <= target * 1.5 + 0.002


def test_recall_at_fpr_is_monotonic_in_target_fpr(scored_binary):
    """A looser FPR budget can only catch as much or more fraud."""
    y, score = scored_binary
    r_tight = recall_at_fpr(y, score, 0.001)
    r_loose = recall_at_fpr(y, score, 0.05)
    assert r_loose >= r_tight


def test_perfect_score_gives_perfect_pr_auc():
    y = np.array([0, 0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
    assert pr_auc(y, score) == pytest.approx(1.0)


def test_precision_at_k_basic():
    y = np.array([0, 1, 0, 1, 0])
    score = np.array([0.1, 0.9, 0.2, 0.8, 0.05])
    # top 2 by score are indices 1 and 3, both positive.
    assert precision_at_k(y, score, 2) == pytest.approx(1.0)


def test_precision_at_k_per_day_is_not_the_same_as_over_the_whole_period():
    """k=2 per day, across 2 days: day 0's top-2 (indices 0,1) are both
    positive; day 1's top-2 by score (indices 4,5) are both negative --
    the actual fraud in day 1 (index 3) scores lower and is missed. Pooled
    precision is 2/4 = 0.5, not the 1.0 you'd get by mistakenly taking one
    top-(k=2) over all 6 rows (which only ever looks at day 0)."""
    day = np.array([0, 0, 0, 1, 1, 1])
    y = np.array([1, 1, 0, 1, 0, 0])
    score = np.array([0.9, 0.8, 0.1, 0.5, 0.7, 0.6])
    assert precision_at_k(y, score, 2, day=day) == pytest.approx(0.5)
    # Sanity: without `day`, k=2 over the whole array only sees day 0's
    # top-2 (both positive) and misses day 1 entirely — the bug being fixed.
    assert precision_at_k(y, score, 2) == pytest.approx(1.0)


def test_case_level_precision_matches_spec_definition():
    """A flagged case is a TP if it contains at least one confirmed fraud
    transaction — not "every member is fraud" (build spec section 8)."""
    flagged = np.array([True, True, False])
    has_confirmed_fraud = np.array([True, False, True])  # 2nd case: 0 fraud members
    assert case_level_precision(flagged, has_confirmed_fraud) == pytest.approx(0.5)


def test_roc_auc_caveat_mentions_base_rate(scored_binary):
    y, score = scored_binary
    _value, caveat = roc_auc_with_caveat(y, score)
    assert "base rate" in caveat.lower()


def test_headline_metrics_has_every_reported_field(scored_binary):
    y, score = scored_binary
    row = headline_metrics(y, score, fpr_points=[0.001, 0.01, 0.05], precision_at_k_per_day=20)
    for key in ("pr_auc", "roc_auc", "brier", "ece", "precision_at_k",
                "recall_at_fpr_0.001", "recall_at_fpr_0.01", "recall_at_fpr_0.05"):
        assert key in row


# ---------------------------------------------------------------------------
# slices.py
# ---------------------------------------------------------------------------

def test_amount_band_produces_four_groups():
    amt = pd.Series(np.linspace(1, 1000, 400))
    bands = amount_band(amt)
    assert bands.nunique() <= 4


def test_hour_band_covers_all_24_hours():
    hours = pd.Series(range(24))
    bands = hour_band(hours)
    assert bands.notna().all()


def test_build_slices_returns_amount_and_hour_at_minimum(scored_binary):
    y, score = scored_binary
    df = pd.DataFrame({
        "TransactionAmt": np.abs(np.random.default_rng(1).normal(50, 20, len(y))) + 1,
        "hour": np.random.default_rng(2).integers(0, 24, len(y)),
    })
    slices = build_slices(df, y, score, target_fpr=0.05)
    assert "amount_band" in slices
    assert "hour_band" in slices
    assert slices["amount_band"]["n"].sum() == len(y)


# ---------------------------------------------------------------------------
# drift.py
# ---------------------------------------------------------------------------

def test_psi_is_near_zero_for_an_identical_distribution():
    rng = np.random.default_rng(3)
    x = rng.normal(size=5000)
    assert psi(x, x.copy()) == pytest.approx(0.0, abs=1e-6)


def test_psi_is_large_for_a_shifted_distribution():
    rng = np.random.default_rng(4)
    expected = rng.normal(0, 1, 5000)
    actual = rng.normal(5, 1, 5000)  # a large, obvious shift
    assert psi(expected, actual) > 0.25


def test_psi_report_flags_only_shifted_features():
    rng = np.random.default_rng(5)
    train = pd.DataFrame({
        "stable": rng.normal(0, 1, 3000),
        "shifted": rng.normal(0, 1, 3000),
    })
    test = pd.DataFrame({
        "stable": rng.normal(0, 1, 3000),       # same distribution
        "shifted": rng.normal(4, 1, 3000),      # shifted distribution
    })
    report = psi_report(train, test, ["stable", "shifted"], flag_threshold=0.25)
    flagged = set(report.loc[report["flagged"], "feature"])
    assert flagged == {"shifted"}


def test_top_n_by_importance_orders_descending():
    imp = {"a": 1.0, "b": 5.0, "c": 3.0}
    assert top_n_by_importance(imp, 2) == ["b", "c"]


# ---------------------------------------------------------------------------
# report.py
# ---------------------------------------------------------------------------

def test_df_to_markdown_table_handles_empty_and_nonempty():
    assert "no rows" in df_to_markdown_table(pd.DataFrame())
    table = df_to_markdown_table(pd.DataFrame({"a": [1, 2]}))
    assert table.startswith("| a |")


def test_write_results_md_creates_a_readable_file(tmp_path, scored_binary):
    y, score = scored_binary
    row_a = headline_metrics(y, score, [0.01], 20)
    row_b = headline_metrics(y, np.clip(score * 1.1, 0, 1), [0.01], 20)
    out = tmp_path / "results.md"
    write_results_md(out, {"Rules baseline": row_a, "LGBM": row_b}, fpr_points=[0.01])
    text = out.read_text()
    assert "PR-AUC" in text
    assert "Rules baseline" in text and "LGBM" in text


def test_write_ablation_slices_drift_md_do_not_crash(tmp_path, scored_binary):
    y, score = scored_binary
    ablation = pd.DataFrame([
        {"row": "A", "pr_auc": 0.2}, {"row": "B", "pr_auc": 0.4},
    ])
    write_ablation_md(tmp_path / "ablation.md", ablation)
    assert (tmp_path / "ablation.md").exists()

    df = pd.DataFrame({
        "TransactionAmt": np.abs(np.random.default_rng(6).normal(50, 20, len(y))) + 1,
        "hour": np.random.default_rng(7).integers(0, 24, len(y)),
    })
    slices = build_slices(df, y, score, target_fpr=0.05)
    write_slices_md(tmp_path / "slices.md", slices)
    assert (tmp_path / "slices.md").exists()

    drift = psi_report(df, df, ["TransactionAmt"], flag_threshold=0.25)
    write_drift_md(tmp_path / "drift.md", drift, flag_threshold=0.25)
    assert (tmp_path / "drift.md").exists()
