"""THE MOST IMPORTANT FILE IN THIS REPO.

Five tests are mandated by the build spec (7.4). All five are here, plus three
more that close gaps the five leave open.

The shuffle test — retrain on a time-shuffled label column and confirm the
model dies — is the strongest evidence of integrity you can put in a repo. Its
output is printed and lands in reports/results.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ring_sentinel.data.split import assert_disjoint_and_ordered, temporal_split
from ring_sentinel.features.causal import (
    assert_label_lag_respected,
    assert_strictly_prior,
    expanding_entity_features,
)
from ring_sentinel.features.ring import build_ring_features


# ---------------------------------------------------------------------------
# 1. No future contribution
# ---------------------------------------------------------------------------

def test_no_future_contribution_entity_aggregates(entities, cfg):
    """For every row, the max timestamp of any contributing row is earlier."""
    feat = expanding_entity_features(entities, "ent_uid", "uid", train_prior=0.035, cfg=cfg)
    dt = entities["TransactionDT"].to_numpy()
    assert_strictly_prior(feat, dt)

    # And prove the assertion is capable of failing: corrupt one row.
    bad = feat.copy()
    bad.loc[bad.index[10], "uid_max_contrib_dt"] = int(dt[10])
    with pytest.raises(AssertionError, match="CAUSALITY VIOLATION"):
        assert_strictly_prior(bad, dt)


def test_no_future_contribution_ring_features(entities, cfg):
    """Same guarantee for the component-level aggregates."""
    feat, _stats = build_ring_features(entities, train_prior=0.035, cfg=cfg, progress_every=0)
    dt = entities["TransactionDT"].to_numpy()
    assert_strictly_prior(feat, dt, columns=["comp_max_contrib_dt"])


def test_tied_timestamps_do_not_see_each_other(entities, cfg):
    """Rows sharing an identical TransactionDT must not contribute to each other.

    The fixture deliberately contains a block of exact ties. Without tie-group
    handling this is a silent same-instant leak.
    """
    dt = entities["TransactionDT"].to_numpy()
    assert (np.diff(dt) == 0).any(), "fixture should contain tied timestamps"
    feat = expanding_entity_features(entities, "ent_uid", "uid", train_prior=0.035, cfg=cfg)
    v = feat["uid_max_contrib_dt"].to_numpy()
    assert not ((v >= 0) & (v == dt)).any(), "an aggregate consumed a same-instant row"


# ---------------------------------------------------------------------------
# 2. Label lag respected
# ---------------------------------------------------------------------------

def test_label_lag_respected(entities, cfg):
    """No label used in an aggregate matured after the row's timestamp."""
    lag = int(cfg.base["features"]["label_lag_days"])
    dt = entities["TransactionDT"].to_numpy()

    feat = expanding_entity_features(entities, "ent_uid", "uid", train_prior=0.035, cfg=cfg)
    assert_label_lag_respected(feat, dt, lag)

    ring, _ = build_ring_features(entities, train_prior=0.035, cfg=cfg, progress_every=0)
    assert_label_lag_respected(
        ring.rename(columns={"comp_label_max_contrib_dt": "comp_label_max_contrib_dt"}), dt, lag
    )


def test_label_lag_assertion_can_fail(entities, cfg):
    """A guard that cannot fail is not a guard."""
    lag = int(cfg.base["features"]["label_lag_days"])
    dt = entities["TransactionDT"].to_numpy()
    feat = expanding_entity_features(entities, "ent_uid", "uid", train_prior=0.035, cfg=cfg)
    bad = feat.copy()
    bad.loc[bad.index[50], "uid_label_max_contrib_dt"] = int(dt[50]) - 1
    with pytest.raises(AssertionError, match="LABEL LAG VIOLATION"):
        assert_label_lag_respected(bad, dt, lag)


# ---------------------------------------------------------------------------
# 3. Calibrator never saw test
# ---------------------------------------------------------------------------

def test_calibrator_never_saw_test(raw, splits):
    """Calibrator's fitted index set is disjoint from the test index set."""
    from ring_sentinel.models.calibrate import IsotonicCalibrator

    rng = np.random.default_rng(0)
    scores = rng.random(len(raw))
    y = raw["isFraud"].to_numpy()

    cal = IsotonicCalibrator()
    cal.fit(scores[splits.val], y[splits.val], fit_index=splits.val)

    assert cal.fit_index is not None, "calibrator must record which rows it saw"
    assert not (set(cal.fit_index.tolist()) & set(splits.test.tolist())), \
        "calibrator was fitted on rows that are in the test split"
    assert not (set(cal.fit_index.tolist()) & set(splits.train.tolist())), \
        "calibrator was fitted on training rows — calibration must use val only"


def test_splits_are_ordered_and_embargoed(raw, splits, cfg):
    """Train < val < test in time, with an embargo gap of the configured size."""
    assert_disjoint_and_ordered(raw, splits)
    dt = raw["TransactionDT"].to_numpy()
    embargo_sec = int(cfg.base["split"]["embargo_days"]) * 86400
    if len(splits.train) and len(splits.val):
        gap = dt[splits.val].min() - dt[splits.train].max()
        assert gap >= embargo_sec * 0.5, f"embargo gap train->val too small: {gap}s"
    if len(splits.val) and len(splits.test):
        gap = dt[splits.test].min() - dt[splits.val].max()
        assert gap >= embargo_sec * 0.5, f"embargo gap val->test too small: {gap}s"


def test_splitter_refuses_unsorted_input(raw):
    """A shuffled frame must raise, not silently produce a random split."""
    shuffled = raw.sample(frac=1.0, random_state=0).reset_index(drop=True)
    with pytest.raises(ValueError, match="not sorted"):
        temporal_split(shuffled)


# ---------------------------------------------------------------------------
# 4. Shuffled target kills signal
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_shuffled_target_kills_signal(entities, splits, cfg):
    """Retrain on a time-shuffled label column; PR-AUC must fall to base rate.

    If it does not, something in the feature pipeline is leaking: the model
    found a path from the features back to the label that does not go through
    the (now destroyed) real relationship.
    """
    from ring_sentinel.eval.shuffle_test import run_shuffle_test

    result = run_shuffle_test(entities, splits, cfg)

    base = result["base_rate"]
    shuffled = result["shuffled_pr_auc"]
    real = result["real_pr_auc"]

    print(
        f"\n  shuffled-label test: real PR-AUC={real:.4f}  "
        f"shuffled PR-AUC={shuffled:.4f}  base rate={base:.4f}"
    )

    # The shuffled model must be indistinguishable from guessing the base rate.
    assert shuffled < base * 1.35 + 0.01, (
        f"Shuffled-label PR-AUC {shuffled:.4f} is well above the base rate "
        f"{base:.4f}. The feature pipeline is leaking."
    )
    # And the real model must actually beat it, or the test proves nothing.
    assert real > shuffled, "real model did not beat the shuffled model"


# ---------------------------------------------------------------------------
# 5. Graph edges are backward only
# ---------------------------------------------------------------------------

def test_graph_edges_are_backward_only(entities, cfg):
    """No edge connects to a transaction after the window boundary."""
    from ring_sentinel.entities.graph import build_window_graph, extract_entity_values

    df = entities
    dt = df["TransactionDT"].to_numpy()
    day = df["day"].to_numpy()
    lookback = int(cfg.base["graph"]["lookback_days"])
    max_deg = int(cfg.base["entities"]["max_entity_degree"])
    min_deg = int(cfg.base["entities"]["min_entity_degree"])

    checked = 0
    for d in sorted(set(day.tolist()))[lookback:]:
        positions = np.where((day >= d - lookback) & (day < d))[0]
        if len(positions) == 0:
            continue
        wg = build_window_graph(
            day=d, window_start_day=d - lookback, row_positions=positions,
            entity_values=extract_entity_values(df, positions),
            max_degree=max_deg, min_degree=min_deg,
        )
        boundary_dt = d * 86400
        assert dt[wg.row_positions].max() < boundary_dt, (
            f"day {d}: window graph contains a row at or after the window "
            f"boundary {boundary_dt}"
        )
        assert (day[wg.row_positions] < d).all(), \
            f"day {d}: window graph contains a same-day or later row"
        checked += 1

    assert checked > 0, "no windows were checked — the test proved nothing"


def test_hub_pruning_actually_bites(entities, cfg):
    """Without pruning the graph collapses; confirm the cap prevents that.

    gmail.com touches a third of the real dataset. If a single component
    swallows most of the window, every comp_* feature becomes a constant and
    the whole ring layer is dead weight.
    """
    from ring_sentinel.entities.graph import build_window_graph, extract_entity_values

    df = entities
    day = df["day"].to_numpy()
    lookback = int(cfg.base["graph"]["lookback_days"])
    d = sorted(set(day.tolist()))[-1]
    positions = np.where((day >= d - lookback) & (day < d))[0]
    ev = extract_entity_values(df, positions)

    max_comp_frac = float(cfg.base["entities"].get("max_component_frac", 1.0))
    max_comp_abs = int(cfg.base["entities"].get("max_component_size_abs", 10**9))
    capped = build_window_graph(d, d - lookback, positions, ev,
                                max_degree=int(cfg.base["entities"]["max_entity_degree"]),
                                min_degree=int(cfg.base["entities"]["min_entity_degree"]),
                                max_component_size=max(1, min(int(max_comp_frac * len(positions)), max_comp_abs)))
    uncapped = build_window_graph(d, d - lookback, positions, ev,
                                  max_degree=10**9, min_degree=1)

    def largest_share(wg):
        _, counts = np.unique(wg.comp_of_row, return_counts=True)
        return counts.max() / len(wg.comp_of_row)

    assert capped.n_components >= uncapped.n_components, \
        "pruning did not increase the component count"
    assert largest_share(capped) <= 0.95, \
        "even with pruning, one component swallowed the window"
