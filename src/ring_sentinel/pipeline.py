"""End-to-end orchestration for `make features` / `make train` / `make eval`.

Every function here takes data in, rather than reaching into data/raw itself
— that is what makes them testable against the same synthetic fixture
`make smoke` already uses (see tests/test_pipeline.py), without needing the
2GB IEEE-CIS download. The Makefile targets are the only callers that point
these at data.load.load_raw() and real Kaggle files.

This is deliberately a thin, honest orchestration layer: it wires together
modules that already exist and are already unit-tested (entities, features,
models, policy, eval) rather than reimplementing any of their logic.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .data.split import Splits, temporal_split
from .entities.resolve import add_entities
from .eval.drift import psi_report, top_n_by_importance
from .eval.metrics import case_level_precision, headline_metrics
from .eval.report import write_ablation_md, write_drift_md, write_results_md, write_slices_md
from .eval.slices import build_slices
from .features.base import build_base_features
from .features.causal import assert_label_lag_respected, assert_strictly_prior, drop_proof_columns
from .features.ring import build_ring_features
from .models.calibrate import IsotonicCalibrator, calibration_report
from .models.gbdt import LGBMScorer
from .models.rules import apply_rules
from .policy.aggregate import case_score
from .policy.engine import expected_costs

log = logging.getLogger(__name__)


@dataclass
class FeatureBundle:
    df: pd.DataFrame            # entity-resolved, sorted, with day/hour
    splits: Splits
    base_feat: pd.DataFrame     # proof columns already dropped
    ring_feat: pd.DataFrame     # proof columns already dropped
    v_keep: list
    email_maps: dict
    train_prior: float
    categorical_features: list


def run_features(df: pd.DataFrame, cfg=None) -> FeatureBundle:
    """Entity resolution -> temporal split -> base + ring features, with the
    causality assertions run inline so a leak fails the build, not just the
    test suite (build spec 7.4)."""
    cfg = cfg or load_config()

    df = add_entities(df)
    splits = temporal_split(df, cfg)
    train_prior = float(df["isFraud"].iloc[splits.train].mean()) if "isFraud" in df.columns else 0.0

    base_feat, v_keep, email_maps = build_base_features(df, splits.train, cfg=cfg)
    ring_feat, graph_stats = build_ring_features(df, train_prior, cfg=cfg, progress_every=0)

    dt = df["TransactionDT"].to_numpy()
    assert_strictly_prior(ring_feat, dt, columns=["comp_max_contrib_dt"])
    lag_days = int(cfg.base["features"]["label_lag_days"])
    assert_label_lag_respected(ring_feat, dt, lag_days)

    ring_feat = drop_proof_columns(ring_feat)
    categorical_features = [c for c in base_feat.columns if base_feat[c].dtype.name == "category"]

    log.info("Feature build: %d base cols, %d ring cols, graph windows=%d",
              base_feat.shape[1], ring_feat.shape[1], len(graph_stats))

    return FeatureBundle(
        df=df, splits=splits, base_feat=base_feat, ring_feat=ring_feat,
        v_keep=v_keep, email_maps=email_maps, train_prior=train_prior,
        categorical_features=categorical_features,
    )


def persist_features(bundle: FeatureBundle, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    full = pd.concat([bundle.base_feat, bundle.ring_feat], axis=1)
    if "isFraud" in bundle.df.columns:
        full["isFraud"] = bundle.df["isFraud"].to_numpy()
    full["day"] = bundle.df["day"].to_numpy()
    full.to_parquet(out_dir / "features.parquet")

    (out_dir / "meta.json").write_text(json.dumps({
        "v_keep": bundle.v_keep,
        "categorical_features": bundle.categorical_features,
        "train_prior": bundle.train_prior,
        "splits": {k: v.tolist() for k, v in bundle.splits.as_dict().items()},
    }))


@dataclass
class TrainResult:
    rules: pd.DataFrame
    scorer_base: LGBMScorer     # base tabular features only (ablation row B)
    scorer_full: LGBMScorer     # base + ring features (ablation row D)
    calibrator: IsotonicCalibrator
    feature_importance: dict


def run_train(bundle: FeatureBundle, cfg=None) -> TrainResult:
    cfg = cfg or load_config()
    y = bundle.df["isFraud"].to_numpy()
    tr, va = bundle.splits.train, bundle.splits.val

    rules = apply_rules(bundle.df, bundle.ring_feat, bundle.ring_feat)

    X_base = bundle.base_feat
    X_full = pd.concat([bundle.base_feat, bundle.ring_feat.drop(columns=["comp_id"], errors="ignore")], axis=1)

    scorer_base = LGBMScorer(cfg).fit(
        X_base.iloc[tr], y[tr], X_base.iloc[va], y[va],
        categorical_features=bundle.categorical_features,
    )
    scorer_full = LGBMScorer(cfg).fit(
        X_full.iloc[tr], y[tr], X_full.iloc[va], y[va],
        categorical_features=bundle.categorical_features,
    )

    val_scores = scorer_full.predict(X_full.iloc[va])
    calibrator = IsotonicCalibrator().fit(val_scores, y[va], fit_index=va)

    importance = dict(zip(scorer_full.feature_names, scorer_full.feature_importance.tolist()))

    return TrainResult(
        rules=rules, scorer_base=scorer_base, scorer_full=scorer_full,
        calibrator=calibrator, feature_importance=importance,
    )


def run_eval(bundle: FeatureBundle, train_result: TrainResult, cfg=None, out_dir: Path | None = None) -> dict:
    """Ablation rows A/B/D (rules / base LightGBM / + ring features) plus
    calibration, slices, and drift on the test split.

    Rows C (causal aggregates alone) and E (case aggregation) are not
    computed here — they would need a third scorer trained on a
    causal-only feature set and a noisy-OR case aggregator respectively,
    which is real additional scope beyond wiring the eval layer this pass
    covers. That gap is reported, not hidden, in the returned dict.
    """
    cfg = cfg or load_config()
    ecfg = cfg.base["eval"]
    y = bundle.df["isFraud"].to_numpy()
    test = bundle.splits.test
    day = bundle.df["day"].to_numpy()[test]

    X_base = bundle.base_feat
    X_full = pd.concat([bundle.base_feat, bundle.ring_feat.drop(columns=["comp_id"], errors="ignore")], axis=1)

    rows = {}
    rows["A_rules_only"] = headline_metrics(
        y[test], train_result.rules["rules_score"].to_numpy()[test],
        ecfg["fpr_points"], ecfg["precision_at_k_per_day"], ecfg["ece_bins"], day=day,
    )
    rows["B_lgbm_base"] = headline_metrics(
        y[test], train_result.scorer_base.predict(X_base.iloc[test]),
        ecfg["fpr_points"], ecfg["precision_at_k_per_day"], ecfg["ece_bins"], day=day,
    )
    raw_full_scores = train_result.scorer_full.predict(X_full.iloc[test])
    rows["D_plus_ring_features"] = headline_metrics(
        y[test], raw_full_scores,
        ecfg["fpr_points"], ecfg["precision_at_k_per_day"], ecfg["ece_bins"], day=day,
    )
    calibrated_scores = train_result.calibrator.transform(raw_full_scores)
    rows["F_plus_calibration"] = headline_metrics(
        y[test], calibrated_scores,
        ecfg["fpr_points"], ecfg["precision_at_k_per_day"], ecfg["ece_bins"], day=day,
    )

    cal_report = calibration_report(y[test], calibrated_scores, ecfg["ece_bins"])

    slices = build_slices(bundle.df.iloc[test], y[test], calibrated_scores, target_fpr=ecfg["fpr_points"][1])

    top_features = top_n_by_importance(train_result.feature_importance, ecfg["psi_top_features"])
    drift = psi_report(
        X_full.iloc[bundle.splits.train], X_full.iloc[test], top_features,
        flag_threshold=ecfg["psi_flag_threshold"],
    )

    result = {
        "headline": rows,
        "calibration": cal_report,
        "slices": slices,
        "drift": drift,
        "note": (
            "Ablation rows C (causal aggregates alone) and E (case-level "
            "noisy-OR aggregation) are not computed by this pipeline run — "
            "only A, B, D, F. See run_eval()'s docstring."
        ),
    }

    if out_dir is not None:
        write_results_md(Path(out_dir) / "results.md", rows, ecfg["fpr_points"])
        write_slices_md(Path(out_dir) / "slices.md", slices)
        write_drift_md(Path(out_dir) / "drift.md", drift, ecfg["psi_flag_threshold"])

    return result


def build_case_queue(
    bundle: FeatureBundle,
    train_result: TrainResult,
    cfg=None,
    top_n: int = 20,
    out_path: Path | None = None,
    report_dir: Path | None = None,
) -> list[dict]:
    """Real cases for the analyst console, scored on the held-out test split.

    Transaction scores are calibrated, then rolled up to a case score per
    `comp_id` (the window-graph component each transaction attached to; see
    features/ring.py) via the noisy-OR aggregation from build spec section 8
    — this is ablation row E, which run_eval() does not compute. Rows with
    comp_id == -1 ("no prior component" — a first sighting with nothing to
    attach to) are excluded; a case queue is about rings, not singletons.

    Returns a list of dicts matching serve/api.py's case schema exactly, so
    the console can load real cases the same way it loads the demo ones.

    If `report_dir` is given, also writes reports/economics.md: case-level
    precision (spec section 8's definition) and total expected cost/saving
    under the policy, aggregated over EVERY ring in the test period — not
    just the top `top_n` shown to an analyst. Without this, the only
    economics anyone sees are the top-20 queue, which is already the
    easiest, highest-confidence slice and overstates precision.
    """
    cfg = cfg or load_config()
    costs_cfg = cfg.costs
    usd_inr = float(costs_cfg["usd_inr"])
    test = bundle.splits.test

    X_full = pd.concat(
        [bundle.base_feat, bundle.ring_feat.drop(columns=["comp_id"], errors="ignore")], axis=1
    )
    raw_scores = train_result.scorer_full.predict(X_full.iloc[test])
    scores = train_result.calibrator.transform(raw_scores)

    df_test = bundle.df.iloc[test].copy()
    df_test["_score"] = scores
    df_test["_comp_id"] = bundle.ring_feat["comp_id"].to_numpy()[test]
    df_test = df_test[df_test["_comp_id"] >= 0]

    top_feature_name, top_feature_gain = max(
        train_result.feature_importance.items(), key=lambda kv: kv[1], default=(None, 0.0)
    )
    top_gain_max = max(train_result.feature_importance.values(), default=1.0) or 1.0

    cases = []
    for comp_id, g in df_test.groupby("_comp_id"):
        if len(g) < 2:
            continue  # a "ring" of one transaction is not a ring
        score = case_score(g["_score"].tolist())
        amount_total_inr = float(g["TransactionAmt"].sum()) * usd_inr
        c_allow, c_block, c_review = expected_costs(score, amount_total_inr, costs_cfg)
        costs_by_action = {"allow": c_allow, "block": c_block, "review": c_review}
        decision = min(costs_by_action.items(), key=lambda x: x[1])[0]
        # "Expected saving" is what the analyst console actually wants to show:
        # cost avoided relative to doing nothing (allowing the ring through),
        # not the raw cost of the chosen action — those are the same number
        # only when the policy chooses to allow, and near a saturated case
        # score (block cost -> 0) the raw cost alone collapses to ~0 for
        # every large ring, which is uninformative for ranking cases.
        expected_saving = c_allow - costs_by_action[decision]

        shared_entity = None
        for etype in ("device", "addr", "card", "uid"):
            counts = g[f"ent_{etype}"].value_counts()
            if not counts.empty:
                shared_entity = {"type": etype, "value": str(counts.index[0]), "n_uid": int(g["ent_uid"].nunique())}
                break
        if shared_entity is None:
            shared_entity = {"type": "uid", "value": "n/a", "n_uid": int(g["ent_uid"].nunique())}

        last = g.sort_values("TransactionDT").iloc[-1]
        top_feature_value = 0.0
        if top_feature_name in X_full.columns:
            raw = X_full.loc[last.name, top_feature_name]
            if pd.notna(raw):
                top_feature_value = float(raw)

        cases.append({
            "case_id": f"case_{int(comp_id)}",
            "score": round(score, 4),
            "decision": decision,
            "expected_cost": round(costs_by_action[decision], 2),
            "expected_saving": round(expected_saving, 2),
            "n_uid": int(g["ent_uid"].nunique()),
            "n_device": int(g["ent_device"].nunique()),
            "n_transactions": int(len(g)),
            "amount_total_inr": round(amount_total_inr, 2),
            "amount_cv": round(float(g["TransactionAmt"].std(ddof=0) / g["TransactionAmt"].mean()), 4)
                         if g["TransactionAmt"].mean() else 0.0,
            "age_hours": round(float((g["TransactionDT"].max() - g["TransactionDT"].min()) / 3600.0), 2),
            "velocity_24h": int((g["TransactionDT"].max() - g["TransactionDT"] <= 86400).sum()),
            "day_start": int(g["day"].min()),
            "day_end": int(g["day"].max()),
            "shared_entity": shared_entity,
            "top_feature": {
                "feature": top_feature_name or "n/a",
                "value": round(top_feature_value, 4),
                "shap": round(top_feature_gain / top_gain_max, 4),  # normalised gain, NOT real SHAP
            },
            "n_labeled": int(len(g)),
            "n_fraud": int(g["isFraud"].sum()) if "isFraud" in g.columns else 0,
            "analyst_decision": None,
            "decided_at": None,
        })

    if report_dir is not None and cases:
        flagged = np.array([c["decision"] in ("block", "review") for c in cases])
        has_fraud = np.array([c["n_fraud"] > 0 for c in cases])
        precision = case_level_precision(flagged, has_fraud)
        total_cost = sum(c["expected_cost"] for c in cases)
        total_saving = sum(c["expected_saving"] for c in cases)
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        text = (
            "# Ring Sentinel — Case economics (full test period)\n\n"
            "Every ring in the test split with >=2 transactions, not just the "
            "top-N shown to an analyst. `case_level_precision` = fraction of "
            "block/review decisions containing >=1 confirmed-fraud "
            "transaction (build spec section 8's definition).\n\n"
            "| n_cases | n_flagged | case_level_precision | total_expected_cost_inr | total_expected_saving_inr |\n"
            "|---|---|---|---|---|\n"
            f"| {len(cases)} | {int(flagged.sum())} | {precision:.4f} "
            f"| {total_cost:,.2f} | {total_saving:,.2f} |\n"
        )
        (Path(report_dir) / "economics.md").write_text(text)

    # Sort by expected saving, not raw score: the noisy-OR score saturates to
    # (numerically near-)1.0 for any reasonably large, reasonably confident
    # ring, so score alone barely differentiates the top cases. Saving scales
    # with amount and is what an analyst queue is actually supposed to
    # prioritise ("show me the biggest wins first").
    cases.sort(key=lambda c: c["expected_saving"], reverse=True)
    top_cases = cases[:top_n]

    log.info("Case queue: %d candidate rings on the test split, showing top %d", len(cases), len(top_cases))

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(top_cases, indent=2))

    return top_cases
