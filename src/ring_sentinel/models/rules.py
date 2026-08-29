"""The rules baseline you must beat.

Three rules a risk analyst would write in an afternoon, from the build spec:

  R1  small amount AND the same uid transacted 3+ times in the past hour
  R2  a device never seen before AND a mismatched billing address (M-flags)
  R3  a component carrying 5+ distinct cards on a single device

Reporting its full metrics is not a formality. "Our model gets 0.72 PR-AUC" is
uninterpretable on its own; "our model gets 0.72 where three obvious rules get
0.31" is a result. The gap is the interesting number, and if the gap is small,
that is worth knowing before you ship a gradient booster into an ops team's
workflow.

The rule engine emits both a binary flag and a graded score (the count of rules
fired, plus tie-breaks). PR-AUC on a 4-value score is coarse by construction —
that limitation is stated in reports/results.md rather than hidden.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SMALL_AMOUNT_USD = 5.0
UID_VELOCITY_1H_MIN = 3
COMP_CARDS_PER_DEVICE_MIN = 5


def apply_rules(
    df: pd.DataFrame,
    entity_feat: pd.DataFrame,
    ring_feat: pd.DataFrame,
    small_amount_usd: float = SMALL_AMOUNT_USD,
) -> pd.DataFrame:
    """Return a frame with r1/r2/r3 booleans, `rules_flag`, and `rules_score`.

    Every input is a strictly-prior feature, so the baseline is held to exactly
    the same causality standard as the model. A baseline allowed to peek would
    make the model's win meaningless.
    """
    n = len(df)
    amt = df["TransactionAmt"].to_numpy(dtype=float)

    uid_v1h = entity_feat.get("uid_velocity_1h", pd.Series(np.zeros(n))).to_numpy()
    uid_prior = entity_feat.get("uid_prior_count", pd.Series(np.zeros(n))).to_numpy()
    dev_prior = entity_feat.get("device_prior_count", pd.Series(np.zeros(n))).to_numpy()

    # R1 — card testing shape: tiny amounts, repeated fast, same account.
    r1 = (amt < small_amount_usd) & (uid_v1h >= UID_VELOCITY_1H_MIN)

    # R2 — unseen device with an address/name mismatch on the card.
    # M-flags encode whether name and address on the card matched. 'F' is a
    # mismatch. Absent M data is NOT treated as a mismatch.
    m_cols = [c for c in ("M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9") if c in df.columns]
    if m_cols:
        mismatch = (df[m_cols] == "F").any(axis=1).to_numpy()
    else:
        mismatch = np.zeros(n, dtype=bool)
    new_device = dev_prior <= 0
    has_device = df["ent_device"].notna().to_numpy()
    r2 = new_device & has_device & mismatch

    # R3 — one device, many cards: the crudest possible ring rule.
    comp_n_dev = ring_feat.get("comp_n_device", pd.Series(np.zeros(n))).to_numpy()
    comp_addr_per_card = ring_feat.get("comp_addr_per_card", pd.Series(np.zeros(n))).to_numpy()
    comp_n_uid = ring_feat.get("comp_n_uid", pd.Series(np.zeros(n))).to_numpy()
    cards_per_device = np.where(comp_n_dev > 0, comp_n_uid / np.maximum(comp_n_dev, 1), 0.0)
    r3 = (comp_n_dev >= 1) & (cards_per_device >= COMP_CARDS_PER_DEVICE_MIN)

    flag = r1 | r2 | r3
    # Graded score so PR-AUC is computable. Rules fired dominates; uid velocity
    # breaks ties within a tier so the ordering is not arbitrary.
    fired = r1.astype(int) + r2.astype(int) + r3.astype(int)
    tiebreak = np.tanh(uid_v1h / 10.0) * 0.9
    score = fired.astype(float) + tiebreak

    return pd.DataFrame({
        "r1_small_amount_velocity": r1,
        "r2_new_device_addr_mismatch": r2,
        "r3_cards_per_device": r3,
        "rules_fired": fired,
        "rules_flag": flag,
        "rules_score": score / (3.0 + 0.9),   # scaled to [0, 1] for the metric code
    }, index=df.index)


def rule_firing_report(rules: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Per-rule precision, recall, and volume. Shows which rule does the work."""
    rows = []
    n_pos = int(y.sum())
    for col in ("r1_small_amount_velocity", "r2_new_device_addr_mismatch",
                "r3_cards_per_device", "rules_flag"):
        f = rules[col].to_numpy(dtype=bool)
        tp = int((f & (y == 1)).sum())
        fired = int(f.sum())
        rows.append({
            "rule": col,
            "n_fired": fired,
            "fire_rate_pct": round(100.0 * fired / len(y), 3),
            "true_positives": tp,
            "precision": round(tp / fired, 4) if fired else None,
            "recall": round(tp / n_pos, 4) if n_pos else None,
        })
    return pd.DataFrame(rows)
