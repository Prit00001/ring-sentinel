"""Base tabular feature block (build spec 7.1).

Everything fitted here — the V-block correlation prune, the email frequency
encoding — is fitted on the TRAIN SPLIT ONLY and then applied. A frequency
encoding fitted on the full frame is a quiet, popular leak.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import load_config

log = logging.getLogger(__name__)

C_COLS = [f"C{i}" for i in range(1, 15)]
D_COLS = [f"D{i}" for i in range(1, 16)]
M_COLS = [f"M{i}" for i in range(1, 10)]
ID_COLS = [f"id_{i:02d}" for i in range(1, 39)]


def v_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("V") and c[1:].isdigit()]


def prune_v_block(df: pd.DataFrame, train_idx: np.ndarray, cfg=None) -> list[str]:
    """Drop one of any V-pair with |r| > threshold, keeping higher coverage.

    The V block is heavily internally correlated and partly redundant with our
    own aggregates. Correlation is computed on a sample of the TRAIN split only.
    Returns the list of V columns to KEEP.
    """
    cfg = cfg or load_config()
    vp = cfg.features["v_pruning"]
    thresh = float(vp["corr_threshold"])
    sample_n = int(vp["sample_rows"])

    vcols = v_columns(df)
    if not vcols:
        return []

    tr = df.iloc[train_idx]
    if len(tr) > sample_n:
        rng = np.random.default_rng(cfg.seed)
        # Sample WITHOUT reordering time — we only need a correlation estimate,
        # but taking a contiguous tail would bias it, so sample positions.
        pos = np.sort(rng.choice(len(tr), size=sample_n, replace=False))
        tr = tr.iloc[pos]

    coverage = tr[vcols].notna().mean()
    sub = tr[vcols].astype("float32")
    corr = sub.corr().abs()

    keep, dropped = [], []
    kept_set: list[str] = []
    # Iterate by descending coverage so the higher-coverage member survives.
    for c in coverage.sort_values(ascending=False).index:
        clash = False
        for k in kept_set:
            r = corr.loc[c, k]
            if pd.notna(r) and r > thresh:
                clash = True
                break
        if clash:
            dropped.append(c)
        else:
            kept_set.append(c)
    keep = kept_set
    log.info("V-block prune: kept %d of %d (|r| > %.2f)", len(keep), len(vcols), thresh)
    return keep


def fit_email_freq(df: pd.DataFrame, train_idx: np.ndarray) -> dict[str, dict]:
    """Frequency encoding fitted on train rows only."""
    maps = {}
    tr = df.iloc[train_idx]
    for col in ("P_emaildomain", "R_emaildomain"):
        if col in df.columns:
            vc = tr[col].value_counts(normalize=True)
            maps[col] = vc.to_dict()
    return maps


def build_base_features(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    v_keep: list[str] | None = None,
    email_maps: dict | None = None,
    cfg=None,
) -> tuple[pd.DataFrame, list[str], dict]:
    """Return (features, v_keep, email_maps) so the fitted objects can be reused."""
    cfg = cfg or load_config()
    v_keep = v_keep if v_keep is not None else prune_v_block(df, train_idx, cfg)
    email_maps = email_maps if email_maps is not None else fit_email_freq(df, train_idx)

    out = pd.DataFrame(index=df.index)

    amt = df["TransactionAmt"].astype("float32")
    out["amount"] = amt
    out["amount_log"] = np.log1p(amt)
    # Cents fraction. Card testing and automated scripts produce suspiciously
    # round or suspiciously repeated cents values.
    out["amount_cents"] = (amt - np.floor(amt)).astype("float32")

    out["hour"] = df["hour"].astype("int16")
    out["day_of_week"] = df["day_of_week"].astype("int8")

    for col in ("card1", "card2", "card3", "card5", "addr1", "addr2"):
        if col in df.columns:
            out[col] = df[col].astype("float32")

    for col in ("ProductCD", "card4", "card6", "DeviceType", "P_emaildomain", "R_emaildomain", *M_COLS):
        if col in df.columns:
            out[col] = df[col].astype("category")

    for col, m in (email_maps or {}).items():
        if col in df.columns:
            out[f"{col}_freq"] = df[col].map(m).astype("float32").fillna(0.0)

    for col in C_COLS + D_COLS:
        if col in df.columns:
            out[col] = df[col].astype("float32")

    for col in v_keep:
        out[col] = df[col].astype("float32")

    # Missingness indicators for the identity block. Whether a row HAS device
    # data is itself informative, and it is not recoverable from the values.
    present = [c for c in ID_COLS if c in df.columns]
    if present:
        out["identity_present"] = df[present].notna().any(axis=1).astype("int8")
        out["identity_n_missing"] = df[present].isna().sum(axis=1).astype("int16")
    for col in ("DeviceInfo", "id_30", "id_31", "id_33"):
        if col in df.columns:
            out[f"{col}_missing"] = df[col].isna().astype("int8")

    return out, v_keep, email_maps


def categorical_columns(feat: pd.DataFrame) -> list[str]:
    return [c for c in feat.columns if str(feat[c].dtype) == "category"]
