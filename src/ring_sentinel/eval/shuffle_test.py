"""The shuffle test: retrain on time-shuffled labels, confirm the model dies."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score
from lightgbm import train as lgb_train, Dataset as lgb_Dataset

from ..features.base import build_base_features
from ..features.ring import build_ring_features
from ..models.gbdt import LGBMScorer


def run_shuffle_test(df, splits, cfg):
    """Retrain with real labels, then with shuffled labels. Compare PR-AUC."""
    rng = np.random.default_rng(cfg.seed)
    
    base_rate = float(df["isFraud"].mean())
    y = df["isFraud"].to_numpy()
    
    # Real model score
    feat, v_keep, email_maps = build_base_features(df, splits.train)
    ring_feat, _ = build_ring_features(df, base_rate, cfg)
    X_train = feat.iloc[splits.train]
    X_val = feat.iloc[splits.val]
    scorer = LGBMScorer(cfg)
    scorer.fit(X_train, y[splits.train], X_val, y[splits.val])
    y_pred_real = scorer.predict(feat.iloc[splits.test])
    pr_auc_real = float(average_precision_score(y[splits.test], y_pred_real))

    # Shuffled model score
    y_shuffled = y.copy()
    rng.shuffle(y_shuffled)
    scorer2 = LGBMScorer(cfg)
    scorer2.fit(X_train, y_shuffled[splits.train], X_val, y_shuffled[splits.val])
    y_pred_shuffled = scorer2.predict(feat.iloc[splits.test])
    pr_auc_shuffled = float(average_precision_score(y[splits.test], y_pred_shuffled))
    
    return {
        "base_rate": base_rate,
        "real_pr_auc": pr_auc_real,
        "shuffled_pr_auc": pr_auc_shuffled,
    }
