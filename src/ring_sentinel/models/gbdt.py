"""LightGBM scorer with scale_pos_weight carefully set.

scale_pos_weight is set to 1.0, not to balance the positive class, because
downstream the policy engine consumes CALIBRATED PROBABILITIES as rupees.
Reweighting the positive class during training distorts the probability scale
that the cost model depends on. Rebalancing happens at the threshold, not in
the loss.
"""

from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np

log = logging.getLogger(__name__)


class LGBMScorer:
    def __init__(self, cfg=None, model_params: dict | None = None):
        from ..config import load_config
        self.cfg = cfg or load_config()
        self.mcfg = self.cfg.base["model"]
        self.model_params = model_params or self._default_params()
        self.bst = None
        self.feature_names = None
        self.feature_importance = None

    def _default_params(self) -> dict:
        return {
            "objective": self.mcfg["objective"],
            "metric": "binary_logloss",
            "num_leaves": int(self.mcfg["num_leaves"]),
            "learning_rate": float(self.mcfg["learning_rate"]),
            "feature_fraction": float(self.mcfg["feature_fraction"]),
            "bagging_fraction": float(self.mcfg["bagging_fraction"]),
            "bagging_freq": int(self.mcfg["bagging_freq"]),
            "lambda_l2": float(self.mcfg["lambda_l2"]),
            "max_bin": int(self.mcfg["max_bin"]),
            "min_child_samples": int(self.mcfg["min_child_samples"]),
            "is_unbalance": bool(self.mcfg["is_unbalance"]),
            "scale_pos_weight": float(self.mcfg["scale_pos_weight"]),
            "seed": self.cfg.seed,
            "num_threads": 4,
            "verbose": 0,
        }

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        categorical_features: list[str] | None = None,
    ):
        """Train with early stopping on the validation set."""
        n_est = int(self.mcfg["n_estimators"])
        early_stop = int(self.mcfg["early_stopping_rounds"])

        train_data = lgb.Dataset(
            X_train, label=y_train, feature_name=list(X_train.columns),
            categorical_feature=categorical_features or [],
        )

        val_data = None
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data,
                                   categorical_feature=categorical_features or [])

        callbacks = []
        if val_data is not None:
            callbacks = [
                lgb.early_stopping(early_stop, verbose=False),
                lgb.log_evaluation(period=0),
            ]

        self.bst = lgb.train(
            self.model_params,
            train_data,
            num_boost_round=n_est,
            valid_sets=[val_data] if val_data else None,
            callbacks=callbacks,
        )
        self.feature_names = list(X_train.columns)
        self.feature_importance = self.bst.feature_importance(importance_type="gain")
        log.info("Trained LightGBM: %d trees", self.bst.num_trees())
        return self

    def predict(self, X):
        """Predict probabilities."""
        if self.bst is None:
            raise RuntimeError("Model not fitted")
        return self.bst.predict(X)

    def shap_values(self, X):
        """SHAP values for explainability."""
        if self.bst is None:
            raise RuntimeError("Model not fitted")
        return self.bst.predict(X, pred_leaf=True, num_iteration=self.bst.num_trees())

    def get_params(self) -> dict:
        return {
            "model_params": self.model_params,
            "feature_names": self.feature_names,
            "n_trees": self.bst.num_trees() if self.bst else None,
        }
