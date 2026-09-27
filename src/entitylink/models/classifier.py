"""Supervised machine learning pair matchers with zero-dependency pure-NumPy fallbacks."""

import os
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np

from entitylink.data.schema import EntityRecord
from entitylink.models.base import BaseMatcher
from entitylink.features.pair_features import PairFeatureExtractor, FEATURE_NAMES


class FastSGDLogisticRegression:
    """High-speed pure-NumPy Logistic Regression with balanced weighting and mini-batch SGD."""

    def __init__(
        self,
        lr: float = 0.05,
        n_epochs: int = 30,
        l2_reg: float = 1e-4,
        batch_size: int = 256,
        class_weight: str = "balanced",
        random_state: int = 42,
    ):
        self.lr = lr
        self.n_epochs = n_epochs
        self.l2_reg = l2_reg
        self.batch_size = batch_size
        self.class_weight = class_weight
        self.random_state = random_state
        self.weights = None
        self.bias = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FastSGDLogisticRegression":
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0

        if self.class_weight == "balanced":
            n_pos = np.sum(y == 1)
            n_neg = np.sum(y == 0)
            w_pos = n_samples / (2.0 * max(n_pos, 1))
            w_neg = n_samples / (2.0 * max(n_neg, 1))
            sample_weights = np.where(y == 1, w_pos, w_neg)
        else:
            sample_weights = np.ones(n_samples, dtype=np.float64)

        indices = np.arange(n_samples)
        for epoch in range(self.n_epochs):
            rng.shuffle(indices)
            lr_t = self.lr / (1.0 + 0.05 * epoch)
            for start in range(0, n_samples, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                X_b = X[batch_idx]
                y_b = y[batch_idx]
                w_b = sample_weights[batch_idx]

                linear = np.dot(X_b, self.weights) + self.bias
                p = 1.0 / (1.0 + np.exp(-np.clip(linear, -25.0, 25.0)))
                err = (p - y_b) * w_b

                grad_w = np.dot(X_b.T, err) / len(batch_idx) + self.l2_reg * self.weights
                grad_b = float(np.mean(err))

                self.weights -= lr_t * grad_w
                self.bias -= lr_t * grad_b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        linear = np.dot(X, self.weights) + self.bias
        p = 1.0 / (1.0 + np.exp(-np.clip(linear, -25.0, 25.0)))
        p = np.clip(p, 1e-7, 1.0 - 1e-7)
        return np.column_stack([1.0 - p, p])


class DecisionStump:
    """Fast decision stump for gradient boosting."""

    def __init__(self):
        self.feature_idx = 0
        self.threshold = 0.0
        self.left_val = 0.0
        self.right_val = 0.0

    def fit(self, X: np.ndarray, residuals: np.ndarray, rng: np.random.RandomState, max_candidates: int = 15) -> "DecisionStump":
        n_samples, n_features = X.shape
        best_loss = float("inf")

        for feat in range(n_features):
            col = X[:, feat]
            vals = np.unique(col)
            if len(vals) <= 1:
                continue
            if len(vals) > max_candidates:
                quantiles = np.linspace(0.05, 0.95, max_candidates)
                thresholds = np.quantile(vals, quantiles)
            else:
                thresholds = (vals[:-1] + vals[1:]) / 2.0

            for th in thresholds:
                left_mask = col <= th
                n_left = int(np.sum(left_mask))
                n_right = n_samples - n_left
                if n_left == 0 or n_right == 0:
                    continue
                sum_left = float(np.sum(residuals[left_mask]))
                sum_right = float(np.sum(residuals[~left_mask]))
                # Fast variance reduction criterion: - (sum_left^2 / n_left + sum_right^2 / n_right)
                loss = -( (sum_left ** 2) / n_left + (sum_right ** 2) / n_right )
                if loss < best_loss:
                    best_loss = loss
                    self.feature_idx = feat
                    self.threshold = th
                    self.left_val = sum_left / n_left
                    self.right_val = sum_right / n_right
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        left_mask = X[:, self.feature_idx] <= self.threshold
        out = np.empty(len(X), dtype=np.float64)
        out[left_mask] = self.left_val
        out[~left_mask] = self.right_val
        return out


class FastGradientBoostingClassifier:
    """Pure-NumPy Gradient Boosted Decision Stumps classifier."""

    def __init__(self, n_estimators: int = 50, learning_rate: float = 0.1, random_state: int = 42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.trees: List[DecisionStump] = []
        self.init_val = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FastGradientBoostingClassifier":
        rng = np.random.RandomState(self.random_state)
        p_pos = np.clip(np.mean(y), 1e-4, 1.0 - 1e-4)
        self.init_val = float(np.log(p_pos / (1.0 - p_pos)))
        F = np.full(len(y), self.init_val, dtype=np.float64)

        for _ in range(self.n_estimators):
            p = 1.0 / (1.0 + np.exp(-np.clip(F, -20.0, 20.0)))
            residuals = y - p
            stump = DecisionStump().fit(X, residuals, rng)
            pred = stump.predict(X)
            F += self.learning_rate * pred
            self.trees.append(stump)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        F = np.full(len(X), self.init_val, dtype=np.float64)
        for stump in self.trees:
            F += self.learning_rate * stump.predict(X)
        p = 1.0 / (1.0 + np.exp(-np.clip(F, -20.0, 20.0)))
        return np.column_stack([1.0 - p, p])



class MLPairMatcher(BaseMatcher):
    """Supervised pair classifier trained on engineered similarity features."""

    def __init__(
        self,
        model_type: str = "gradient_boosting",
        random_state: int = 42,
        class_weight: str = "balanced",
    ):
        self.model_type = model_type
        self.random_state = random_state
        self.class_weight = class_weight
        self.extractor = PairFeatureExtractor(FEATURE_NAMES)
        self.model = self._init_model()
        self.is_fitted = False

    def _init_model(self):
        if self.model_type == "logistic_regression":
            return FastSGDLogisticRegression(
                class_weight=self.class_weight,
                random_state=self.random_state,
            )
        else:  # default: gradient_boosting
            return FastGradientBoostingClassifier(
                n_estimators=50,
                learning_rate=0.1,
                random_state=self.random_state,
            )

    def fit(
        self,
        pairs: List[Tuple[EntityRecord, EntityRecord]],
        labels: List[int],
    ) -> "MLPairMatcher":
        """Fit model on training pairs and binary match labels (1=match, 0=non-match)."""
        if not pairs:
            raise ValueError("Cannot fit on empty pair list.")

        X = np.array([self.extractor.extract_vector(r1, r2) for r1, r2 in pairs], dtype=np.float64)
        y = np.array(labels, dtype=np.int32)

        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def score_pair(
        self,
        record1: EntityRecord,
        record2: EntityRecord,
    ) -> float:
        """Predict match probability using trained classifier."""
        if not self.is_fitted:
            # Fallback to feature average if unfitted
            feats = self.extractor.extract_features(record1, record2)
            return float(feats.get("combined_geom_sim", 0.0))

        feat_vector = np.array([self.extractor.extract_vector(record1, record2)])
        proba = self.model.predict_proba(feat_vector)[0][1]
        return float(proba)

    def save(self, filepath: str) -> None:
        """Persist model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"model": self.model, "type": self.model_type, "fitted": self.is_fitted}, filepath)

    @classmethod
    def load(cls, filepath: str) -> "MLPairMatcher":
        """Load persisted model from disk."""
        data = joblib.load(filepath)
        matcher = cls(model_type=data["type"])
        matcher.model = data["model"]
        matcher.is_fitted = data["fitted"]
        return matcher
