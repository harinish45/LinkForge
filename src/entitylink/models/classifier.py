"""Supervised machine learning pair matchers."""

import os
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from entitylink.data.schema import EntityRecord
from entitylink.models.base import BaseMatcher
from entitylink.features.pair_features import PairFeatureExtractor, FEATURE_NAMES


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
            return LogisticRegression(
                max_iter=1000,
                class_weight=self.class_weight,
                random_state=self.random_state,
            )
        elif self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight=self.class_weight,
                random_state=self.random_state,
                n_jobs=-1,
            )
        else:  # default: gradient_boosting
            return HistGradientBoostingClassifier(
                max_iter=150,
                max_depth=6,
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

        X = np.array([self.extractor.extract_vector(r1, r2) for r1, r2 in pairs])
        y = np.array(labels)

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
