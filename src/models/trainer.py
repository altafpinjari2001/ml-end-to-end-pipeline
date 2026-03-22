"""
ML End-to-End Pipeline - Model Trainer.

Supports multiple ML models with hyperparameter tuning.
"""

import logging
import joblib
from pathlib import Path
from typing import Optional, Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "logistic_regression": LogisticRegression,
}

# Optional imports for XGBoost and LightGBM
try:
    from xgboost import XGBClassifier
    MODEL_REGISTRY["xgboost"] = XGBClassifier
except ImportError:
    pass

try:
    from lightgbm import LGBMClassifier
    MODEL_REGISTRY["lightgbm"] = LGBMClassifier
except ImportError:
    pass


class ModelTrainer:
    """Multi-model trainer with hyperparameter tuning support."""

    def __init__(
        self,
        model_type: str = "xgboost",
        params: Optional[dict] = None,
    ):
        if model_type not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model: {model_type}. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )

        self.model_type = model_type
        self.model_cls = MODEL_REGISTRY[model_type]
        self.params = params or {}
        self.model = None

        logger.info(f"ModelTrainer: {model_type}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Any:
        """Train the model."""
        self.model = self.model_cls(**self.params)
        self.model.fit(X_train, y_train)
        logger.info(f"Model trained: {self.model_type}")
        return self.model

    def train_with_tuning(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: dict,
        cv: int = 5,
        scoring: str = "f1",
    ) -> Any:
        """Train with grid search hyperparameter tuning."""
        base_model = self.model_cls(**self.params)

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
        )
        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_
        logger.info(
            f"Best params: {grid_search.best_params_}"
        )
        logger.info(
            f"Best {scoring}: {grid_search.best_score_:.4f}"
        )
        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate probability predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict_proba(X)

    def save(self, path: str | Path) -> None:
        """Save the trained model."""
        joblib.dump(self.model, path)
        logger.info(f"Model saved: {path}")

    @classmethod
    def load(cls, path: str | Path) -> Any:
        """Load a trained model."""
        return joblib.load(path)
