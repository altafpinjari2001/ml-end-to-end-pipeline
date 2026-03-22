"""
ML End-to-End Pipeline - Feature Engineering.

Automated feature preprocessing pipeline using Scikit-learn.
"""

import logging
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Automated feature preprocessing pipeline."""

    def __init__(
        self,
        numeric_features: list[str],
        categorical_features: list[str],
    ):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

        # Numeric pipeline
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        # Categorical pipeline
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                drop="first",
                sparse_output=False,
                handle_unknown="ignore",
            )),
        ])

        # Combined pipeline
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_features),
                ("cat", categorical_pipeline, categorical_features),
            ],
            remainder="drop",
        )

        logger.info(
            f"FeaturePipeline initialized: "
            f"{len(numeric_features)} numeric, "
            f"{len(categorical_features)} categorical features"
        )

    def fit_transform(
        self, X: pd.DataFrame
    ) -> np.ndarray:
        """Fit the pipeline and transform the data."""
        result = self.preprocessor.fit_transform(X)
        logger.info(
            f"Fitted & transformed: {X.shape} → {result.shape}"
        )
        return result

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted pipeline."""
        return self.preprocessor.transform(X)

    def save(self, path: str | Path) -> None:
        """Save the fitted pipeline to disk."""
        joblib.dump(self.preprocessor, path)
        logger.info(f"Pipeline saved to: {path}")

    @classmethod
    def load(cls, path: str | Path) -> "FeaturePipeline":
        """Load a fitted pipeline from disk."""
        instance = cls.__new__(cls)
        instance.preprocessor = joblib.load(path)
        return instance
