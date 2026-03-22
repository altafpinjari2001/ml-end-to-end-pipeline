"""
ML End-to-End Pipeline - Model Evaluation.

Comprehensive evaluation metrics and visualization.
"""

import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates ML models with comprehensive metrics."""

    def __init__(self, model):
        self.model = model

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        """
        Compute all evaluation metrics.

        Returns:
            Dictionary of metric name → value.
        """
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(
                y_test, y_pred, average="weighted"
            ),
            "recall": recall_score(
                y_test, y_pred, average="weighted"
            ),
            "f1_score": f1_score(
                y_test, y_pred, average="weighted"
            ),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        # Log classification report
        report = classification_report(y_test, y_pred)
        logger.info(f"\n{report}")

        return metrics

    def plot_confusion_matrix(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        save_path: str = "artifacts/",
    ) -> None:
        """Generate and save confusion matrix plot."""
        Path(save_path).mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_estimator(
            self.model, X_test, y_test,
            cmap="Blues", ax=ax,
        )
        ax.set_title("Confusion Matrix", fontsize=14)
        plt.tight_layout()
        plt.savefig(
            f"{save_path}/confusion_matrix.png", dpi=150
        )
        plt.close()
        logger.info("Confusion matrix saved")

    def plot_roc_curve(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        save_path: str = "artifacts/",
    ) -> None:
        """Generate and save ROC curve plot."""
        Path(save_path).mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        RocCurveDisplay.from_estimator(
            self.model, X_test, y_test, ax=ax
        )
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_title("ROC Curve", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{save_path}/roc_curve.png", dpi=150)
        plt.close()
        logger.info("ROC curve saved")
