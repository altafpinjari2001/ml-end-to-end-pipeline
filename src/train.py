"""
ML End-to-End Pipeline - Training Pipeline.

Orchestrates data loading, preprocessing, model training,
evaluation, and MLflow experiment tracking.
"""

import argparse
import logging
from pathlib import Path

import yaml
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split

from data.preprocessing import FeaturePipeline
from models.trainer import ModelTrainer
from models.evaluation import ModelEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)s │ %(levelname)s │ %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main(config_path: str):
    """Execute the full ML pipeline."""
    config = load_config(config_path)
    logger.info("🏗️ ML End-to-End Pipeline")
    logger.info("=" * 50)

    # ── Data Loading ─────────────────────────────────────────
    logger.info("Loading data...")
    df = pd.read_csv(config["data"]["train_path"])
    logger.info(f"Dataset shape: {df.shape}")

    target_col = config["data"]["target_column"]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"].get("test_size", 0.2),
        random_state=42,
        stratify=y,
    )

    # ── Feature Engineering ──────────────────────────────────
    logger.info("Building feature pipeline...")
    feature_pipeline = FeaturePipeline(
        numeric_features=config["features"]["numeric"],
        categorical_features=config["features"]["categorical"],
    )

    X_train_processed = feature_pipeline.fit_transform(X_train)
    X_test_processed = feature_pipeline.transform(X_test)

    # ── Model Training with MLflow ───────────────────────────
    mlflow.set_experiment(config.get("experiment_name", "ml-pipeline"))

    with mlflow.start_run(
        run_name=config.get("run_name", "training-run")
    ):
        # Log parameters
        mlflow.log_params(config.get("model_params", {}))

        # Train model
        logger.info(f"Training {config['model']['type']}...")
        trainer = ModelTrainer(
            model_type=config["model"]["type"],
            params=config.get("model_params", {}),
        )
        model = trainer.train(X_train_processed, y_train)

        # Evaluate
        evaluator = ModelEvaluator(model)
        metrics = evaluator.evaluate(
            X_test_processed, y_test
        )
        evaluator.plot_confusion_matrix(
            X_test_processed, y_test, save_path="artifacts/"
        )
        evaluator.plot_roc_curve(
            X_test_processed, y_test, save_path="artifacts/"
        )

        # Log metrics
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifacts("artifacts/")

        logger.info(f"📊 Metrics: {metrics}")
        logger.info("✅ Training complete! Model logged to MLflow.")

    # Save model
    output_dir = Path(config.get("output_dir", "outputs"))
    output_dir.mkdir(exist_ok=True)
    trainer.save(output_dir / "model.joblib")
    feature_pipeline.save(output_dir / "feature_pipeline.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
