"""
ML End-to-End Pipeline - FastAPI Application.

REST API for serving ML model predictions.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Prediction API",
    description="Customer Churn Prediction API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and pipeline at startup
MODEL_PATH = Path("outputs/model.joblib")
PIPELINE_PATH = Path("outputs/feature_pipeline.joblib")

model = None
pipeline = None


@app.on_event("startup")
def load_model():
    global model, pipeline
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded")
    if PIPELINE_PATH.exists():
        pipeline = joblib.load(PIPELINE_PATH)
        logger.info("Feature pipeline loaded")


class PredictionRequest(BaseModel):
    """Input schema for predictions."""
    tenure: int = Field(..., ge=0, description="Months of tenure")
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    contract: str = Field(..., description="Contract type")
    payment_method: str = Field(...)
    internet_service: str = Field(...)


class PredictionResponse(BaseModel):
    """Output schema for predictions."""
    prediction: str
    probability: float
    confidence: str


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.get("/model-info")
def model_info():
    if model is None:
        raise HTTPException(404, "Model not loaded")
    return {
        "model_type": type(model).__name__,
        "features": getattr(model, "n_features_in_", "N/A"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Generate a churn prediction."""
    if model is None:
        raise HTTPException(503, "Model not loaded")

    import pandas as pd
    input_df = pd.DataFrame([request.model_dump()])

    if pipeline:
        features = pipeline.transform(input_df)
    else:
        features = input_df.values

    proba = model.predict_proba(features)[0]
    prediction = int(proba[1] >= 0.5)
    confidence = "High" if max(proba) >= 0.8 else (
        "Medium" if max(proba) >= 0.6 else "Low"
    )

    return PredictionResponse(
        prediction="Churn" if prediction else "No Churn",
        probability=round(float(proba[1]), 4),
        confidence=confidence,
    )
