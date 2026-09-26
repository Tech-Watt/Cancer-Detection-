"""FastAPI service for cancer risk-level predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "cancer_risk_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

FEATURE_FIELDS = [
    "Age",
    "Gender",
    "Air Pollution",
    "Alcohol use",
    "Dust Allergy",
    "OccuPational Hazards",
    "Genetic Risk",
    "chronic Lung Disease",
    "Balanced Diet",
    "Obesity",
    "Smoking",
    "Passive Smoker",
    "Chest Pain",
    "Coughing of Blood",
    "Fatigue",
    "Weight Loss",
    "Shortness of Breath",
    "Wheezing",
    "Swallowing Difficulty",
    "Clubbing of Finger Nails",
    "Frequent Cold",
    "Dry Cough",
    "Snoring",
]


class PatientFeatures(BaseModel):
    Age: int = Field(..., ge=1, le=120, description="Patient age in years")
    Gender: Literal[1, 2] = Field(..., description="1 or 2, matching the training data encoding")
    air_pollution: int = Field(..., alias="Air Pollution", ge=1, le=9)
    alcohol_use: int = Field(..., alias="Alcohol use", ge=1, le=9)
    dust_allergy: int = Field(..., alias="Dust Allergy", ge=1, le=9)
    occupational_hazards: int = Field(..., alias="OccuPational Hazards", ge=1, le=9)
    genetic_risk: int = Field(..., alias="Genetic Risk", ge=1, le=9)
    chronic_lung_disease: int = Field(..., alias="chronic Lung Disease", ge=1, le=9)
    balanced_diet: int = Field(..., alias="Balanced Diet", ge=1, le=9)
    Obesity: int = Field(..., ge=1, le=9)
    Smoking: int = Field(..., ge=1, le=9)
    passive_smoker: int = Field(..., alias="Passive Smoker", ge=1, le=9)
    chest_pain: int = Field(..., alias="Chest Pain", ge=1, le=9)
    coughing_of_blood: int = Field(..., alias="Coughing of Blood", ge=1, le=9)
    Fatigue: int = Field(..., ge=1, le=9)
    weight_loss: int = Field(..., alias="Weight Loss", ge=1, le=9)
    shortness_of_breath: int = Field(..., alias="Shortness of Breath", ge=1, le=9)
    Wheezing: int = Field(..., ge=1, le=9)
    swallowing_difficulty: int = Field(..., alias="Swallowing Difficulty", ge=1, le=9)
    clubbing_of_finger_nails: int = Field(..., alias="Clubbing of Finger Nails", ge=1, le=9)
    frequent_cold: int = Field(..., alias="Frequent Cold", ge=1, le=9)
    dry_cough: int = Field(..., alias="Dry Cough", ge=1, le=9)
    Snoring: int = Field(..., ge=1, le=9)

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "Age": 35,
                "Gender": 1,
                "Air Pollution": 4,
                "Alcohol use": 5,
                "Dust Allergy": 6,
                "OccuPational Hazards": 5,
                "Genetic Risk": 5,
                "chronic Lung Disease": 4,
                "Balanced Diet": 6,
                "Obesity": 7,
                "Smoking": 2,
                "Passive Smoker": 3,
                "Chest Pain": 4,
                "Coughing of Blood": 8,
                "Fatigue": 8,
                "Weight Loss": 7,
                "Shortness of Breath": 9,
                "Wheezing": 2,
                "Swallowing Difficulty": 1,
                "Clubbing of Finger Nails": 4,
                "Frequent Cold": 6,
                "Dry Cough": 7,
                "Snoring": 2,
            }
        },
    }


class PredictionResponse(BaseModel):
    prediction: str
    probabilities: dict[str, float]
    model_name: str


def load_artifact() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run `python train.py` first to create models/cancer_risk_model.joblib."
        )
    return joblib.load(MODEL_PATH)


def load_metrics() -> dict:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {}


_artifact: dict | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _artifact
    _artifact = load_artifact()
    yield


app = FastAPI(
    title="Cancer Risk Prediction API",
    description=(
        "Predicts Low / Medium / High cancer risk from patient lifestyle, "
        "exposure, and symptom scores. This is a research model, not a medical diagnosis."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _artifact is not None}


@app.get("/features")
def features() -> dict:
    if _artifact is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return {
        "features": _artifact["feature_names"],
        "classes": _artifact["class_names"],
        "notes": {
            "Gender": "Use 1 or 2, as encoded in the original dataset.",
            "other_scores": "Most remaining fields are ordinal scores from the source data, typically 1-8 or 1-9.",
        },
    }


@app.get("/metrics")
def metrics() -> dict:
    return load_metrics()


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures) -> PredictionResponse:
    if _artifact is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    payload = patient.model_dump(by_alias=True)
    frame = pd.DataFrame([payload], columns=_artifact["feature_names"])
    pipeline = _artifact["pipeline"]
    encoder = _artifact["label_encoder"]

    class_index = int(pipeline.predict(frame)[0])
    label = str(encoder.inverse_transform([class_index])[0])
    proba = pipeline.predict_proba(frame)[0]
    probabilities = {
        str(cls): round(float(p), 4)
        for cls, p in zip(encoder.classes_, proba)
    }
    return PredictionResponse(
        prediction=label,
        probabilities=probabilities,
        model_name=_artifact["model_name"],
    )
