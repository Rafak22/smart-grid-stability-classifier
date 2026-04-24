"""
Smart Grid Stability Classifier — FastAPI
POST /predict — takes 12 grid parameters, returns stable/unstable + confidence
"""

import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

# ── Load model ────────────────────────────────────────────────────────────────
with open("grid_stability_model.pkl", "rb") as f:
    model_package = pickle.load(f)

model       = model_package["model"]
feature_cols = model_package["feature_cols"]
class_names  = model_package["class_names"]

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Smart Grid Stability Classifier",
    description=(
        "Predicts whether a 4-node smart grid configuration is stable or unstable. "
        "Trained on the UCI Electrical Grid Stability Simulated Dataset. "
        "Aligned with Saudi Vision 2030 smart grid modernization."
    ),
    version="1.0.0",
)

# ── Input schema ──────────────────────────────────────────────────────────────
class GridInput(BaseModel):
    tau1: float = Field(..., ge=0.5, le=10.0, description="Producer reaction time (seconds)")
    tau2: float = Field(..., ge=0.5, le=10.0, description="Consumer 1 reaction time (seconds)")
    tau3: float = Field(..., ge=0.5, le=10.0, description="Consumer 2 reaction time (seconds)")
    tau4: float = Field(..., ge=0.5, le=10.0, description="Consumer 3 reaction time (seconds)")
    p1:   float = Field(..., ge=0.0,  le=6.0,  description="Producer power output")
    p2:   float = Field(..., ge=-2.0, le=0.0,  description="Consumer 1 power draw (negative)")
    p3:   float = Field(..., ge=-2.0, le=0.0,  description="Consumer 2 power draw (negative)")
    p4:   float = Field(..., ge=-2.0, le=0.0,  description="Consumer 3 power draw (negative)")
    g1:   float = Field(..., ge=0.05, le=1.0,  description="Producer price elasticity")
    g2:   float = Field(..., ge=0.05, le=1.0,  description="Consumer 1 price elasticity")
    g3:   float = Field(..., ge=0.05, le=1.0,  description="Consumer 2 price elasticity")
    g4:   float = Field(..., ge=0.05, le=1.0,  description="Consumer 3 price elasticity")

    class Config:
        json_schema_extra = {
            "example": {
                "tau1": 2.959, "tau2": 3.079, "tau3": 8.381, "tau4": 9.780,
                "p1": 3.763, "p2": -0.782, "p3": -1.257, "p4": -1.723,
                "g1": 0.650, "g2": 0.859, "g3": 0.887, "g4": 0.958
            }
        }

# ── Output schema ─────────────────────────────────────────────────────────────
class PredictionOutput(BaseModel):
    prediction:        Literal["stable", "unstable"]
    confidence:        float
    confidence_pct:    str
    risk_level:        Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    action:            str
    probabilities:     dict
    engineered_features: dict

# ── Feature engineering ───────────────────────────────────────────────────────
def engineer_features(data: GridInput) -> np.ndarray:
    tau = [data.tau1, data.tau2, data.tau3, data.tau4]
    g   = [data.g1,   data.g2,   data.g3,   data.g4]
    p   = [data.p1,   data.p2,   data.p3,   data.p4]

    total_reaction_time    = sum(tau)
    reaction_time_variance = float(np.var(tau))
    avg_consumer_tau       = (data.tau2 + data.tau3 + data.tau4) / 3
    producer_consumer_ratio = data.tau1 / avg_consumer_tau
    avg_price_elasticity   = sum(g) / 4
    net_power_balance      = sum(p)
    tau1_x_g1              = data.tau1 * (1 - data.g1)

    features = [
        data.tau1, data.tau2, data.tau3, data.tau4,
        data.p1,   data.p2,   data.p3,   data.p4,
        data.g1,   data.g2,   data.g3,   data.g4,
        total_reaction_time, reaction_time_variance, producer_consumer_ratio,
        avg_price_elasticity, net_power_balance, tau1_x_g1
    ]

    engineered = {
        "total_reaction_time":     round(total_reaction_time, 4),
        "reaction_time_variance":  round(reaction_time_variance, 4),
        "producer_consumer_ratio": round(producer_consumer_ratio, 4),
        "avg_price_elasticity":    round(avg_price_elasticity, 4),
        "net_power_balance":       round(net_power_balance, 6),
        "tau1_x_g1":               round(tau1_x_g1, 4),
    }

    return np.array(features).reshape(1, -1), engineered

# ── Decision logic ────────────────────────────────────────────────────────────
def get_risk_and_action(prediction: str, confidence: float) -> tuple:
    if prediction == "unstable":
        if confidence >= 0.80:
            return "CRITICAL", "Immediate automated protective action — load shedding or rerouting required"
        else:
            return "HIGH", "Warning — monitor grid closely and prepare protective measures"
    else:
        if confidence >= 0.75:
            return "LOW", "Normal operation — no intervention required"
        else:
            return "MEDIUM", "Grid appears stable but confidence is low — increase monitoring frequency"

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "Smart Grid Stability Classifier API",
        "version": "1.0.0",
        "model": model_package["model_name"],
        "metrics": model_package["metrics"],
        "endpoints": {
            "POST /predict": "Predict grid stability from 12 parameters",
            "GET /health":   "Check API health",
            "GET /docs":     "Interactive API documentation"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_name": model_package["model_name"],
        "features_expected": len(feature_cols)
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(data: GridInput):
    try:
        features, engineered = engineer_features(data)
        pred_index  = model.predict(features)[0]
        pred_proba  = model.predict_proba(features)[0]
        prediction  = class_names[pred_index]
        confidence  = float(pred_proba.max())
        risk, action = get_risk_and_action(prediction, confidence)

        return PredictionOutput(
            prediction=prediction,
            confidence=round(confidence, 4),
            confidence_pct=f"{confidence * 100:.1f}%",
            risk_level=risk,
            action=action,
            probabilities={
                class_names[0]: round(float(pred_proba[0]), 4),
                class_names[1]: round(float(pred_proba[1]), 4),
            },
            engineered_features=engineered
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
