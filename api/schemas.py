"""
Pydantic schemas — define the shape of API requests and responses.

Pydantic validates every field automatically. If a client sends the wrong
type (e.g. a string where a float is expected) FastAPI returns a clear
422 error instead of crashing silently.
"""

from pydantic import BaseModel, ConfigDict, Field


class FeatureContribution(BaseModel):
    """One feature's contribution to a single prediction (from SHAP)."""
    feature:   str
    value:     float = Field(description="Actual feature value for this applicant")
    shap:      float = Field(description="SHAP value — positive = pushes toward default")
    direction: str   = Field(description="'increases_risk' or 'decreases_risk'")


class PredictionResponse(BaseModel):
    """Full prediction response returned by the scoring API."""
    model_config = ConfigDict(protected_namespaces=())

    merchant_id:      str
    pd_score:         float  = Field(ge=0.0, le=1.0,
                                     description="Probability of default (0–1)")
    risk_tier:        str    = Field(description="low | medium | high")
    top_features:     list[FeatureContribution]
    model_name:       str
    model_version:    str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status:       str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name:    str
    model_version: str
    experiment:    str
