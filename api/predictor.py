"""
Predictor — loads the trained pipeline and serves predictions.

Model loading strategy
----------------------
1. Try to load the @champion model from the MLflow Model Registry.
2. If MLflow is unavailable, fall back to models/saved/best_model.joblib.

Inference flow for a given merchant_id
---------------------------------------
  MongoDB  ──┐
  PostgreSQL ─┤→ build_feature_matrix() → Pipeline.predict_proba() → PD score
  Bureau API ─┘                                                      → SHAP top-5
"""

import os
import warnings
from pathlib import Path

import joblib
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from dotenv import load_dotenv

from features.feature_store import build_feature_matrix
from api.schemas import FeatureContribution, PredictionResponse

load_dotenv()
warnings.filterwarnings("ignore")

MLFLOW_URI    = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
SAVED_MODEL   = Path(__file__).parent.parent / "models" / "saved" / "best_model.joblib"
TOP_N_FEATURES = 5


def _risk_tier(pd_score: float) -> str:
    if pd_score < 0.25:  return "low"
    if pd_score < 0.50:  return "medium"
    return "high"


class Predictor:
    """
    Singleton-style predictor loaded once at API startup.

    Attributes
    ----------
    pipeline     : fitted sklearn Pipeline (preprocessing + classifier)
    model_name   : human-readable name (e.g. "LogisticRegression")
    model_version: MLflow version string or "local"
    """

    def __init__(self):
        self.pipeline      = None
        self.model_name    = None
        self.model_version = None
        self._load_model()

    # ── model loading ──────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Try MLflow registry first, fall back to joblib file."""
        loaded = False

        # ── attempt 1: MLflow Model Registry ──────────────────────────────
        try:
            mlflow.set_tracking_uri(MLFLOW_URI)
            client = mlflow.tracking.MlflowClient()

            # Resolve @champion alias to a concrete version
            mv = client.get_model_version_by_alias("CreditScoringModel", "champion")
            model_uri = f"models:/CreditScoringModel@champion"

            self.pipeline      = mlflow.sklearn.load_model(model_uri)
            self.model_version = f"v{mv.version} (@champion)"
            # Derive model name directly from the loaded classifier class
            self.model_name    = type(self.pipeline.named_steps["clf"]).__name__
            loaded = True
            print(f"[Predictor] Loaded from MLflow registry: "
                  f"CreditScoringModel {self.model_version}")
        except Exception as e:
            print(f"[Predictor] MLflow load failed ({e}), trying local fallback …")

        # ── attempt 2: local joblib file ───────────────────────────────────
        if not loaded:
            if not SAVED_MODEL.exists():
                raise RuntimeError(
                    "No model found. Run 'python -m pipeline.mlflow_train' first."
                )
            bundle             = joblib.load(SAVED_MODEL)
            self.pipeline      = bundle["model"]
            self.model_name    = bundle["name"]
            self.model_version = "local"
            print(f"[Predictor] Loaded from local file: {self.model_name}")

    # ── SHAP explanations ──────────────────────────────────────────────────────

    def _top_shap_features(
        self,
        X_single: pd.DataFrame,
    ) -> list[FeatureContribution]:
        """
        Compute SHAP values for one applicant and return top-N features.

        The pipeline transforms X before reaching the classifier, so we
        extract the classifier and apply the preprocessing steps manually
        to get correctly scaled input for SHAP.
        """
        try:
            clf      = self.pipeline.named_steps["clf"]
            clf_type = type(clf).__name__

            # Apply all preprocessing steps and keep as numpy array —
            # avoids any DataFrame column-count mismatch during transform
            preprocessed = self.pipeline[:-1].transform(X_single)  # (1, n_features)

            # Get the authoritative feature names from the fitted clipper
            # (these match what the classifier was actually trained on)
            feature_names = list(
                self.pipeline.named_steps["clipper"].feature_names_in_
            )

            if clf_type in ("XGBClassifier", "LGBMClassifier"):
                explainer = shap.TreeExplainer(clf)
                sv        = explainer.shap_values(preprocessed)
                if isinstance(sv, list):
                    sv = sv[1]   # class-1 (default) SHAP values
                shap_vals = sv[0]
            else:
                # LogisticRegression: SHAP = coef × scaled_feature_value (exact)
                shap_vals = clf.coef_[0] * preprocessed[0]

            # Map original (unscaled) feature values for display
            orig_vals = dict(zip(X_single.columns, X_single.iloc[0].values))

            contributions = []
            for feat, sv_val in zip(feature_names, shap_vals):
                contributions.append(
                    FeatureContribution(
                        feature   = feat,
                        value     = round(float(orig_vals.get(feat, 0.0)), 4),
                        shap      = round(float(sv_val), 4),
                        direction = "increases_risk" if sv_val > 0 else "decreases_risk",
                    )
                )

            contributions.sort(key=lambda c: abs(c.shap), reverse=True)
            return contributions[:TOP_N_FEATURES]

        except Exception as e:
            print(f"[Predictor] SHAP failed: {type(e).__name__}: {e}")
            return []

    # ── public predict method ──────────────────────────────────────────────────

    def predict(self, merchant_id: str) -> PredictionResponse:
        """
        Run end-to-end prediction for a single merchant.

        Fetches data from all three sources, builds features,
        scores the applicant, and returns SHAP explanations.
        """
        # Build feature matrix for this one merchant
        X, _ = build_feature_matrix(
            merchant_ids=[merchant_id],
            verbose=False,
        )

        if X.empty:
            raise ValueError(f"Merchant '{merchant_id}' not found in any data source.")

        # PD score
        pd_score = float(self.pipeline.predict_proba(X)[0, 1])

        # SHAP explanations
        top_features = self._top_shap_features(X)

        return PredictionResponse(
            merchant_id   = merchant_id,
            pd_score      = round(pd_score, 4),
            risk_tier     = _risk_tier(pd_score),
            top_features  = top_features,
            model_name    = self.model_name,
            model_version = self.model_version,
        )

    def model_info(self) -> dict:
        return {
            "model_name":    self.model_name,
            "model_version": self.model_version,
            "experiment":    "credit-scoring",
        }
