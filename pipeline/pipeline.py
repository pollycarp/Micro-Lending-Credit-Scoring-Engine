"""
Reproducible scikit-learn Pipelines for credit scoring.

Why pipelines?
--------------
A Pipeline chains preprocessing and modelling into a single object.
This means:
  1. No train/test leakage — the Scaler sees only training data.
  2. Inference is one call — pipeline.predict_proba(X_new).
  3. The whole thing serialises to a single .joblib file.
  4. MLflow can log the entire pipeline as one artifact.

Structure for each model
------------------------
  MedianImputer          — fills any NaN with training-set medians
       ↓
  OutlierClipper         — clips to [1st, 99th] percentile
       ↓
  StandardScaler         — zero-mean, unit-variance (tree models ignore this,
                           but it's needed for Logistic Regression)
       ↓
  Classifier             — LogisticRegression / XGBoost / LightGBM
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from pipeline.transformers import MedianImputer, OutlierClipper

SEED = 42


def build_pipeline(classifier) -> Pipeline:
    """
    Wrap any classifier in a standardised preprocessing pipeline.

    Parameters
    ----------
    classifier : scikit-learn compatible estimator

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    return Pipeline([
        ("imputer", MedianImputer()),
        ("clipper", OutlierClipper(lower_pct=1.0, upper_pct=99.0)),
        ("scaler",  StandardScaler()),
        ("clf",     classifier),
    ])


def get_all_pipelines() -> dict[str, Pipeline]:
    """
    Return a dict of {model_name: Pipeline} for all three models.
    These are the pipelines that will be logged to MLflow.
    """
    return {
        "LogisticRegression": build_pipeline(
            LogisticRegression(
                max_iter=1000,
                C=0.1,
                class_weight="balanced",
                random_state=SEED,
            )
        ),
        "XGBoost": build_pipeline(
            XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=2.5,
                eval_metric="logloss",
                random_state=SEED,
                verbosity=0,
            )
        ),
        "LightGBM": build_pipeline(
            LGBMClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight="balanced",
                random_state=SEED,
                verbose=-1,
                n_jobs=1,
            )
        ),
    }


# ── quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from features.feature_store import build_feature_matrix
    from models.train import temporal_split

    X, y = build_feature_matrix(verbose=False)
    X_train, X_test, y_train, y_test = temporal_split(X, y)

    pipelines = get_all_pipelines()
    pipe = pipelines["LogisticRegression"]
    pipe.fit(X_train, y_train)
    preds = pipe.predict_proba(X_test)[:, 1]
    print(f"Pipeline test — first 5 PD scores: {preds[:5].round(3)}")
    print("Pipeline works correctly.")
