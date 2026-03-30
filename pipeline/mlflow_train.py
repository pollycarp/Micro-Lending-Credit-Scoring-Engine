"""
MLflow experiment tracking for the credit scoring pipeline.

What gets logged per run
------------------------
  Parameters  : all hyperparameters of the classifier step
  Metrics     : roc_auc, avg_precision, log_loss, brier_score (test set)
  Artifact    : the full serialised Pipeline (ready for inference)
  Tags        : model name, split info

After all runs, the best model is registered in the MLflow Model Registry
under the name "CreditScoringModel" with alias "champion".

Usage
-----
  python -m pipeline.mlflow_train

Then open the MLflow UI to explore all runs:
  mlflow ui
  → http://localhost:5000
"""

import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from features.feature_store import build_feature_matrix
from models.train import temporal_split
from models.evaluate import evaluate_model
from pipeline.pipeline import get_all_pipelines

# ── MLflow setup ───────────────────────────────────────────────────────────────
EXPERIMENT_NAME = "credit-scoring"
MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


# ── helpers ────────────────────────────────────────────────────────────────────

def _get_clf_params(pipeline) -> dict:
    """Extract only the classifier's hyperparameters from the pipeline."""
    clf = pipeline.named_steps["clf"]
    params = clf.get_params()
    # Prefix with clf__ so it's clear these are model params, not pipeline params
    return {f"clf__{k}": v for k, v in params.items()
            if not callable(v) and v is not None}


def run_experiment(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    y_train: pd.Series,
    y_test:  pd.Series,
) -> str:
    """
    Train all pipelines, log every run to MLflow, register the best model.

    Returns
    -------
    str — name of the best model
    """
    pipelines = get_all_pipelines()
    cv        = StratifiedKFold(n_splits=5, shuffle=False)

    best_run_id  = None
    best_auc     = 0.0
    best_name    = None
    results      = []

    for model_name, pipeline in pipelines.items():
        print(f"\n── {model_name} ──────────────────────────────────────────────")

        with mlflow.start_run(run_name=model_name) as run:

            # ── cross-validation on train set ──────────────────────────
            print("  Cross-validating …", end=" ", flush=True)
            cv_scores = cross_val_score(
                pipeline, X_train, y_train,
                cv=cv, scoring="roc_auc", n_jobs=1,
            )
            print(f"AUC = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

            # ── fit on full train set ──────────────────────────────────
            print("  Fitting …", end=" ", flush=True)
            pipeline.fit(X_train, y_train)
            print("done.")

            # ── evaluate on test set ───────────────────────────────────
            metrics = evaluate_model(pipeline, X_test, y_test)

            # ── log to MLflow ──────────────────────────────────────────
            mlflow.log_params(_get_clf_params(pipeline))
            mlflow.log_params({
                "model_name":   model_name,
                "train_size":   len(X_train),
                "test_size":    len(X_test),
                "n_features":   X_train.shape[1],
                "cv_folds":     5,
            })
            mlflow.log_metrics({
                **metrics,
                "cv_auc_mean": round(cv_scores.mean(), 4),
                "cv_auc_std":  round(cv_scores.std(),  4),
            })
            mlflow.set_tags({
                "model_family": model_name,
                "split_type":   "temporal",
            })

            # Log the full pipeline (preprocessing + model)
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="pipeline",
                input_example=X_test.iloc[:3],
            )

            test_auc = metrics["roc_auc"]
            print(f"  Test AUC = {test_auc:.4f}  |  run_id = {run.info.run_id[:8]}…")

            results.append({
                "model":      model_name,
                "run_id":     run.info.run_id,
                **metrics,
                "cv_auc":     round(cv_scores.mean(), 4),
            })

            if test_auc > best_auc:
                best_auc    = test_auc
                best_run_id = run.info.run_id
                best_name   = model_name

    # ── summary table ──────────────────────────────────────────────────────────
    df = pd.DataFrame(results).set_index("model")
    print(f"\n{'═' * 60}")
    print("  Results Summary")
    print(f"{'═' * 60}")
    print(df[["cv_auc", "roc_auc", "avg_precision",
              "log_loss", "brier_score"]].to_string())

    # ── register best model in MLflow Model Registry ───────────────────────────
    model_uri = f"runs:/{best_run_id}/pipeline"
    reg = mlflow.register_model(model_uri, "CreditScoringModel")

    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(
        name    = "CreditScoringModel",
        alias   = "champion",
        version = reg.version,
    )

    # Also save to local joblib so the Docker fallback path uses the full pipeline
    saved_dir = Path(__file__).parent.parent / "models" / "saved"
    saved_dir.mkdir(exist_ok=True)
    joblib_path = saved_dir / "best_model.joblib"
    best_pipeline = pipelines[best_name]   # already fitted above
    joblib.dump({"name": best_name, "model": best_pipeline}, joblib_path)

    print(f"\n  Best model   : {best_name}  (AUC = {best_auc:.4f})")
    print(f"  Registered as: CreditScoringModel v{reg.version} @champion")
    print(f"  Saved to     : {joblib_path}")
    print(f"\n  View all runs:  mlflow ui  →  http://localhost:5000")

    return best_name


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading feature matrix …")
    X, y = build_feature_matrix(verbose=False)

    print("Splitting data …")
    X_train, X_test, y_train, y_test = temporal_split(X, y)

    run_experiment(X_train, X_test, y_train, y_test)
