"""
Model training for the Micro-Lending Credit Scoring Engine.

Three models are trained and compared:
  1. Logistic Regression  — interpretable baseline
  2. XGBoost              — gradient boosted trees
  3. LightGBM             — fast gradient boosting

Split strategy
--------------
We use a temporal-style split: the first 700 merchants are used for
training (representing historical applicants) and the last 300 are
held out as the test set (representing future applicants the model
has never seen).  This is the correct methodology for credit models —
never shuffle and split randomly, as that leaks future information.

Outputs
-------
Trained model objects returned by train_all_models().
Best model saved to models/saved/best_model.joblib
"""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from features.feature_store import build_feature_matrix
from models.evaluate import evaluate_model, print_report

SAVED_DIR = Path(__file__).parent / "saved"
SAVED_DIR.mkdir(exist_ok=True)

# ── reproducibility ────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)


# ── train / test split ─────────────────────────────────────────────────────────

def temporal_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float = 0.70,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Simulate a temporal split by preserving merchant ordering.

    In a real system you would split on loan origination date.
    Here merchant IDs are already in generation order (M0001 = oldest),
    so splitting on position is equivalent.
    """
    n_train = int(len(X) * train_size)
    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]

    print(f"Train set : {len(X_train):>4} merchants  "
          f"(default rate {y_train.mean():.1%})")
    print(f"Test  set : {len(X_test):>4} merchants  "
          f"(default rate {y_test.mean():.1%})")
    return X_train, X_test, y_train, y_test


# ── model definitions ──────────────────────────────────────────────────────────

def _models() -> dict:
    """
    Return a dict of {name: estimator}.

    Logistic Regression is wrapped in a Pipeline with StandardScaler
    because it is sensitive to feature scale.
    XGBoost and LightGBM are tree-based and scale-invariant.
    """
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                C=0.1,
                class_weight="balanced",
                random_state=SEED,
            )),
        ]),

        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(679 / 321),   # handles class imbalance
            eval_metric="logloss",
            random_state=SEED,
            verbosity=0,
        ),

        "LightGBM": LGBMClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=SEED,
            verbose=-1,
            n_jobs=1,   # avoid Windows multiprocessing issues
        ),
    }


# ── cross-validation ───────────────────────────────────────────────────────────

def cross_validate_models(
    models: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
) -> pd.DataFrame:
    """
    Run stratified k-fold CV on the training set and report AUC-ROC.

    StratifiedKFold preserves the class ratio in each fold, which is
    important with our 68/32 imbalanced split.
    """
    print(f"\n── Cross-validation ({cv_folds}-fold Stratified) ────────────────")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=False)   # no shuffle = temporal order

    results = {}
    for name, model in models.items():
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv, scoring="roc_auc", n_jobs=1,  # n_jobs=-1 can deadlock on Windows
        )
        results[name] = {
            "cv_auc_mean": scores.mean(),
            "cv_auc_std":  scores.std(),
        }
        print(f"  {name:<22}  AUC = {scores.mean():.4f} ± {scores.std():.4f}")

    return pd.DataFrame(results).T


# ── training ───────────────────────────────────────────────────────────────────

def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models: dict,
) -> dict:
    """Fit every model on the full training set."""
    print("\n── Training on full training set ────────────────────────────")
    trained = {}
    for name, model in models.items():
        print(f"  Fitting {name} …", end=" ", flush=True)
        model.fit(X_train, y_train)
        trained[name] = model
        print("done.")
    return trained


# ── model selection + persistence ─────────────────────────────────────────────

def select_and_save_best(
    trained_models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[str, object]:
    """
    Evaluate all models on the held-out test set, print a comparison
    table, save the best model to disk, and return (name, model).
    """
    print("\n── Test-set evaluation ──────────────────────────────────────")
    rows = []
    for name, model in trained_models.items():
        metrics = evaluate_model(model, X_test, y_test, model_name=name)
        rows.append({"model": name, **metrics})

    results_df = pd.DataFrame(rows).set_index("model")
    print("\n── Summary table ────────────────────────────────────────────")
    print(results_df[["roc_auc", "avg_precision", "log_loss", "brier_score"]].to_string())

    best_name = results_df["roc_auc"].idxmax()
    best_model = trained_models[best_name]

    path = SAVED_DIR / "best_model.joblib"
    joblib.dump({"name": best_name, "model": best_model}, path)
    print(f"\n  Best model : {best_name}  (AUC = {results_df.loc[best_name, 'roc_auc']:.4f})")
    print(f"  Saved to   : {path}")

    return best_name, best_model


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading feature matrix …")
    X, y = build_feature_matrix(verbose=False)

    print("\nSplitting data …")
    X_train, X_test, y_train, y_test = temporal_split(X, y)

    models = _models()

    cv_results = cross_validate_models(models, X_train, y_train)

    trained_models = train_all_models(X_train, y_train, models)

    best_name, best_model = select_and_save_best(trained_models, X_test, y_test)

    print("\nPrinting full report for best model …")
    print_report(best_model, X_test, y_test, model_name=best_name)
