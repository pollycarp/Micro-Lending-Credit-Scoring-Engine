"""
Model explainability using SHAP (SHapley Additive exPlanations).

Why SHAP matters in credit scoring
-----------------------------------
Regulators (and applicants) can ask "why was this loan denied?".
SHAP answers that by assigning each feature a contribution score
for every individual prediction — not just overall importance.

Functions
---------
get_shap_values(model, X)      → raw SHAP values matrix
plot_summary(model, X)         → beeswarm plot (global feature importance)
plot_waterfall(model, X, idx)  → waterfall for one applicant
plot_bar(model, X)             → mean absolute SHAP bar chart
"""

import warnings
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")


def _get_explainer(model, X: pd.DataFrame):
    """
    Return the right SHAP explainer for the given model type.

    - XGBoost / LightGBM → TreeExplainer (fast, exact)
    - Logistic Regression (inside a Pipeline) → LinearExplainer
    """
    # Unwrap sklearn Pipeline to get the underlying estimator
    estimator = model
    X_input   = X

    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("clf", list(model.named_steps.values())[-1])
        # Apply all transformers before the final estimator
        for step_name, step in list(model.named_steps.items())[:-1]:
            X_input = step.transform(X_input)
        X_input = pd.DataFrame(X_input, columns=X.columns, index=X.index)

    model_type = type(estimator).__name__
    if model_type in ("XGBClassifier", "LGBMClassifier"):
        explainer = shap.TreeExplainer(estimator)
    else:
        explainer = shap.LinearExplainer(estimator, X_input)

    return explainer, X_input


def get_shap_values(model, X: pd.DataFrame) -> np.ndarray:
    """
    Compute SHAP values for all rows in X.

    Returns a 2-D array of shape (n_samples, n_features).
    Positive values push the prediction toward default (class 1).
    """
    explainer, X_input = _get_explainer(model, X)
    sv = explainer.shap_values(X_input)

    # LightGBM returns a list [class0_values, class1_values]
    if isinstance(sv, list):
        sv = sv[1]
    return sv


def plot_summary(model, X: pd.DataFrame, max_display: int = 15) -> None:
    """
    Beeswarm plot — shows how each feature affects predictions across
    all applicants.  High (red) feature values that push right = higher PD.
    """
    explainer, X_input = _get_explainer(model, X)
    sv = explainer.shap_values(X_input)
    if isinstance(sv, list):
        sv = sv[1]

    print(f"Generating SHAP summary plot for {X.shape[0]} applicants …")
    shap.summary_plot(sv, X_input, max_display=max_display, show=True)


def plot_bar(model, X: pd.DataFrame, max_display: int = 15) -> None:
    """
    Bar chart of mean absolute SHAP values — classic global feature importance.
    """
    sv = get_shap_values(model, X)
    mean_abs = pd.Series(
        np.abs(sv).mean(axis=0),
        index=X.columns,
        name="mean_abs_shap",
    ).sort_values(ascending=False)

    print("\nTop features by mean |SHAP| value:")
    print(mean_abs.head(max_display).round(4).to_string())

    shap.summary_plot(sv, X, plot_type="bar", max_display=max_display, show=True)


def plot_waterfall(model, X: pd.DataFrame, idx: int = 0) -> None:
    """
    Waterfall plot for a single applicant — shows exactly which features
    pushed the PD score up or down from the baseline.

    Parameters
    ----------
    idx : integer position in X (0 = first row)
    """
    explainer, X_input = _get_explainer(model, X)

    merchant_id = X.index[idx]
    print(f"\nWaterfall explanation for merchant {merchant_id}")
    print(f"  Actual PD prob : "
          f"{model.predict_proba(X.iloc[[idx]])[:, 1][0]:.3f}")

    sv = explainer(X_input)
    if hasattr(sv, "values") and len(sv.values.shape) == 3:
        # LightGBM multi-output — take class 1
        import copy
        sv_single = copy.copy(sv[idx])
        sv_single.values       = sv[idx].values[:, 1]
        sv_single.base_values  = sv[idx].base_values[1]
    else:
        sv_single = sv[idx]

    shap.waterfall_plot(sv_single, show=True)


def feature_importance_df(model, X: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame of features ranked by mean absolute SHAP value.
    Useful for reports and the README.
    """
    sv = get_shap_values(model, X)
    return (
        pd.Series(np.abs(sv).mean(axis=0), index=X.columns, name="mean_abs_shap")
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "feature"})
    )


# ── quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import joblib
    from pathlib import Path
    from features.feature_store import build_feature_matrix
    from models.train import temporal_split

    print("Loading data and model …")
    X, y = build_feature_matrix(verbose=False)
    _, X_test, _, y_test = temporal_split(X, y)

    model_path = Path(__file__).parent / "saved" / "best_model.joblib"
    if not model_path.exists():
        print("No saved model found — run python -m models.train first.")
        raise SystemExit(1)

    bundle    = joblib.load(model_path)
    model_name = bundle["name"]
    model      = bundle["model"]
    print(f"Loaded model: {model_name}")

    # Top features
    imp = feature_importance_df(model, X_test)
    print("\nTop 10 features by mean |SHAP|:")
    print(imp.head(10).to_string(index=False))

    # Plots (opens matplotlib windows)
    plot_bar(model, X_test)
    plot_waterfall(model, X_test, idx=0)
