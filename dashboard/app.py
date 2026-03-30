"""
Micro-Lending Credit Scoring Dashboard
---------------------------------------
Streamlit app with three pages:

  Score a Merchant     — live PD prediction + SHAP waterfall
  Portfolio Overview   — risk distribution across all merchants
  Model Performance    — ROC curve, PR curve, calibration

Run from the project root (venv activated):
  streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

# Make project root importable when running as `streamlit run dashboard/app.py`
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from features.feature_store import build_feature_matrix
from models.evaluate import get_roc_data, get_pr_data
from models.train import temporal_split

# ── page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "Credit Scoring Engine",
    page_icon  = "📊",
    layout     = "wide",
)

# ── shared resources (cached) ──────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    path = ROOT / "models" / "saved" / "best_model.joblib"
    bundle = joblib.load(path)
    return bundle["model"], bundle["name"]


@st.cache_data(show_spinner="Building feature matrix…")
def load_features():
    X, y = build_feature_matrix(verbose=False)
    return X, y


# ── sidebar navigation ─────────────────────────────────────────────────────────

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Score a Merchant", "Portfolio Overview", "Model Performance"],
)

pipeline, model_name = load_model()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Model:** {model_name}")
st.sidebar.markdown("**Experiment:** credit-scoring")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Score a Merchant
# ══════════════════════════════════════════════════════════════════════════════

if page == "Score a Merchant":
    st.title("Score a Merchant")
    st.markdown(
        "Enter a merchant ID to get a real-time probability-of-default (PD) "
        "prediction and a SHAP explanation of the top drivers."
    )

    col_input, _ = st.columns([1, 2])
    with col_input:
        merchant_id = st.text_input(
            "Merchant ID",
            value="M0001",
            placeholder="e.g. M0001 … M1000",
        )
        score_btn = st.button("Score", type="primary")

    if score_btn and merchant_id:
        with st.spinner("Fetching data and scoring…"):
            try:
                X_single, _ = build_feature_matrix(
                    merchant_ids=[merchant_id], verbose=False
                )
            except Exception as e:
                st.error(f"Failed to build features: {e}")
                st.stop()

            if X_single.empty:
                st.error(f"Merchant **{merchant_id}** not found.")
                st.stop()

            pd_score = float(pipeline.predict_proba(X_single)[0, 1])

        # ── risk tier ─────────────────────────────────────────────────────
        if pd_score < 0.25:
            tier, colour = "LOW RISK", "green"
        elif pd_score < 0.50:
            tier, colour = "MEDIUM RISK", "orange"
        else:
            tier, colour = "HIGH RISK", "red"

        # ── KPI row ───────────────────────────────────────────────────────
        k1, k2, k3 = st.columns(3)
        k1.metric("PD Score", f"{pd_score:.1%}")
        k2.metric("Risk Tier", tier)
        k3.metric("Merchant", merchant_id)

        # ── gauge chart ───────────────────────────────────────────────────
        fig_gauge = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = round(pd_score * 100, 1),
            number = {"suffix": "%"},
            title  = {"text": "Probability of Default"},
            gauge  = {
                "axis": {"range": [0, 100]},
                "bar":  {"color": colour},
                "steps": [
                    {"range": [0,  25], "color": "#d4edda"},
                    {"range": [25, 50], "color": "#fff3cd"},
                    {"range": [50, 100], "color": "#f8d7da"},
                ],
                "threshold": {
                    "line":  {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": pd_score * 100,
                },
            },
        ))
        fig_gauge.update_layout(height=300, margin=dict(t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── SHAP bar chart ────────────────────────────────────────────────
        st.subheader("Top Feature Contributions (SHAP)")

        clf      = pipeline.named_steps["clf"]
        clf_type = type(clf).__name__
        preprocessed = pipeline[:-1].transform(X_single)
        feature_names = list(pipeline.named_steps["clipper"].feature_names_in_)

        if clf_type in ("XGBClassifier", "LGBMClassifier"):
            import shap
            explainer = shap.TreeExplainer(clf)
            sv = explainer.shap_values(preprocessed)
            if isinstance(sv, list):
                sv = sv[1]
            shap_vals = sv[0]
        else:
            shap_vals = clf.coef_[0] * preprocessed[0]

        shap_df = pd.DataFrame({
            "feature": feature_names,
            "shap":    shap_vals,
        }).assign(abs_shap=lambda d: d["shap"].abs())
        shap_df = shap_df.nlargest(10, "abs_shap")
        shap_df["colour"] = shap_df["shap"].apply(
            lambda v: "Increases Risk" if v > 0 else "Decreases Risk"
        )
        shap_df = shap_df.sort_values("shap")

        fig_shap = px.bar(
            shap_df,
            x     = "shap",
            y     = "feature",
            color = "colour",
            color_discrete_map={
                "Increases Risk":  "#dc3545",
                "Decreases Risk":  "#28a745",
            },
            labels = {"shap": "SHAP Value", "feature": ""},
            title  = f"Top 10 SHAP contributions for {merchant_id}",
            orientation = "h",
        )
        fig_shap.update_layout(
            legend_title_text="",
            height=400,
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        # ── raw feature values ────────────────────────────────────────────
        with st.expander("Raw feature values"):
            st.dataframe(X_single.T.rename(columns={X_single.index[0]: "value"}))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Portfolio Overview
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Portfolio Overview":
    st.title("Portfolio Overview")
    st.markdown(
        "PD scores and risk distribution across all merchants in the dataset."
    )

    with st.spinner("Scoring all merchants…"):
        X, y = load_features()
        scores = pipeline.predict_proba(X)[:, 1]

    portfolio = pd.DataFrame({
        "merchant_id": X.index,
        "pd_score":    scores,
        "actual_default": y.values,
    })
    portfolio["risk_tier"] = pd.cut(
        portfolio["pd_score"],
        bins   = [0, 0.25, 0.50, 1.01],
        labels = ["Low", "Medium", "High"],
    )

    # ── KPI row ───────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Merchants", len(portfolio))
    k2.metric("Predicted High Risk",
              int((portfolio["risk_tier"] == "High").sum()))
    k3.metric("Actual Default Rate",
              f"{portfolio['actual_default'].mean():.1%}")
    k4.metric("Avg PD Score",
              f"{portfolio['pd_score'].mean():.1%}")

    col_left, col_right = st.columns(2)

    # ── risk tier pie ─────────────────────────────────────────────────────
    with col_left:
        tier_counts = portfolio["risk_tier"].value_counts().reset_index()
        tier_counts.columns = ["Risk Tier", "Count"]
        fig_pie = px.pie(
            tier_counts,
            names  = "Risk Tier",
            values = "Count",
            color  = "Risk Tier",
            color_discrete_map={
                "Low":    "#28a745",
                "Medium": "#ffc107",
                "High":   "#dc3545",
            },
            title = "Portfolio Risk Distribution",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── PD score histogram ────────────────────────────────────────────────
    with col_right:
        fig_hist = px.histogram(
            portfolio,
            x      = "pd_score",
            color  = "risk_tier",
            color_discrete_map={
                "Low":    "#28a745",
                "Medium": "#ffc107",
                "High":   "#dc3545",
            },
            nbins  = 40,
            labels = {"pd_score": "PD Score", "risk_tier": "Risk Tier"},
            title  = "Distribution of PD Scores",
        )
        fig_hist.add_vline(x=0.25, line_dash="dash", line_color="orange",
                           annotation_text="Medium threshold")
        fig_hist.add_vline(x=0.50, line_dash="dash", line_color="red",
                           annotation_text="High threshold")
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── PD score vs actual default (box plot) ─────────────────────────────
    fig_box = px.box(
        portfolio,
        x      = "actual_default",
        y      = "pd_score",
        color  = "actual_default",
        color_discrete_map={0: "#28a745", 1: "#dc3545"},
        labels = {
            "actual_default": "Actual Default (0=No, 1=Yes)",
            "pd_score":       "Predicted PD Score",
        },
        title  = "PD Score Distribution by Actual Default Status",
    )
    fig_box.update_layout(showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

    # ── full table ────────────────────────────────────────────────────────
    with st.expander("Full portfolio table"):
        display_df = portfolio.copy()
        display_df["pd_score"] = display_df["pd_score"].round(4)
        st.dataframe(display_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Model Performance":
    st.title("Model Performance")
    st.markdown(
        "Evaluation metrics and curves computed on the held-out test set "
        "(last 300 merchants — never seen during training)."
    )

    with st.spinner("Computing metrics…"):
        X, y = load_features()
        X_train, X_test, y_train, y_test = temporal_split(X, y)
        scores_test = pipeline.predict_proba(X_test)[:, 1]

        from sklearn.metrics import (
            roc_auc_score, average_precision_score,
            log_loss, brier_score_loss,
        )
        metrics = {
            "AUC-ROC":         roc_auc_score(y_test, scores_test),
            "Avg Precision":   average_precision_score(y_test, scores_test),
            "Log Loss":        log_loss(y_test, scores_test),
            "Brier Score":     brier_score_loss(y_test, scores_test),
        }

    # ── KPI row ───────────────────────────────────────────────────────────
    cols = st.columns(4)
    for col, (name, val) in zip(cols, metrics.items()):
        col.metric(name, f"{val:.4f}")

    col_left, col_right = st.columns(2)

    # ── ROC curve ─────────────────────────────────────────────────────────
    with col_left:
        roc_df = get_roc_data(pipeline, X_test, y_test)
        auc = metrics["AUC-ROC"]
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=roc_df["fpr"], y=roc_df["tpr"], mode="lines",
            name=f"{model_name} (AUC={auc:.4f})",
            line=dict(color="#007bff", width=2),
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Random classifier",
            line=dict(color="grey", dash="dash"),
        ))
        fig_roc.update_layout(
            title  = "ROC Curve",
            xaxis_title = "False Positive Rate",
            yaxis_title = "True Positive Rate",
            legend = dict(x=0.6, y=0.1),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # ── Precision-Recall curve ────────────────────────────────────────────
    with col_right:
        pr_df = get_pr_data(pipeline, X_test, y_test)
        ap = metrics["Avg Precision"]
        baseline = float(y_test.mean())
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(
            x=pr_df["recall"], y=pr_df["precision"], mode="lines",
            name=f"{model_name} (AP={ap:.4f})",
            line=dict(color="#28a745", width=2),
        ))
        fig_pr.add_hline(
            y=baseline, line_dash="dash", line_color="grey",
            annotation_text=f"Baseline (prevalence={baseline:.2f})",
        )
        fig_pr.update_layout(
            title  = "Precision-Recall Curve",
            xaxis_title = "Recall",
            yaxis_title = "Precision",
            legend = dict(x=0.5, y=0.9),
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    # ── calibration plot ──────────────────────────────────────────────────
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_test, scores_test, n_bins=10)

    fig_cal = go.Figure()
    fig_cal.add_trace(go.Scatter(
        x=prob_pred, y=prob_true, mode="lines+markers",
        name=model_name, line=dict(color="#fd7e14", width=2),
    ))
    fig_cal.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Perfect calibration",
        line=dict(color="grey", dash="dash"),
    ))
    fig_cal.update_layout(
        title  = "Calibration Curve (Reliability Diagram)",
        xaxis_title = "Mean Predicted Probability",
        yaxis_title = "Fraction of Positives",
        legend = dict(x=0.05, y=0.9),
    )
    st.plotly_chart(fig_cal, use_container_width=True)

    # ── feature importance ────────────────────────────────────────────────
    st.subheader("Global Feature Importance (SHAP)")

    import shap as shap_lib

    clf      = pipeline.named_steps["clf"]
    clf_type = type(clf).__name__
    X_sample = X_test.sample(min(200, len(X_test)), random_state=42)
    preprocessed_sample = pipeline[:-1].transform(X_sample)
    feature_names = list(pipeline.named_steps["clipper"].feature_names_in_)

    if clf_type in ("XGBClassifier", "LGBMClassifier"):
        explainer = shap_lib.TreeExplainer(clf)
        sv = explainer.shap_values(preprocessed_sample)
        if isinstance(sv, list):
            sv = sv[1]
        mean_abs_shap = np.abs(sv).mean(axis=0)
    else:
        mean_abs_shap = np.abs(clf.coef_[0]) * np.abs(preprocessed_sample).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": mean_abs_shap,
    }).sort_values("importance", ascending=True).tail(15)

    fig_imp = px.bar(
        importance_df,
        x           = "importance",
        y           = "feature",
        orientation = "h",
        labels      = {"importance": "Mean |SHAP|", "feature": ""},
        title       = "Top 15 Features by Mean |SHAP| (test set sample)",
        color       = "importance",
        color_continuous_scale = "Blues",
    )
    fig_imp.update_layout(coloraxis_showscale=False, height=500)
    st.plotly_chart(fig_imp, use_container_width=True)
