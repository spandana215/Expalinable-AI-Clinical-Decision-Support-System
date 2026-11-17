"""cdss_xai.py - SHAP + LIME helpers shared by the notebook and the Streamlit app."""
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cdss_utils import decode_features, transform_features

TREE_MODELS = ("Random Forest", "Decision Tree", "XGBoost")
CLASS_NAMES = ["Not readmitted within 30 days", "Readmitted within 30 days"]


class ShapEngine:
    """SHAP values for class 1 (readmitted <30 days) as one (n_rows, n_features) array,
    whatever the model type or the installed SHAP version returns."""

    def __init__(self, model_raw, model_name, background_scaled=None):
        import shap
        self.model, self.model_name = model_raw, model_name
        self.explainer, self.backend = None, "shap"
        if model_name in TREE_MODELS:
            # Random Forest / Decision Tree explain a probability; XGBoost explains log-odds
            self.units = "log-odds" if model_name == "XGBoost" else "probability"
            try:
                self.explainer = shap.TreeExplainer(model_raw)
            except Exception:
                if model_name != "XGBoost":
                    raise
                self.backend = "xgboost-native"   # exact TreeSHAP built into XGBoost
        else:
            self.units = "log-odds"
            self.explainer = shap.LinearExplainer(model_raw, np.asarray(background_scaled, dtype=float))

    def _native(self, X):
        import xgboost as xgb
        contribs = self.model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
        return np.asarray(contribs[:, :-1], dtype=float), float(contribs[0, -1])

    def values(self, X_scaled):
        X = np.asarray(X_scaled, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.backend == "xgboost-native":
            return self._native(X)
        try:
            if self.model_name in TREE_MODELS:
                out = self.explainer.shap_values(X, check_additivity=False)
            else:
                out = self.explainer.shap_values(X)
        except Exception:
            if self.model_name != "XGBoost":
                raise
            self.backend = "xgboost-native"
            return self._native(X)
        if isinstance(out, list):                 # old SHAP: [class0, class1]
            sv = np.asarray(out[-1], dtype=float)
        else:
            out = np.asarray(out, dtype=float)
            sv = out[:, :, -1] if out.ndim == 3 else out   # new SHAP: (rows, features, classes)
        base = np.ravel(np.asarray(self.explainer.expected_value, dtype=float))
        return sv, float(base[-1])


def build_lime_explainer(background_enc, state, random_state=42):
    from lime.lime_tabular import LimeTabularExplainer
    feats = state["feature_columns"]
    cat_idx = [feats.index(c) for c in state["categorical_cols"]]
    cat_names = {feats.index(c): [str(v) for v in state["label_encoders"][c].classes_]
                 for c in state["categorical_cols"]}
    return LimeTabularExplainer(
        training_data=np.asarray(background_enc, dtype=float),
        feature_names=feats, class_names=CLASS_NAMES,
        categorical_features=cat_idx, categorical_names=cat_names,
        mode="classification", discretize_continuous=True, random_state=random_state)


def lime_explain(explainer, row_enc, predict_fn, feature_columns, num_features=10):
    exp = explainer.explain_instance(np.asarray(row_enc, dtype=float), predict_fn,
                                     num_features=num_features, labels=(1,))
    pairs = exp.as_map()[1]                      # [(feature index, weight)] for class 1
    rules = [d for d, _ in exp.as_list(label=1)]  # readable rule, same order
    rows = [{"feature": feature_columns[int(i)], "lime_rule": r, "lime_weight": float(w)}
            for (i, w), r in zip(pairs, rules)]
    return exp, pd.DataFrame(rows)


def direction(v):
    return "increases risk" if v > 0 else ("decreases risk" if v < 0 else "no effect")


def compare_shap_lime(shap_df, lime_df, top_k=10):
    """Rank/direction agreement between the two methods (raw values are NOT comparable)."""
    s = shap_df.head(top_k).reset_index(drop=True)
    l = lime_df.reindex(lime_df["lime_weight"].abs().sort_values(ascending=False).index)
    l = l.head(top_k).reset_index(drop=True)
    s_rank = {f: i + 1 for i, f in enumerate(s["feature"])}
    l_rank = {f: i + 1 for i, f in enumerate(l["feature"])}
    s_val = dict(zip(s["feature"], s["shap_value"]))
    l_val = dict(zip(l["feature"], l["lime_weight"]))
    union = list(s["feature"]) + [f for f in l["feature"] if f not in s_rank]
    common = [f for f in s["feature"] if f in l_rank]
    table = pd.DataFrame({
        "feature": union,
        "shap_rank": [s_rank.get(f) for f in union],
        "lime_rank": [l_rank.get(f) for f in union],
        "shap_direction": [direction(s_val[f]) if f in s_val else "outside top %d" % top_k for f in union],
        "lime_direction": [direction(l_val[f]) if f in l_val else "outside top %d" % top_k for f in union],
    })
    agree = [f for f in common if np.sign(s_val[f]) == np.sign(l_val[f])]
    summary = {
        "top_k": top_k,
        "common_features": common,
        "jaccard": round(len(common) / len(union), 3) if union else 0.0,
        "sign_agreement": round(len(agree) / len(common), 3) if common else None,
        "same_direction_features": agree,
    }
    return table, summary


def load_bundle(artifacts_dir="artifacts"):
    """Load every saved artifact and rebuild the SHAP / LIME explainers."""
    with open(os.path.join(artifacts_dir, "config.json")) as f:
        config = json.load(f)
    b = {
        "config": config,
        "model": joblib.load(os.path.join(artifacts_dir, "model.joblib")),          # calibrated
        "model_raw": joblib.load(os.path.join(artifacts_dir, "model_raw.joblib")),  # tuned, uncalibrated
        "scaler": joblib.load(os.path.join(artifacts_dir, "scaler.joblib")),
        "state": joblib.load(os.path.join(artifacts_dir, "preprocessor.joblib")),
        "background": joblib.load(os.path.join(artifacts_dir, "lime_background.joblib")),
        "threshold": float(config["decision_threshold"]),
        "model_name": config["best_model_name"],
    }
    b["feature_columns"] = b["state"]["feature_columns"]
    b["predict_fn"] = lambda d: b["model"].predict_proba(np.asarray(d, dtype=float))
    b["shap_engine"] = ShapEngine(b["model_raw"], b["model_name"],
                                  b["scaler"].transform(b["background"]))
    b["lime_explainer"] = build_lime_explainer(b["background"], b["state"])
    return b


def analyze_encoded(row_enc, bundle, top_k=10):
    """Prediction + SHAP + LIME + comparison for one already-encoded (unscaled) row."""
    feats = bundle["feature_columns"]
    x_enc = np.asarray(row_enc, dtype=float).reshape(1, -1)
    x_scaled = bundle["scaler"].transform(x_enc)
    proba = bundle["model"].predict_proba(x_enc)[0]          # calibrated model = the prediction
    p1 = float(proba[1])
    pred = int(p1 >= bundle["threshold"])
    sv, base = bundle["shap_engine"].values(x_scaled)        # explains the tuned uncalibrated model
    values = decode_features(x_enc[0], bundle["state"])
    shap_df = pd.DataFrame({"feature": feats,
                            "patient_value": [values[f] for f in feats],
                            "shap_value": sv[0]})
    shap_df = shap_df.reindex(shap_df["shap_value"].abs().sort_values(ascending=False).index)
    shap_df = shap_df.reset_index(drop=True)
    shap_df["direction"] = shap_df["shap_value"].apply(direction)
    lime_exp, lime_df = lime_explain(bundle["lime_explainer"], x_enc[0], bundle["predict_fn"],
                                     feats, num_features=top_k)
    lime_df["patient_value"] = [values[f] for f in lime_df["feature"]]
    lime_df["direction"] = lime_df["lime_weight"].apply(direction)
    compare_df, agreement = compare_shap_lime(shap_df, lime_df, top_k)
    return {
        "predicted_class": pred, "predicted_label": CLASS_NAMES[pred],
        "proba_not_readmitted": float(proba[0]), "proba_readmitted": p1,
        "threshold": bundle["threshold"], "model_name": bundle["model_name"],
        "x_enc": x_enc[0], "x_scaled": x_scaled[0], "values": values,
        "shap_values": sv[0], "shap_base": base, "shap_units": bundle["shap_engine"].units,
        "shap_df": shap_df, "lime_df": lime_df, "lime_exp": lime_exp,
        "compare_df": compare_df, "agreement": agreement, "top_k": top_k,
    }


def analyze_patient(patient, bundle, top_k=10):
    """Raw patient record (dict) -> full analysis, using the saved preprocessing."""
    X_enc, unseen = transform_features(pd.DataFrame([patient]), bundle["state"], return_unseen=True)
    res = analyze_encoded(X_enc.values[0], bundle, top_k)
    res["unseen_categories"] = unseen
    return res


def plot_contributions(df, value_col, title, ax=None, top_k=10, xlabel=None):
    """Horizontal bar chart: red pushes towards readmission, blue pushes away from it."""
    d = df.reindex(df[value_col].abs().sort_values(ascending=False).index).head(top_k).iloc[::-1]
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(7, 0.42 * len(d) + 1.6))
    else:
        fig = ax.figure
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in d[value_col]]
    ax.barh(d["feature"].astype(str), d[value_col], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel or "contribution (red = increases risk, blue = decreases risk)")
    if created:
        fig.tight_layout()
    return fig
