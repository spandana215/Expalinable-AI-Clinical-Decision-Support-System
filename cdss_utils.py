"""cdss_utils.py - shared preprocessing for the Diabetes Readmission CDSS.

The SAME functions are used for training, testing, SHAP, LIME, new-patient
inference and the Streamlit app, so preprocessing can never drift apart.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

TARGET = "readmitted"
DROP_COLS = ["encounter_id", "patient_nbr", "weight", "payer_code", "medical_specialty"]
DIAG_COLS = ["diag_1", "diag_2", "diag_3"]
ENGINEERED_COLS = ["previous_admissions", "medication_load", "time_since_last_visit",
                   "diag_1_num", "diag_2_num", "diag_3_num", "diag_severity"]
MISSING_TOKENS = ["?", ""]   # NOTE: the text "None" is a real category (test not done), not missing


def safe_float(x):
    """Convert an ICD-9 code to a number. V/E codes, blanks and bad input -> 0.0."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return v if np.isfinite(v) else 0.0


def severity_score(code):
    """Rule-based severity of one ICD-9 diagnosis code."""
    if 390 <= code < 460:
        return 3          # circulatory diseases (390-459)
    if 460 <= code < 520:
        return 2          # respiratory diseases (460-519)
    if 520 <= code < 580:
        return 2          # digestive diseases (520-579)
    if 250 <= code < 251:
        return 3          # diabetes (250.xx)
    if 800 <= code < 1000:
        return 1          # injuries (800-999)
    return 0


def add_engineered_features(df):
    """Row-wise feature engineering (no statistics are learned, so no leakage)."""
    out = df.copy()
    for c in ["number_inpatient", "number_emergency", "number_outpatient", "num_medications"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    out["previous_admissions"] = out["number_inpatient"] + out["number_emergency"]
    out["medication_load"] = out["num_medications"]
    out["time_since_last_visit"] = (out["number_outpatient"] * 30
                                    + out["number_inpatient"] * 60
                                    + out["number_emergency"] * 15)
    for c in DIAG_COLS:
        out[c + "_num"] = out[c].apply(safe_float)
    out["diag_severity"] = sum(out[c + "_num"].apply(severity_score) for c in DIAG_COLS)
    return out


def fit_preprocessor(X_train):
    """Learn imputation values and label encoders from the TRAINING data only."""
    X = X_train.replace(MISSING_TOKENS, np.nan)
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num_cols = [c for c in X.columns if c not in cat_cols]
    impute = {}
    for c in cat_cols:
        impute[c] = str(X[c].dropna().astype(str).mode().iloc[0])     # mode
    for c in num_cols:
        impute[c] = float(X[c].median())                              # median
    state = {"raw_columns": list(X.columns), "categorical_cols": cat_cols,
             "numeric_cols": num_cols, "impute_values": impute}
    X = add_engineered_features(_clean_and_impute(X, state))
    encoders = {}
    for c in cat_cols:
        encoders[c] = LabelEncoder().fit(X[c].astype(str))
    state["label_encoders"] = encoders
    state["feature_columns"] = list(X.columns)
    return state


def _clean_and_impute(df, state):
    X = pd.DataFrame(df).copy()
    for c in state["raw_columns"]:
        if c not in X.columns:
            X[c] = np.nan
    X = X[state["raw_columns"]].replace(MISSING_TOKENS, np.nan)
    for c in state["numeric_cols"]:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(state["impute_values"][c])
    for c in state["categorical_cols"]:
        col = X[c].astype(object)
        col = col.where(col.notna(), state["impute_values"][c])
        X[c] = col.astype(str).str.strip()
    return X


def transform_features(df, state, return_unseen=False):
    """Raw patient rows -> model-ready numeric matrix (same columns, same order as training).

    Categories never seen in training are replaced by the most frequent training
    category of that column (and reported when return_unseen=True)."""
    X = add_engineered_features(_clean_and_impute(df, state))
    unseen = {}
    for c in state["categorical_cols"]:
        classes = state["label_encoders"][c].classes_
        mapping = {v: i for i, v in enumerate(classes)}
        codes = X[c].map(mapping)
        bad = codes.isna()
        if bad.any():
            unseen[c] = sorted(set(X.loc[bad, c]))
            codes = codes.fillna(mapping[state["impute_values"][c]])
        X[c] = codes.astype(int)
    X = X[state["feature_columns"]].astype(float)
    return (X, unseen) if return_unseen else X


def decode_features(row_enc, state):
    """Encoded row -> {feature: human-readable value} (categories back to their labels)."""
    out = {}
    for f, v in zip(state["feature_columns"], np.asarray(row_enc, dtype=float)):
        if f in state["label_encoders"]:
            classes = state["label_encoders"][f].classes_
            i = int(round(v))
            out[f] = str(classes[i]) if 0 <= i < len(classes) else "unknown"
        else:
            out[f] = int(v) if float(v).is_integer() else round(float(v), 2)
    return out
