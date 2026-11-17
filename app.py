"""Streamlit front-end: Explainable AI Clinical Decision Support System
30-day readmission prediction for Type 2 Diabetes patients.
Run locally with:  streamlit run app.py"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from cdss_llm import CAUTION, generate_explanation
from cdss_xai import CLASS_NAMES, analyze_patient, load_bundle, plot_contributions

st.set_page_config(page_title="Diabetes Readmission CDSS", page_icon="🏥", layout="wide")

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
PAGES = ["1. Project Overview", "2. Patient Information", "3. Prediction Results",
         "4. Explainable AI", "5. LLM Explanation", "6. Clinical Decision Support"]
MED_COLS = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
            "glipizide", "glyburide", "pioglitazone", "rosiglitazone", "acarbose",
            "glyburide-metformin"]


@st.cache_resource(show_spinner="Loading model and explainers ...")
def get_bundle():
    return load_bundle(ARTIFACTS_DIR)


def get_llm_key():
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def need_result():
    if "result" not in st.session_state:
        st.info("No prediction yet. Open **2. Patient Information**, fill the form and press "
                "**Predict readmission risk**.")
        return None
    return st.session_state["result"]


def pick(label, col, ui, default=None):
    options = ui["categorical"][col]
    idx = options.index(default) if default in options else 0
    return st.selectbox(label, options, index=idx)


def number(label, col, ui):
    r = ui["numeric"][col]
    return st.number_input(label, min_value=0, max_value=int(max(r["max"], 1)),
                           value=int(r["median"]), step=1)


bundle = get_bundle()
config, state = bundle["config"], bundle["state"]
ui = config["ui_options"]

st.sidebar.title("🏥 Readmission CDSS")
page = st.sidebar.radio("Navigation", PAGES)
st.sidebar.markdown("---")
st.sidebar.caption("Model: %s  \nDecision threshold: %.3f" % (bundle["model_name"], bundle["threshold"]))
st.sidebar.warning("Decision support only. Not a medical diagnosis.")

# ------------------------------------------------------------------ PAGE 1
if page == PAGES[0]:
    st.title("Explainable AI-Based Clinical Decision Support System for Diabetes")
    st.subheader("Predicting Type 2 Diabetes patient readmission within 30 days of discharge")
    st.markdown("""
**Objective.** Estimate whether a diabetes patient is likely to be readmitted to hospital within
30 days of discharge, and explain *why* the model gave that estimate.

**How the system works**
1. Patient details are entered on the *Patient Information* page.
2. The same preprocessing and feature engineering used in training are applied.
3. A calibrated **%s** model returns the probability of readmission within 30 days.
4. **SHAP** and **LIME** show which features pushed the prediction up or down.
5. An **LLM** rewrites that evidence in plain language (a template is used if the LLM is unavailable).

**Data.** Diabetes 130-US Hospitals dataset (UCI Machine Learning Repository), 1999-2008,
about 101,766 hospital encounters.
""" % bundle["model_name"])
    m = config.get("test_metrics", {})
    if m:
        st.markdown("**Held-out test-set performance of the deployed model**")
        cols = st.columns(len(m))
        for c, (k, v) in zip(cols, m.items()):
            c.metric(k, "%.3f" % v)
    st.warning(CAUTION)

# ------------------------------------------------------------------ PAGE 2
elif page == PAGES[1]:
    st.title("Patient Information")
    st.caption("Enter the encounter details. Fields not shown use the most common training value.")
    with st.form("patient_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Demographics & admission**")
            race = pick("Race", "race", ui, "Caucasian")
            gender = pick("Gender", "gender", ui, "Female")
            age = pick("Age group", "age", ui, "[60-70)")
            adm_type = st.selectbox("Admission type ID", ui["ids"]["admission_type_id"])
            disch = st.selectbox("Discharge disposition ID", ui["ids"]["discharge_disposition_id"])
            adm_src = st.selectbox("Admission source ID", ui["ids"]["admission_source_id"],
                                   index=ui["ids"]["admission_source_id"].index(7)
                                   if 7 in ui["ids"]["admission_source_id"] else 0)
        with c2:
            st.markdown("**Hospital stay & history**")
            time_in_hospital = number("Time in hospital (days)", "time_in_hospital", ui)
            num_lab = number("Number of lab procedures", "num_lab_procedures", ui)
            num_proc = number("Number of procedures", "num_procedures", ui)
            num_meds = number("Number of medications", "num_medications", ui)
            n_out = number("Outpatient visits (past year)", "number_outpatient", ui)
            n_emg = number("Emergency visits (past year)", "number_emergency", ui)
            n_inp = number("Inpatient visits (past year)", "number_inpatient", ui)
            n_diag = number("Number of diagnoses", "number_diagnoses", ui)
        with c3:
            st.markdown("**Diagnoses, tests & medication**")
            diag_1 = st.text_input("Primary diagnosis (ICD-9)", "428")
            diag_2 = st.text_input("Secondary diagnosis (ICD-9)", "250")
            diag_3 = st.text_input("Tertiary diagnosis (ICD-9)", "401")
            glu = pick("Max glucose serum", "max_glu_serum", ui, "None")
            a1c = pick("HbA1c result", "A1Cresult", ui, "None")
            insulin = pick("Insulin", "insulin", ui, "No")
            change = pick("Change in medications", "change", ui, "No")
            diabetes_med = pick("Diabetes medication prescribed", "diabetesMed", ui, "Yes")
            other_meds = st.multiselect("Other diabetes medicines taken (steady dose)",
                                        [m for m in MED_COLS if m in state["raw_columns"]])
        use_llm = st.checkbox("Generate LLM explanation", value=True)
        submitted = st.form_submit_button("Predict readmission risk", type="primary")

    if submitted:
        patient = dict(state["impute_values"])          # defaults = most common training values
        patient.update({
            "race": race, "gender": gender, "age": age,
            "admission_type_id": adm_type, "discharge_disposition_id": disch,
            "admission_source_id": adm_src, "time_in_hospital": time_in_hospital,
            "num_lab_procedures": num_lab, "num_procedures": num_proc,
            "num_medications": num_meds, "number_outpatient": n_out,
            "number_emergency": n_emg, "number_inpatient": n_inp, "number_diagnoses": n_diag,
            "max_glu_serum": glu, "A1Cresult": a1c, "insulin": insulin,
            "diag_1": diag_1.strip(), "diag_2": diag_2.strip(), "diag_3": diag_3.strip(),
            "change": change, "diabetesMed": diabetes_med,
        })
        for m in other_meds:
            patient[m] = "Steady"
        with st.spinner("Running prediction, SHAP and LIME ..."):
            result = analyze_patient(patient, bundle)
        with st.spinner("Writing explanation ..."):
            explanation = generate_explanation(result, api_key=get_llm_key(), use_llm=use_llm)
        st.session_state["patient"] = patient
        st.session_state["result"] = result
        st.session_state["explanation"] = explanation
        st.success("Prediction complete: **%s** (probability %.1f%%). Open the other pages for details."
                   % (result["predicted_label"], 100 * result["proba_readmitted"]))
        for col, vals in result["unseen_categories"].items():
            st.warning("Value %s for '%s' was not seen in training; the most common training "
                       "value was used instead." % (vals, col))

# ------------------------------------------------------------------ PAGE 3
elif page == PAGES[2]:
    st.title("Prediction Results")
    res = need_result()
    if res:
        c1, c2, c3 = st.columns(3)
        c1.metric("Model prediction", "Readmitted < 30 days" if res["predicted_class"] else "Not readmitted < 30 days")
        c2.metric("Probability of readmission", "%.1f%%" % (100 * res["proba_readmitted"]))
        c3.metric("Decision threshold", "%.1f%%" % (100 * res["threshold"]))
        if res["predicted_class"] == 1:
            st.error("Model output: **higher predicted risk** - probability is at or above the threshold.")
        else:
            st.success("Model output: **lower predicted risk** - probability is below the threshold.")
        st.progress(min(max(res["proba_readmitted"], 0.0), 1.0))
        st.write("Probability of **not** being readmitted within 30 days: %.1f%%"
                 % (100 * res["proba_not_readmitted"]))
        st.caption("The threshold was chosen on out-of-fold training predictions to balance "
                   "precision and recall for the readmitted class. " + CAUTION)
        with st.expander("Patient input summary"):
            shown = {k: v for k, v in st.session_state["patient"].items()}
            st.dataframe(pd.DataFrame({"value": pd.Series(shown).astype(str)}))

# ------------------------------------------------------------------ PAGE 4
elif page == PAGES[3]:
    st.title("Explainable AI: SHAP and LIME")
    res = need_result()
    if res:
        k = res["top_k"]
        st.caption("Red bars push the prediction towards readmission, blue bars away from it. "
                   "SHAP values are in %s units of the tuned model; LIME weights come from a local "
                   "surrogate model. Compare direction and ranking, not the raw numbers."
                   % res["shap_units"])
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("SHAP (this patient)")
            st.pyplot(plot_contributions(res["shap_df"], "shap_value", "Top %d SHAP contributions" % k, top_k=k))
            st.dataframe(res["shap_df"].head(k)[["feature", "patient_value", "shap_value", "direction"]], hide_index=True)
        with c2:
            st.subheader("LIME (this patient)")
            st.pyplot(plot_contributions(res["lime_df"], "lime_weight", "Top %d LIME weights" % k, top_k=k))
            st.dataframe(res["lime_df"][["feature", "lime_rule", "lime_weight", "direction"]], hide_index=True)
        st.subheader("SHAP waterfall")
        try:
            import shap
            plt.figure()
            shap.plots.waterfall(shap.Explanation(
                values=res["shap_values"], base_values=res["shap_base"], data=res["x_enc"],
                feature_names=bundle["feature_columns"]), max_display=12, show=False)
            st.pyplot(plt.gcf())
            plt.close("all")
        except Exception as err:
            st.caption("Waterfall plot unavailable (%s)." % err)
        st.subheader("Do SHAP and LIME agree?")
        a = res["agreement"]
        m1, m2 = st.columns(2)
        m1.metric("Top-%d overlap (Jaccard)" % k, "%.2f" % a["jaccard"])
        m2.metric("Same direction on shared features",
                  "n/a" if a["sign_agreement"] is None else "%.0f%%" % (100 * a["sign_agreement"]))
        st.dataframe(res["compare_df"], hide_index=True)
        gpath = os.path.join(ARTIFACTS_DIR, "global_shap_importance.csv")
        if os.path.exists(gpath):
            st.subheader("Important features overall (global SHAP)")
            g = pd.read_csv(gpath).head(15).iloc[::-1]
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.barh(g["feature"], g["mean_abs_shap"], color="#2a7f9e")
            ax.set_xlabel("mean |SHAP value| on a test-set sample")
            fig.tight_layout()
            st.pyplot(fig)

# ------------------------------------------------------------------ PAGE 5
elif page == PAGES[4]:
    st.title("LLM Explanation")
    res = need_result()
    if res:
        exp = st.session_state["explanation"]
        st.caption("Source: %s. The prediction and probability shown here always come from the "
                   "machine-learning model, never from the LLM." % exp["source"])
        if exp["note"]:
            st.info(exp["note"])
        c1, c2 = st.columns(2)
        c1.metric("Model prediction", res["predicted_label"])
        c2.metric("Probability of readmission", "%.1f%%" % (100 * res["proba_readmitted"]))
        st.text(exp["text"])
        with st.expander("Evidence sent to the LLM (no identifiers)"):
            st.json(exp["payload"])

# ------------------------------------------------------------------ PAGE 6
else:
    st.title("Clinical Decision Support Summary")
    res = need_result()
    if res:
        st.subheader("Final prediction summary")
        st.markdown("- **Model prediction:** %s\n- **Probability of readmission within 30 days:** %.1f%%\n"
                    "- **Decision threshold:** %.1f%%\n- **Model:** calibrated %s"
                    % (res["predicted_label"], 100 * res["proba_readmitted"],
                       100 * res["threshold"], res["model_name"]))
        st.subheader("Important model evidence")
        up = res["shap_df"][res["shap_df"]["shap_value"] > 0].head(5)
        down = res["shap_df"][res["shap_df"]["shap_value"] < 0].head(5)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Pushed the prediction up**")
            for r in up.itertuples():
                st.markdown("- %s = %s" % (r.feature, r.patient_value))
        with c2:
            st.markdown("**Pushed the prediction down**")
            for r in down.itertuples():
                st.markdown("- %s = %s" % (r.feature, r.patient_value))
        st.caption("Features confirmed by both SHAP and LIME: %s"
                   % (", ".join(res["agreement"]["same_direction_features"]) or "none"))
        st.subheader("Clinical caution")
        st.warning(CAUTION)
        report = ("DIABETES READMISSION CDSS REPORT\n\n" + st.session_state["explanation"]["text"])
        st.download_button("Download report (.txt)", report, file_name="cdss_report.txt")
