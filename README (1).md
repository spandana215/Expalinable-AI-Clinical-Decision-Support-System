# Explainable AI-Based Clinical Decision Support System for Diabetes

**Case study: predicting Type 2 Diabetes patient readmission within 30 days of discharge**

A machine-learning system that estimates whether a diabetes patient is likely to be readmitted to hospital within 30 days of discharge, and explains every prediction with SHAP, LIME and an LLM-written summary. It is delivered as a six-page Streamlit dashboard.

> **Decision support only.** The output is a model prediction, not a medical diagnosis, and it does not replace the judgement of a healthcare professional.

**Live app:** https://expalinable-ai-clinical-decision-support-system-cgzj8rp7qtzjnu.streamlit.app/

---

## Features

- Predicts 30-day readmission (class 1) vs. no readmission within 30 days (class 0)
- Compares four algorithms: Random Forest, Decision Tree, Logistic Regression, XGBoost
- Leak-free pipeline: scaling and SMOTE are applied inside each cross-validation fold; the test set is never resampled
- Hyperparameter tuning, probability calibration and decision-threshold optimization
- Global and per-patient explanations with **SHAP** and **LIME**, plus a SHAP-vs-LIME agreement check
- **LLM explanation** (Google Gemini) that rewrites the model evidence in plain language, with a template fallback when no API key is available
- Streamlit dashboard for entering a patient and viewing the prediction and explanations

## Dataset

**Diabetes 130-US Hospitals (1999-2008)**, UCI Machine Learning Repository.

| Item | Value |
|---|---|
| Hospital encounters | 101,766 |
| Original columns | 50 |
| Features used by the models | 51 (44 original + 7 engineered) |
| Readmitted within 30 days (class 1) | 11,357 (11.2%) |
| Not readmitted within 30 days (class 0) | 90,409 (88.8%) |
| Training set (80%) | 81,412 rows, 144,652 after SMOTE |
| Test set (20%, untouched) | 20,354 rows, 11.16% positive |

The dataset is not included in this repository. Download `diabetic_data.csv` from the UCI repository to re-run the notebook.

## Methodology

1. **Cleaning** - `?` replaced by missing; `encounter_id`, `patient_nbr`, `weight`, `payer_code`, `medical_specialty` dropped; `None` in `max_glu_serum` / `A1Cresult` kept as a real category ("test not done").
2. **Target** - `<30` -> 1; `>30` and `NO` -> 0.
3. **Feature engineering** - `previous_admissions`, `medication_load`, `time_since_last_visit`, numeric ICD-9 codes (`diag_1_num`, `diag_2_num`, `diag_3_num`) and `diag_severity`.
4. **Split** - 80:20, stratified, `random_state=42`.
5. **Preprocessing fitted on training data only** - mode/median imputation, `LabelEncoder` (33 categorical columns), `StandardScaler`.
6. **Class imbalance** - SMOTE on the training data only.
7. **Model comparison** - four models, 5-fold stratified cross-validation with an `imblearn` pipeline (scaler -> SMOTE -> model).
8. **Model selection** - by cross-validated ROC-AUC and PR-AUC.
9. **Tuning** - `RandomizedSearchCV` (`n_iter=15`, `cv=3`, ROC-AUC).
10. **Calibration** - `CalibratedClassifierCV` (sigmoid).
11. **Threshold** - chosen from out-of-fold training predictions (F1-optimal), then evaluated once on the test set.
12. **Explainability** - SHAP (`TreeExplainer`), LIME (`LimeTabularExplainer`), LLM summary.

## Results

All numbers below come from the project notebook run on Google Colab.

### 5-fold cross-validation (training data, leak-free)

| Model | CV ROC-AUC | CV PR-AUC | CV Recall (class 1) | CV Precision (class 1) | CV F1 (class 1) |
|---|---|---|---|---|---|
| **XGBoost** | **0.6670** | **0.2173** | 0.0195 | 0.4724 | 0.0374 |
| Random Forest | 0.6382 | 0.1925 | 0.0281 | 0.3751 | 0.0522 |
| Logistic Regression | 0.6284 | 0.1893 | 0.5188 | 0.1593 | 0.2438 |
| Decision Tree | 0.5257 | 0.1184 | 0.1825 | 0.1487 | 0.1638 |

**Selected model: XGBoost** (best ROC-AUC and PR-AUC).

### Test set, default threshold 0.5 (before tuning)

| Model | Accuracy | Precision (class 1) | Recall (class 1) | F1 (class 1) | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| XGBoost | 0.8885 | 0.5062 | 0.0181 | 0.0349 | 0.6810 | 0.2254 |
| Random Forest | 0.8876 | 0.4518 | 0.0330 | 0.0616 | 0.6450 | 0.2014 |
| Logistic Regression | 0.6473 | 0.1650 | 0.5324 | 0.2520 | 0.6398 | 0.1968 |
| Decision Tree | 0.7894 | 0.1491 | 0.1885 | 0.1665 | 0.5267 | 0.1186 |

Accuracy is misleading on this dataset: always predicting "not readmitted" already gives 0.8884 accuracy with zero recall. At the default 0.5 threshold XGBoost finds under 2% of readmissions, which is why the threshold is optimized.

### Final deployed model (tuned + calibrated XGBoost)

**Best hyperparameters:** `n_estimators=300`, `max_depth=5`, `learning_rate=0.1`, `subsample=0.7`, `colsample_bytree=0.8` (cross-validated ROC-AUC 0.6640)

**Decision threshold:** 0.1198 (F1-optimal, chosen from out-of-fold training predictions)

Performance on the untouched test set (20,354 encounters):

| Metric | Default threshold 0.50 | Chosen threshold 0.120 |
|---|---|---|
| Recall (class 1) | 0.0211 | **0.4566** |
| Precision (class 1) | 0.5106 | 0.2048 |
| F1 (class 1) | 0.0406 | **0.2828** |
| Accuracy | 0.8885 | 0.7416 |

| Threshold-independent metric | Value |
|---|---|
| ROC-AUC | 0.6785 |
| PR-AUC | 0.2274 |

At the chosen threshold the system identifies about 46% of patients who are readmitted within 30 days, compared with about 2% at the default threshold. The trade-off is lower precision: roughly 1 in 5 flagged patients is actually readmitted.

**Calibration:** the mean predicted probability after calibration is 0.1112, against an actual test-set readmission rate of 0.1116 (Brier score 0.0946).

### Most important features (global SHAP)

1. `discharge_disposition_id`
2. `age`
3. `number_inpatient`
4. `num_procedures`
5. `time_in_hospital`
6. `number_diagnoses`
7. `diag_severity`
8. `time_since_last_visit`
9. `diag_1`
10. `diag_3`

Three of the engineered features (`diag_severity`, `time_since_last_visit`, and `previous_admissions` at rank 11) appear among the most influential.

### Limitations

- Discrimination is moderate (test ROC-AUC 0.68). Thirty-day readmission is hard to predict from administrative data alone.
- The data is from US hospitals in 1999-2008 and may not reflect current practice or other populations.
- A patient can appear in more than one encounter, and encounters ending in death or hospice were not excluded.
- SHAP and LIME describe what influenced the model, not what causes readmission.

## Application pages

1. **Project Overview** - title, objective, system description
2. **Patient Information** - input form and predict button
3. **Prediction Results** - predicted class, probability, threshold
4. **Explainable AI** - SHAP and LIME charts, agreement, global importance
5. **LLM Explanation** - natural-language summary
6. **Clinical Decision Support** - final summary, key evidence, clinical caution

## Project structure

```
├── app.py                  Streamlit dashboard
├── cdss_utils.py           preprocessing and feature engineering (shared by training and app)
├── cdss_xai.py             SHAP and LIME helpers
├── cdss_llm.py             LLM explanation with template fallback
├── requirements.txt
├── .streamlit/
│   └── secrets.toml.example
└── artifacts/
    ├── model.joblib                calibrated model (used for predictions)
    ├── model_raw.joblib            tuned uncalibrated model (used by SHAP)
    ├── scaler.joblib
    ├── preprocessor.joblib         imputation values and label encoders
    ├── label_encoders_dict.joblib
    ├── feature_columns.json
    ├── config.json                 model name, threshold, metrics, versions
    ├── lime_background.joblib
    └── global_shap_importance.csv
```

## Run locally

```bash
git clone https://github.com/spandana215/expalinable-ai-clinical-decision-support-system.git
cd expalinable-ai-clinical-decision-support-system
pip install -r requirements.txt
streamlit run app.py
```

### LLM key (optional)

Without a key the app shows a template explanation. To enable the LLM:

1. Create a Gemini API key at https://aistudio.google.com/app/apikey
2. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
3. Set `GEMINI_API_KEY = "your-key"`

Never commit `secrets.toml`; it is listed in `.gitignore`.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. At https://share.streamlit.io choose **Create app**, select the repository, branch `main`, main file `app.py`.
3. In **Advanced settings** select **Python 3.13** (the version used for training) and add `GEMINI_API_KEY = "your-key"` under Secrets.
4. Deploy.

## Environment used for training

Python 3.13.18, scikit-learn 1.6.1, imbalanced-learn 0.14.2, xgboost 3.4.1, shap 0.54.1, pandas 2.2.3, numpy 2.4.6.

## Team

Department of Information Science and Engineering, Nitte Meenakshi Institute of Technology, Bengaluru.

| Role | Name |
|---|---|
| Team member | Spandana V R |
| Team member | Anusha M S |
| Team member | Myna P M |
| Team member | Kumkum M K |
| Guide | Asst. Prof. Usha C S |

## Acknowledgement

Dataset: Strack et al., "Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records", BioMed Research International, 2014 (Diabetes 130-US Hospitals, UCI Machine Learning Repository).
