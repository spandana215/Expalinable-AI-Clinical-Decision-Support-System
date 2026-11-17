# Explainable AI-Based Clinical Decision Support System for Diabetes

Predicts 30-day hospital readmission for Type 2 Diabetes patients and explains each prediction
with SHAP, LIME and an LLM-written summary.

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

Optional LLM key: copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add your
Gemini key. Without a key the app uses a template explanation.

Python version used for training: 3.13.18 (use the same major.minor version when deploying).

Decision support only. Not a medical diagnosis and not a replacement for healthcare professionals.
