"""cdss_llm.py - turns the ML prediction + SHAP/LIME evidence into plain language.

The LLM is ONLY an explanation writer. The prediction and probability always come
from the trained model; if the LLM is unavailable or its text fails the checks,
a deterministic template explanation is returned instead."""
import json
import os

DEFAULT_MODELS = ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-2.5-flash"]  # tried in order

SECTION_TITLES = [
    "Model Prediction",
    "Prediction Probability",
    "Important Risk-Increasing Factors",
    "Important Risk-Reducing Factors",
    "SHAP and LIME Explanation",
    "Human-Readable Summary",
    "Clinical Caution",
]

CAUTION = ("This is the output of a machine-learning model trained on historical hospital data. "
           "It is decision support only: it is not a diagnosis, it does not recommend treatment, "
           "and it must not replace the judgement of a qualified healthcare professional.")

SYSTEM_PROMPT = """You are the explanation component of a clinical decision support system that
predicts 30-day hospital readmission for diabetes patients. A trained machine-learning model has
ALREADY made the prediction. Your only job is to explain the supplied evidence in clear language
for a healthcare professional.

Strict rules:
1. Use ONLY the facts in the JSON evidence. Do not add medical knowledge, diagnoses, causes,
   treatments, medication advice or recommendations of any kind.
2. Never change, round differently, or re-estimate the prediction, probability or threshold.
   Copy "prediction", "probability_text" and "threshold_text" exactly as given.
3. Describe features as factors that pushed the MODEL'S prediction up or down. Do not claim
   they cause readmission.
4. SHAP and LIME numbers are on different scales: compare direction and ranking only.
5. Write exactly these seven numbered sections, with these titles, in this order:
   1. Model Prediction
   2. Prediction Probability
   3. Important Risk-Increasing Factors
   4. Important Risk-Reducing Factors
   5. SHAP and LIME Explanation
   6. Human-Readable Summary
   7. Clinical Caution
6. Keep it under 300 words. Plain text, short sentences, no tables."""


def build_llm_payload(result, top_n=6):
    """Only the prediction and the top explanation evidence - no IDs, no full patient record."""
    shap_rows = result["shap_df"].head(top_n)
    lime_rows = result["lime_df"].reindex(
        result["lime_df"]["lime_weight"].abs().sort_values(ascending=False).index).head(top_n)
    return {
        "prediction": result["predicted_label"],
        "probability_text": "%.1f%%" % (100 * result["proba_readmitted"]),
        "threshold_text": "%.1f%%" % (100 * result["threshold"]),
        "decision_rule": "classified as readmitted when probability >= threshold",
        "model": result["model_name"],
        "shap_units": result["shap_units"],
        "shap_top_features": [
            {"feature": r.feature, "patient_value": str(r.patient_value),
             "contribution": round(float(r.shap_value), 4), "direction": r.direction}
            for r in shap_rows.itertuples()],
        "lime_top_features": [
            {"feature": r.feature, "rule": r.lime_rule,
             "weight": round(float(r.lime_weight), 4), "direction": r.direction}
            for r in lime_rows.itertuples()],
        "shap_lime_agreement": {
            "features_in_both_top_lists": result["agreement"]["common_features"],
            "same_direction_in_both": result["agreement"]["same_direction_features"],
            "jaccard_overlap": result["agreement"]["jaccard"],
        },
    }


def _join(items):
    return "; ".join(items) if items else "none among the top factors"


def template_explanation(payload):
    """Deterministic fallback with the same seven sections (no LLM needed)."""
    up = ["%s = %s" % (f["feature"], f["patient_value"])
          for f in payload["shap_top_features"] if f["contribution"] > 0]
    down = ["%s = %s" % (f["feature"], f["patient_value"])
            for f in payload["shap_top_features"] if f["contribution"] < 0]
    lime_up = [f["rule"] for f in payload["lime_top_features"] if f["weight"] > 0]
    lime_down = [f["rule"] for f in payload["lime_top_features"] if f["weight"] < 0]
    agree = payload["shap_lime_agreement"]
    side = "at or above" if payload["prediction"].startswith("Readmitted") else "below"
    lines = [
        "1. Model Prediction",
        "The model classifies this patient as: %s." % payload["prediction"],
        "",
        "2. Prediction Probability",
        "Estimated probability of readmission within 30 days: %s. This is %s the decision "
        "threshold of %s." % (payload["probability_text"], side, payload["threshold_text"]),
        "",
        "3. Important Risk-Increasing Factors",
        "According to SHAP, these pushed the prediction towards readmission: %s." % _join(up),
        "",
        "4. Important Risk-Reducing Factors",
        "According to SHAP, these pushed the prediction away from readmission: %s." % _join(down),
        "",
        "5. SHAP and LIME Explanation",
        "LIME rules raising the prediction: %s. LIME rules lowering it: %s. "
        "Features in the top list of both methods: %s (overlap score %.2f); of these, both "
        "methods give the same direction for: %s. The two methods use different scales, so "
        "only ranking and direction are compared."
        % (_join(lime_up), _join(lime_down), _join(agree["features_in_both_top_lists"]),
           agree["jaccard_overlap"], _join(agree["same_direction_in_both"])),
        "",
        "6. Human-Readable Summary",
        "Based on the %s model, this patient is predicted as '%s' with a readmission "
        "probability of %s. The factors listed above describe what influenced the model, "
        "not what causes readmission."
        % (payload["model"], payload["prediction"], payload["probability_text"]),
        "",
        "7. Clinical Caution",
        CAUTION,
    ]
    return "\n".join(lines)


def get_api_key():
    """Colab Secrets first, then environment variables. Never hardcode the key."""
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        try:
            from google.colab import userdata
            key = userdata.get(name)
            if key:
                return key
        except Exception:
            pass
        if os.environ.get(name):
            return os.environ[name]
    return None


def call_gemini(payload, api_key, models=None, temperature=0.2):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    prompt = ("Explain this model output. Evidence (JSON):\n"
              + json.dumps(payload, indent=2, default=str))
    last_error = None
    for name in (models or DEFAULT_MODELS):
        try:
            resp = client.models.generate_content(
                model=name, contents=prompt,
                config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT,
                                                   temperature=temperature))
            text = (getattr(resp, "text", None) or "").strip()
            if text:
                return text, name
            last_error = "empty response from " + name
        except Exception as err:          # wrong model name, quota, network ...
            last_error = "%s: %s" % (name, err)
    raise RuntimeError(str(last_error))


def check_llm_text(text, payload):
    """Reject LLM text that drops sections or alters the model's numbers/label."""
    low = text.lower()
    problems = [t for t in SECTION_TITLES if t.lower() not in low]
    if payload["probability_text"] not in text:
        problems.append("probability changed or missing")
    if payload["prediction"].lower() not in low:
        problems.append("prediction label changed or missing")
    return problems


def generate_explanation(result, api_key=None, models=None, use_llm=True):
    """Returns {"text", "source", "note", "payload"}; never raises."""
    payload = build_llm_payload(result)
    note = "LLM disabled - template explanation used."
    if use_llm:
        key = api_key or get_api_key()
        if not key:
            note = "No API key found - template explanation used."
        else:
            try:
                text, used = call_gemini(payload, key, models)
                problems = check_llm_text(text, payload)
                if not problems:
                    return {"text": text, "source": "LLM (%s)" % used, "note": "", "payload": payload}
                note = "LLM text failed safety checks (%s) - template used." % ", ".join(problems)
            except Exception as err:
                note = "LLM unavailable (%s) - template explanation used." % str(err)[:200]
    return {"text": template_explanation(payload), "source": "template", "note": note,
            "payload": payload}
