"""
app.py
------
Flask web application for the Fake News Detection System.

Routes
------
GET  /           – Render the main UI (index.html)
POST /predict    – Accept news text, return REAL/FAKE + confidence JSON
GET  /health     – Simple health-check endpoint

Usage
-----
    python app.py

The server runs on http://0.0.0.0:5000 by default.

Prerequisites
-------------
    model.h5 and tokenizer.pkl must exist (run train.py first).
"""
from fact_api import google_fact_check
import os
import json

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np

from utils import (
    combined_prediction,
    fact_check_score,
    load_tokenizer,
    texts_to_padded_sequences,
    MAX_SEQ_LENGTH,
)


BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")


app = Flask(__name__)

_model = None
_tokenizer = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Please run train.py first."
            )
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        if not os.path.exists(TOKENIZER_PATH):
            raise FileNotFoundError(
                f"Tokenizer file not found at {TOKENIZER_PATH}. "
                "Please run train.py first."
            )
        _tokenizer = load_tokenizer(TOKENIZER_PATH)
    return _tokenizer



def predict_news(text: str):
    model = get_model()
    tokenizer = get_tokenizer()

    X = texts_to_padded_sequences([text], tokenizer, maxlen=MAX_SEQ_LENGTH)
    ml_prob = float(model.predict(X, verbose=0)[0][0])

    rb_score = fact_check_score(text)

    # NEW FACT CHECK
    fact_score = google_fact_check(text)

    # COMBINE ALL
    final_score = (ml_prob * 0.6) + (rb_score * 0.2) + (fact_score * 0.2)

    label = "FAKE" if final_score > 0.5 else "REAL"
    confidence = round(final_score * 100, 2)

    return label, confidence, ml_prob, rb_score, fact_score



@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided."}), 400

    try:
        label, confidence, ml_prob, rb_score, fact_score = predict_news(text)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503

    return jsonify(
        {
            "label": label,
            "confidence": confidence,
            "ml_probability": round(ml_prob * 100, 1),
            "rule_based_score": round(rb_score * 100, 1),
            "fact_check_score": round(fact_score * 100, 1),
            "explanation": _build_explanation(label, ml_prob, rb_score),
        }
    )


@app.route("/health")
def health():
    return jsonify({"status": "ok"})



def _build_explanation(label: str, ml_prob: float, rb_score: float) -> str:
    parts = []
    if label == "FAKE":
        parts.append(
            f"The ML model assigned a {ml_prob * 100:.1f}% fake-news probability."
        )
        if rb_score > 0:
            parts.append(
                f"The text also contains sensationalist language "
                f"(rule-based score: {rb_score * 100:.0f}%)."
            )
        else:
            parts.append("No strong sensationalist keywords were detected.")
    else:
        parts.append(
            f"The ML model assigned only a {ml_prob * 100:.1f}% fake-news probability."
        )
        parts.append("The text does not appear to contain sensationalist language.")
    return " ".join(parts)



if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
