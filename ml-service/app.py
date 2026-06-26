from pathlib import Path
import pickle
import re

import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS
from tensorflow.keras.preprocessing.sequence import pad_sequences


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATHS = [MODELS_DIR / "spam_model.keras", MODELS_DIR / "spam_model.h5"]
TOKENIZER_PATH = MODELS_DIR / "tokenizer.pkl"
MAX_LENGTH = 100
THRESHOLD = 0.5

app = Flask(__name__)
CORS(app)

model = None
tokenizer = None


def clean_text(text):
    """Apply the same basic normalization used before tokenization."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return " ".join(text.split())


def load_artifacts():
    """Load the saved TensorFlow model and tokenizer once at startup."""
    global model, tokenizer

    model_path = next((path for path in MODEL_PATHS if path.exists()), None)
    if model_path is None:
        raise FileNotFoundError("No trained model found. Run 3_train_model.py first.")
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError("No tokenizer found. Run 3_train_model.py first.")

    model = tf.keras.models.load_model(model_path)
    with TOKENIZER_PATH.open("rb") as file:
        tokenizer = pickle.load(file)

    print(f"Loaded model: {model_path}")
    print(f"Loaded tokenizer: {TOKENIZER_PATH}")


def predict_email(email_text):
    """Return spam prediction data for a single email."""
    cleaned = clean_text(email_text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=MAX_LENGTH, padding="post")
    spam_probability = float(model.predict(padded, verbose=0)[0][0])
    label = "spam" if spam_probability >= THRESHOLD else "ham"
    confidence = spam_probability if label == "spam" else 1 - spam_probability

    return {
        "label": label,
        "isSpam": label == "spam",
        "spamProbability": round(spam_probability, 6),
        "confidence": round(confidence, 6),
        "threshold": THRESHOLD,
    }


@app.get("/health")
def health():
    return jsonify({"status": "ok", "modelLoaded": model is not None})


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    email_text = data.get("email") or data.get("text")

    if not email_text:
        return jsonify({"error": "Email text is required."}), 400

    result = predict_email(email_text)
    return jsonify({"email": email_text, **result})


@app.post("/batch_predict")
def batch_predict():
    data = request.get_json(silent=True) or {}
    emails = data.get("emails")

    if not isinstance(emails, list) or not emails:
        return jsonify({"error": "emails must be a non-empty list."}), 400

    results = [{"email": email, **predict_email(email)} for email in emails]
    return jsonify({"count": len(results), "results": results})


if __name__ == "__main__":
    load_artifacts()
    app.run(host="0.0.0.0", port=5000, debug=True)
