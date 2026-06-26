from pathlib import Path
import json
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow.keras.preprocessing.sequence import pad_sequences


BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MAX_LENGTH = 100
THRESHOLD = 0.5


def load_test_data():
    """Load the processed test split."""
    test_path = PROCESSED_DIR / "test.csv"

    if not test_path.exists():
        raise FileNotFoundError(
            f"Missing test data: {test_path}. Run 2_preprocess_data.py first."
        )

    test = pd.read_csv(test_path)
    required_columns = {"text", "label", "label_encoded"}
    missing_columns = required_columns - set(test.columns)
    if missing_columns:
        raise ValueError(
            "test.csv is missing required columns: " + ", ".join(sorted(missing_columns))
        )

    test["text"] = test["text"].fillna("").astype(str)
    print(f"Loaded {len(test)} test emails")
    return test


def load_tokenizer():
    """Load the tokenizer saved during model training."""
    tokenizer_path = MODELS_DIR / "tokenizer.pkl"

    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Missing tokenizer: {tokenizer_path}. Run 3_train_model.py first."
        )

    with tokenizer_path.open("rb") as file:
        tokenizer = pickle.load(file)

    print(f"Loaded tokenizer from {tokenizer_path}")
    return tokenizer


def load_model():
    """Load the trained TensorFlow model."""
    keras_model_path = MODELS_DIR / "spam_model.keras"
    h5_model_path = MODELS_DIR / "spam_model.h5"

    if keras_model_path.exists():
        model_path = keras_model_path
    elif h5_model_path.exists():
        model_path = h5_model_path
    else:
        raise FileNotFoundError(
            f"Missing trained model in {MODELS_DIR}. Run 3_train_model.py first."
        )

    model = tf.keras.models.load_model(model_path)
    print(f"Loaded model from {model_path}")
    return model


def vectorize_texts(tokenizer, texts):
    """Convert email text to padded token sequences."""
    sequences = tokenizer.texts_to_sequences(texts.astype(str).values)
    return pad_sequences(sequences, maxlen=MAX_LENGTH, padding="post")


def evaluate(model, x_test, y_test):
    """Evaluate predictions and return metric data."""
    probabilities = model.predict(x_test, verbose=0).ravel()
    predictions = (probabilities >= THRESHOLD).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "threshold": THRESHOLD,
    }

    report = classification_report(
        y_test,
        predictions,
        target_names=["ham", "spam"],
        zero_division=0,
        output_dict=True,
    )

    matrix = confusion_matrix(y_test, predictions)
    return probabilities, predictions, metrics, report, matrix


def save_report(metrics, report, matrix):
    """Save evaluation metrics as JSON."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = MODELS_DIR / "evaluation_report.json"

    payload = {
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    print(f"Saved evaluation report to {report_path}")


def save_confusion_matrix_plot(matrix):
    """Save a confusion matrix image."""
    output_path = MODELS_DIR / "confusion_matrix.png"

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)

    labels = ["Ham", "Spam"]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Spam Detector Confusion Matrix")

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            ax.text(
                col_index,
                row_index,
                str(matrix[row_index, col_index]),
                ha="center",
                va="center",
                color="black",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)

    print(f"Saved confusion matrix plot to {output_path}")


def print_results(metrics, matrix):
    """Print evaluation results to the terminal."""
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()

    print("\nTest Results:")
    print(f"   Accuracy:  {metrics['accuracy']:.2%}")
    print(f"   Precision: {metrics['precision']:.2%}")
    print(f"   Recall:    {metrics['recall']:.2%}")
    print(f"   F1-Score:  {metrics['f1_score']:.2%}")

    print("\nConfusion Matrix:")
    print(f"   True Ham: predicted ham={true_negative}, predicted spam={false_positive}")
    print(f"   True Spam: predicted ham={false_negative}, predicted spam={true_positive}")


if __name__ == "__main__":
    test_df = load_test_data()
    saved_tokenizer = load_tokenizer()
    spam_model = load_model()

    x_test = vectorize_texts(saved_tokenizer, test_df["text"])
    y_test = test_df["label_encoded"].values

    _, _, evaluation_metrics, classification_metrics, confusion = evaluate(
        spam_model, x_test, y_test
    )

    print_results(evaluation_metrics, confusion)
    save_report(evaluation_metrics, classification_metrics, confusion)
    save_confusion_matrix_plot(confusion)

    print("\n" + "=" * 50)
    print("Phase 4 Evaluation Complete!")
    print("=" * 50)
