from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
VOCAB_SIZE = 5000
MAX_LENGTH = 100


def load_data():
    """Load preprocessed data."""
    print("Loading preprocessed data...")

    required_files = ["train.csv", "val.csv", "test.csv"]
    missing_files = [
        filename for filename in required_files if not (PROCESSED_DIR / filename).exists()
    ]
    if missing_files:
        raise FileNotFoundError(
            "Missing processed files: "
            + ", ".join(missing_files)
            + ". Run 2_preprocess_data.py first."
        )

    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    val = pd.read_csv(PROCESSED_DIR / "val.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")

    train["text"] = train["text"].fillna("")
    val["text"] = val["text"].fillna("")
    test["text"] = test["text"].fillna("")

    print(f"Train: {len(train)}")
    print(f"Val: {len(val)}")
    print(f"Test: {len(test)}")

    return train, val, test


def tokenize_texts(train, val, test, vocab_size=VOCAB_SIZE):
    """Tokenize and pad text data."""
    print(f"\nTokenizing texts with vocab_size={vocab_size}...")

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(train["text"].astype(str).values)

    x_train = tokenizer.texts_to_sequences(train["text"].astype(str).values)
    x_val = tokenizer.texts_to_sequences(val["text"].astype(str).values)
    x_test = tokenizer.texts_to_sequences(test["text"].astype(str).values)

    x_train = pad_sequences(x_train, maxlen=MAX_LENGTH, padding="post")
    x_val = pad_sequences(x_val, maxlen=MAX_LENGTH, padding="post")
    x_test = pad_sequences(x_test, maxlen=MAX_LENGTH, padding="post")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with (MODELS_DIR / "tokenizer.pkl").open("wb") as file:
        pickle.dump(tokenizer, file)

    print(f"Tokenized {len(x_train)} train sequences")
    print(f"Padded to length: {MAX_LENGTH}")
    print("Tokenizer saved")

    return x_train, x_val, x_test, tokenizer


def build_model(vocab_size=VOCAB_SIZE, max_length=MAX_LENGTH):
    """Build a TensorFlow neural network."""
    print("\nBuilding neural network model...")

    model = Sequential(
        [
            keras.Input(shape=(max_length,)),
            Embedding(input_dim=vocab_size, output_dim=128),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.2),
            Bidirectional(LSTM(32)),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    print("Model created")
    model.summary()
    return model


def train_model(model, x_train, y_train, x_val, y_val):
    """Train the neural network."""
    print("\nTraining model...")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=15,
        batch_size=32,
        validation_data=(x_val, y_val),
        callbacks=[early_stop],
        verbose=1,
    )

    print("Training complete")
    return history


def evaluate_model(model, x_test, y_test):
    """Evaluate model on test data."""
    print("\nEvaluating model on test data...")
    loss, accuracy, precision, recall = model.evaluate(x_test, y_test, verbose=0)
    f1_score = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    print("\nTest Results:")
    print(f"   Loss:      {loss:.4f}")
    print(f"   Accuracy:  {accuracy:.2%}")
    print(f"   Precision: {precision:.2%}")
    print(f"   Recall:    {recall:.2%}")
    print(f"   F1-Score:  {f1_score:.2%}")

    return loss, accuracy, precision, recall


def save_model(model):
    """Save trained model."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / "spam_model.keras")
    model.save(MODELS_DIR / "spam_model.h5")
    print(f"Model saved to {MODELS_DIR / 'spam_model.keras'}")


def plot_training_history(history):
    """Plot training and validation metrics."""
    print("\nGenerating training plots...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Model Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(MODELS_DIR / "training_history.png", dpi=100)
    plt.close(fig)
    print(f"Training plots saved to {MODELS_DIR / 'training_history.png'}")


if __name__ == "__main__":
    train_df, val_df, test_df = load_data()

    x_train, x_val, x_test, _ = tokenize_texts(train_df, val_df, test_df)

    y_train = train_df["label_encoded"].values
    y_val = val_df["label_encoded"].values
    y_test = test_df["label_encoded"].values

    spam_model = build_model()
    training_history = train_model(spam_model, x_train, y_train, x_val, y_val)
    evaluate_model(spam_model, x_test, y_test)
    save_model(spam_model)
    plot_training_history(training_history)

    print("\n" + "=" * 50)
    print("Phase 3 Complete!")
    print("=" * 50)
