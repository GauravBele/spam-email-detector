from pathlib import Path
import pickle
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "emails.txt"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def ensure_nltk_data():
    """Download the NLTK resources needed for tokenization."""
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


def load_data():
    """Load emails from the UCI dataset."""
    print("Loading dataset...")

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing dataset: {RAW_DATA_PATH}. Run 1_download_data.py first."
        )

    emails = []
    labels = []

    with RAW_DATA_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                label, text = parts
                labels.append(label)
                emails.append(text)

    df = pd.DataFrame({"label": labels, "text": emails})
    print(f"Loaded {len(df)} emails")
    return df


def clean_text(text):
    """Clean and normalize email text."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return " ".join(text.split())


def tokenize_and_filter(text, stop_words):
    """Tokenize and remove stop words."""
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(tokens)


def preprocess(df):
    """Apply all preprocessing steps."""
    print("\nPreprocessing emails...")
    ensure_nltk_data()

    df = df.copy()
    df["text"] = df["text"].apply(clean_text)
    print("Cleaned text")

    stop_words = set(stopwords.words("english"))
    df["text"] = df["text"].apply(lambda value: tokenize_and_filter(value, stop_words))
    print("Tokenized and filtered")

    label_encoder = LabelEncoder()
    df["label_encoded"] = label_encoder.fit_transform(df["label"])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with (PROCESSED_DIR / "label_encoder.pkl").open("wb") as file:
        pickle.dump(label_encoder, file)
    print("Encoded labels")

    return df, label_encoder


def split_data(df):
    """Split data into train, validation, and test sets."""
    print("\nSplitting data...")

    train, temp = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df["label"]
    )
    val, test = train_test_split(
        temp, test_size=0.5, random_state=42, stratify=temp["label"]
    )

    print(f"Train: {len(train)} ({len(train) / len(df) * 100:.1f}%)")
    print(f"Val:   {len(val)} ({len(val) / len(df) * 100:.1f}%)")
    print(f"Test:  {len(test)} ({len(test) / len(df) * 100:.1f}%)")

    return train, val, test


def save_splits(train, val, test):
    """Save train, validation, and test splits to CSV."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    print("Saved processed data")


if __name__ == "__main__":
    dataset = load_data()
    processed_dataset, _ = preprocess(dataset)
    train_df, val_df, test_df = split_data(processed_dataset)
    save_splits(train_df, val_df, test_df)

    print("\nFinal Stats:")
    print(
        f"Train - Ham: {(train_df['label'] == 'ham').sum()}, "
        f"Spam: {(train_df['label'] == 'spam').sum()}"
    )
    print(
        f"Val   - Ham: {(val_df['label'] == 'ham').sum()}, "
        f"Spam: {(val_df['label'] == 'spam').sum()}"
    )
    print(
        f"Test  - Ham: {(test_df['label'] == 'ham').sum()}, "
        f"Spam: {(test_df['label'] == 'spam').sum()}"
    )