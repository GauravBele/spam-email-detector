from pathlib import Path
import urllib.request
import zipfile


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
DOWNLOAD_PATH = RAW_DIR / "spam.zip"
EXTRACT_PATH = RAW_DIR
EMAILS_PATH = RAW_DIR / "emails.txt"


def download_uci_spam():
    """Download the UCI SMS Spam Collection dataset."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"

    print("Downloading UCI Spam Collection dataset...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(url, DOWNLOAD_PATH)
        print(f"Downloaded to {DOWNLOAD_PATH}")

        with zipfile.ZipFile(DOWNLOAD_PATH, "r") as zip_ref:
            zip_ref.extractall(EXTRACT_PATH)
        print(f"Extracted to {EXTRACT_PATH}")

        source_path = RAW_DIR / "SMSSpamCollection"
        if source_path.exists():
            if EMAILS_PATH.exists():
                EMAILS_PATH.unlink()
            source_path.rename(EMAILS_PATH)
            print(f"Dataset ready at {EMAILS_PATH}")

        with EMAILS_PATH.open("r", encoding="utf-8") as file:
            lines = file.readlines()

        total = len(lines)
        ham = sum(1 for line in lines if line.startswith("ham"))
        spam = sum(1 for line in lines if line.startswith("spam"))

        print("\nDataset Stats:")
        print(f"   Total: {total} emails")
        print(f"   Ham: {ham} ({ham / total * 100:.1f}%)")
        print(f"   Spam: {spam} ({spam / total * 100:.1f}%)")

    except Exception as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    download_uci_spam()