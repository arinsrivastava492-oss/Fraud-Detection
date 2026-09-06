"""
Downloads the Credit Card Fraud Detection dataset (ULB / Kaggle mirror)
and unpacks it into data/creditcard.csv.

The raw file is ~150 MB, so it isn't committed to the repo. Run this
once before training.
"""
import io
import os
import urllib.request
import zipfile

MIRROR_URL = (
    "https://raw.githubusercontent.com/stat432/credit-analysis/main/"
    "data-raw/creditcard.csv.zip"
)
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_PATH = os.path.join(OUT_DIR, "creditcard.csv")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(OUT_PATH):
        print(f"Dataset already present at {OUT_PATH}, skipping download.")
        return

    print(f"Downloading dataset from {MIRROR_URL} ...")
    with urllib.request.urlopen(MIRROR_URL) as resp:
        data = resp.read()

    print("Unzipping ...")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(OUT_DIR)

    print(f"Done. Dataset saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
