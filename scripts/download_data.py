"""Download the M5 inputs into data/raw with no account or manual step.

Default source: Nixtla's public mirror (the archive behind ``datasetsforecast``),
pinned by SHA-256 and checked against the official M5 shapes. ``--source kaggle``
uses the official competition download instead (requires accepted rules).

    python scripts/download_data.py            # public mirror
    python scripts/download_data.py --source kaggle
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import ssl
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MIRROR_URL = "https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip"
MIRROR_SHA256 = "cc704ba15d6802f8262e6ec7d4c6041e4ad6366a94365e8c84f721e450eed774"
FILES = ("sales_train_evaluation.csv", "sales_train_validation.csv", "calendar.csv", "sell_prices.csv")
EXPECTED = {
    "sales_series": 30490,
    "sales_last_day": "d_1941",
    "calendar_days": 1969,
    "price_rows": 6841121,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ssl_context() -> ssl.SSLContext:
    """Verify TLS with certifi's CA bundle when present (python.org macOS builds ship no system CAs)."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def from_mirror() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "m5.zip"
        print(f"Downloading {MIRROR_URL} (~50 MB)")
        with urllib.request.urlopen(MIRROR_URL, context=ssl_context()) as response, archive.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        actual = sha256(archive)
        if actual != MIRROR_SHA256:
            sys.exit(f"Checksum mismatch: expected {MIRROR_SHA256}, got {actual}. The mirror changed; refusing to use it.")
        with zipfile.ZipFile(archive) as bundle:
            for name in FILES:
                bundle.extract(name, RAW)


def from_kaggle() -> None:
    subprocess.run(["kaggle", "competitions", "download", "-c", "m5-forecasting-accuracy", "-p", str(RAW)], check=True)
    with zipfile.ZipFile(RAW / "m5-forecasting-accuracy.zip") as bundle:
        bundle.extractall(RAW)


def validate() -> None:
    """Fail loudly unless the files have the official M5 shapes."""
    sales = pd.read_csv(RAW / "sales_train_evaluation.csv", usecols=["item_id", "store_id", "d_1941"])
    header = pd.read_csv(RAW / "sales_train_evaluation.csv", nrows=0).columns
    days = [column for column in header if column.startswith("d_")]
    checks = {
        "sales_series": len(sales),
        "sales_last_day": days[-1],
        "calendar_days": len(pd.read_csv(RAW / "calendar.csv", usecols=["date"])),
        "price_rows": len(pd.read_csv(RAW / "sell_prices.csv", usecols=["sell_price"])),
    }
    failed = {key: (value, EXPECTED[key]) for key, value in checks.items() if value != EXPECTED[key]}
    if failed:
        sys.exit(f"M5 files do not match the official shapes (got, expected): {failed}")
    print("Validated:", ", ".join(f"{key}={value}" for key, value in checks.items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["mirror", "kaggle"], default="mirror")
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    from_mirror() if args.source == "mirror" else from_kaggle()
    validate()
    print("M5 files ready in", RAW)
