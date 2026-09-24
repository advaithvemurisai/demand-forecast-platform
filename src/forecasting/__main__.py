"""Command-line entry point for the data stage: raw M5 CSVs -> silver Parquet, one state at a time."""
import argparse
import gc
from pathlib import Path

from forecasting.data import prepare_m5, write_silver_partitioned
from forecasting.pipeline import sales_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", nargs="+", default=["CA", "TX", "WI"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    raw, silver = root / "data" / "raw", root / "data" / "silver"
    source = sales_file(raw)
    for state in args.states:
        fact = prepare_m5(raw, states=[state], sales_file=source)
        write_silver_partitioned(fact, silver)
        print(f"{state}: wrote {len(fact):,} rows from {source} to data/silver")
        del fact
        gc.collect()
