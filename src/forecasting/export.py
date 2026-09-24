"""Export gold outputs for Tableau or DuckDB ingestion."""
from pathlib import Path

import pandas as pd


def export_gold(outputs: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    """Write each table as Parquet (typed, compact) and CSV (Tableau Public friendly)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_parquet(output_dir / f"{name}.parquet", index=False)
        frame.to_csv(output_dir / f"{name}.csv", index=False)
