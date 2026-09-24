"""Demand drift detection and retraining triggers."""
from __future__ import annotations

import numpy as np
import pandas as pd


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population stability index over quantile bins of ``expected``.

    The outer bins are open-ended so that values outside the reference range
    (the usual symptom of drift) are counted rather than silently dropped.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    inner = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1))[1:-1])
    edges = np.concatenate([[-np.inf], inner, [np.inf]])
    if len(edges) < 3:
        return 0.0
    expected_counts = np.histogram(expected, bins=edges)[0] / len(expected)
    actual_counts = np.histogram(actual, bins=edges)[0] / len(actual)
    expected_counts = np.clip(expected_counts, 1e-6, None)
    actual_counts = np.clip(actual_counts, 1e-6, None)
    return float(np.sum((actual_counts - expected_counts) * np.log(actual_counts / expected_counts)))


def detect_drift(expected: pd.Series, actual: pd.Series, threshold: float = 0.2) -> dict[str, float | bool]:
    score = psi(expected, actual)
    return {"psi": score, "drift": score >= threshold}


def retraining_flag(
    psi_scores: pd.Series,
    recent_wmape: float,
    reference_wmape: float,
    psi_threshold: float = 0.2,
    degradation: float = 0.10,
) -> dict[str, object]:
    """Retrain when inputs drift materially or accuracy decays by more than ``degradation``."""
    drifted = psi_scores[psi_scores >= psi_threshold]
    decay = (recent_wmape - reference_wmape) / reference_wmape if reference_wmape else 0.0
    reasons = []
    if len(drifted):
        reasons.append(f"PSI >= {psi_threshold} for {len(drifted)} node(s): {', '.join(map(str, drifted.index))}")
    if decay > degradation:
        reasons.append(f"WMAPE degraded {decay:.1%} vs reference")
    return {"retrain": bool(reasons), "wmape_decay": decay, "reasons": "; ".join(reasons) or "stable"}
