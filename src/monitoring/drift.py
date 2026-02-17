from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import chisquare, ks_2samp


def compute_drift_report(X_ref: pd.DataFrame, X_cur: pd.DataFrame) -> Dict:
    """
    Compare reference vs current datasets and compute base drift statistics.
    """

    report: Dict[str, Dict] = {}

    # Align columns
    common_cols = [c for c in X_ref.columns if c in X_cur.columns]
    X_ref = X_ref[common_cols]
    X_cur = X_cur[common_cols]

    for col in common_cols:
        ref_col = X_ref[col]
        cur_col = X_cur[col]

        if pd.api.types.is_numeric_dtype(ref_col):
            # KS test on non-null values
            ref_vals = ref_col.dropna().to_numpy()
            cur_vals = cur_col.dropna().to_numpy()
            if len(ref_vals) == 0 or len(cur_vals) == 0:
                continue
            stat, p_value = ks_2samp(ref_vals, cur_vals)
            report[col] = {
                "type": "numeric",
                "statistic": float(stat),
                "p_value": float(p_value),
            }
        else:
            # Categorical: chi-square on normalized frequencies
            ref_counts = ref_col.value_counts(normalize=True)
            cur_counts = cur_col.value_counts(normalize=True)
            all_categories = sorted(set(ref_counts.index).union(cur_counts.index))

            ref_freqs = np.array([ref_counts.get(cat, 0.0) for cat in all_categories])
            cur_freqs = np.array([cur_counts.get(cat, 0.0) for cat in all_categories])

            # Avoid division by zero; add small epsilon
            eps = 1e-12
            expected = ref_freqs + eps
            observed = cur_freqs + eps

            stat, p_value = chisquare(f_obs=observed, f_exp=expected)
            report[col] = {
                "type": "categorical",
                "statistic": float(stat),
                "p_value": float(p_value),
            }

    # Identify top drifting features
    sorted_features = sorted(
        report.items(), key=lambda kv: kv[1]["p_value"] if not np.isnan(kv[1]["p_value"]) else 1.0
    )
    top = [name for name, _ in sorted_features[:10]]

    return {
        "features": report,
        "top_drifting_features": top,
    }

