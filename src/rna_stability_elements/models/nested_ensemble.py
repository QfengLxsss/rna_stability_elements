from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


METRICS = ("pearson", "spearman", "r2", "rmse", "mae")


def blend(frame: pd.DataFrame, ramht_weight: float) -> np.ndarray:
    return (
        ramht_weight * frame["y_pred_ramht"].to_numpy()
        + (1.0 - ramht_weight) * frame["y_pred_xgboost"].to_numpy()
    )


def summarize_nested(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (label_id, evaluation), group in metrics.groupby(["label_id", "evaluation"], sort=False):
        row = {
            "label_id": label_id,
            "evaluation": evaluation,
            "n_splits": group["split_name"].nunique(),
            "selected_ramht_weight_mean": group["selected_ramht_weight"].mean(),
            "selected_ramht_weight_median": group["selected_ramht_weight"].median(),
            "residual_correlation_mean": group["residual_correlation"].mean(),
        }
        for metric in METRICS:
            row[f"xgboost_{metric}_mean"] = group[f"xgboost_{metric}"].mean()
            row[f"blend_{metric}_mean"] = group[f"blend_{metric}"].mean()
            row[f"delta_{metric}_mean"] = group[f"delta_{metric}"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def paired_statistics(
    metrics: pd.DataFrame,
    *,
    bootstrap_repeats: int,
    permutation_repeats: int,
    random_state: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows = []
    groups = list(metrics.groupby(["label_id", "evaluation"], sort=False))
    groups.extend(
        ((label_id, "all_outer_splits"), group)
        for label_id, group in metrics.groupby("label_id", sort=False)
    )
    for (label_id, evaluation), group in groups:
        for metric in METRICS:
            delta = group[f"delta_{metric}"].dropna().to_numpy(dtype=float)
            if not len(delta):
                continue
            bootstrap = rng.choice(delta, size=(bootstrap_repeats, len(delta)), replace=True).mean(axis=1)
            observed = float(delta.mean())
            signs = rng.choice(np.array([-1.0, 1.0]), size=(permutation_repeats, len(delta)))
            permuted = (signs * delta).mean(axis=1)
            try:
                wilcoxon_p = float(wilcoxon(delta, alternative="two-sided").pvalue)
            except ValueError:
                wilcoxon_p = 1.0
            higher_is_better = metric not in {"rmse", "mae"}
            wins = int((delta > 0).sum()) if higher_is_better else int((delta < 0).sum())
            rows.append(
                {
                    "label_id": label_id,
                    "evaluation": evaluation,
                    "metric": metric,
                    "n_splits": len(delta),
                    "mean_delta": observed,
                    "median_delta": float(np.median(delta)),
                    "wins": wins,
                    "losses": int(len(delta) - wins - (delta == 0).sum()),
                    "ties": int((delta == 0).sum()),
                    "bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
                    "bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
                    "wilcoxon_p": wilcoxon_p,
                    "permutation_p": float(
                        ((np.abs(permuted) >= abs(observed)).sum() + 1)
                        / (permutation_repeats + 1)
                    ),
                }
            )
    return pd.DataFrame(rows)
