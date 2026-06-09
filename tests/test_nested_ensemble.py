from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rna_stability_elements.models.nested_ensemble import blend, paired_statistics, summarize_nested


def test_nested_ensemble_helpers_report_wins_and_summary():
    frame = pd.DataFrame(
        {
            "y_pred_ramht": [1.0, 2.0],
            "y_pred_xgboost": [0.0, 1.0],
        }
    )
    assert np.allclose(blend(frame, 0.25), [0.25, 1.25])

    metrics = pd.DataFrame(
        {
            "label_id": ["label"] * 3,
            "evaluation": ["repeated_random"] * 3,
            "split_name": ["a", "b", "c"],
            "selected_ramht_weight": [0.1, 0.2, 0.3],
            "residual_correlation": [0.2, 0.3, 0.4],
            **{
                f"xgboost_{metric}": [0.4, 0.5, 0.6]
                for metric in ["pearson", "spearman", "r2", "rmse", "mae"]
            },
            **{
                f"blend_{metric}": [0.5, 0.6, 0.7]
                for metric in ["pearson", "spearman", "r2"]
            },
            "blend_rmse": [0.3, 0.4, 0.5],
            "blend_mae": [0.3, 0.4, 0.5],
        }
    )
    for metric in ["pearson", "spearman", "r2", "rmse", "mae"]:
        metrics[f"delta_{metric}"] = metrics[f"blend_{metric}"] - metrics[f"xgboost_{metric}"]

    summary = summarize_nested(metrics)
    statistics = paired_statistics(
        metrics,
        bootstrap_repeats=100,
        permutation_repeats=100,
        random_state=1,
    )

    assert summary.iloc[0]["selected_ramht_weight_mean"] == pytest.approx(0.2)
    pearson = statistics[
        (statistics["evaluation"] == "repeated_random") & (statistics["metric"] == "pearson")
    ].iloc[0]
    rmse = statistics[
        (statistics["evaluation"] == "repeated_random") & (statistics["metric"] == "rmse")
    ].iloc[0]
    assert pearson["wins"] == 3
    assert rmse["wins"] == 3
