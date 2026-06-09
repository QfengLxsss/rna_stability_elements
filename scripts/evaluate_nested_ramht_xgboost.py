from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from rna_stability_elements.models.baselines import regression_metrics
from rna_stability_elements.models.nested_ensemble import (
    METRICS,
    blend,
    paired_statistics,
    summarize_nested,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nested validation selection for RAMHT/XGBoost blends.")
    parser.add_argument("--ramht-predictions", required=True)
    parser.add_argument(
        "--xgboost-predictions",
        default="data/processed/nested_ensemble_xgboost_predictions.tsv",
    )
    parser.add_argument("--out-prefix", default="data/processed/nested_ramht_xgboost")
    parser.add_argument("--weights", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--permutation-repeats", type=int, default=100000)
    parser.add_argument("--random-state", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = [float(item) for item in args.weights.split(",")]
    ramht = pd.read_csv(args.ramht_predictions, sep="\t", low_memory=False)
    xgboost = pd.read_csv(args.xgboost_predictions, sep="\t", low_memory=False)
    required = {"gene_id", "label_id", "split_name", "prediction_role", "y_true", "y_pred"}
    for name, frame in [("RAMHT", ramht), ("XGBoost", xgboost)]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} predictions missing columns: {sorted(missing)}")

    paired = xgboost.merge(
        ramht[["gene_id", "label_id", "split_name", "prediction_role", "y_true", "y_pred"]],
        on=["gene_id", "label_id", "split_name", "prediction_role"],
        how="inner",
        suffixes=("_xgboost", "_ramht"),
        validate="one_to_one",
    )
    if not np.allclose(paired["y_true_xgboost"], paired["y_true_ramht"], equal_nan=True):
        raise ValueError("RAMHT and XGBoost y_true values do not match.")
    paired["y_true"] = paired["y_true_xgboost"]

    selection_rows = []
    metric_rows = []
    test_prediction_frames = []
    for (label_id, split_name), group in paired.groupby(["label_id", "split_name"], sort=False):
        validation = group[group["prediction_role"] == "validation"].copy()
        test = group[group["prediction_role"] == "test"].copy()
        if validation.empty or test.empty:
            raise ValueError(f"Missing validation or test predictions for {label_id} / {split_name}")
        candidates = []
        for weight in weights:
            prediction = blend(validation, weight)
            metrics = regression_metrics(validation["y_true"], prediction)
            candidates.append((metrics["pearson"], -weight, weight, metrics))
        _score, _tie_break, selected_weight, selected_validation_metrics = max(candidates)
        selection_rows.append(
            {
                "label_id": label_id,
                "evaluation": test["evaluation"].iloc[0],
                "split_name": split_name,
                "selected_ramht_weight": selected_weight,
                **{f"validation_{key}": value for key, value in selected_validation_metrics.items()},
            }
        )
        xgboost_metrics = regression_metrics(test["y_true"], test["y_pred_xgboost"])
        blend_prediction = blend(test, selected_weight)
        blend_metrics = regression_metrics(test["y_true"], blend_prediction)
        residual_corr = pearsonr(
            test["y_pred_ramht"] - test["y_true"],
            test["y_pred_xgboost"] - test["y_true"],
        ).statistic
        metric_rows.append(
            {
                "label_id": label_id,
                "evaluation": test["evaluation"].iloc[0],
                "split_name": split_name,
                "selected_ramht_weight": selected_weight,
                "residual_correlation": residual_corr,
                **{f"xgboost_{key}": value for key, value in xgboost_metrics.items()},
                **{f"blend_{key}": value for key, value in blend_metrics.items()},
                **{f"delta_{key}": blend_metrics[key] - xgboost_metrics[key] for key in METRICS},
            }
        )
        test_predictions = test.copy()
        test_predictions["selected_ramht_weight"] = selected_weight
        test_predictions["y_pred_blend"] = blend_prediction
        test_predictions["residual_blend"] = blend_prediction - test_predictions["y_true"]
        test_prediction_frames.append(test_predictions)

    selections = pd.DataFrame(selection_rows)
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(test_prediction_frames, ignore_index=True)
    summary = summarize_nested(metrics)
    statistics = paired_statistics(
        metrics,
        bootstrap_repeats=args.bootstrap_repeats,
        permutation_repeats=args.permutation_repeats,
        random_state=args.random_state,
    )
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    selections.to_csv(f"{prefix}_weight_selections.tsv", sep="\t", index=False)
    metrics.to_csv(f"{prefix}_metrics.tsv", sep="\t", index=False)
    predictions.to_csv(f"{prefix}_predictions.tsv", sep="\t", index=False)
    summary.to_csv(f"{prefix}_summary.tsv", sep="\t", index=False)
    statistics.to_csv(f"{prefix}_statistics.tsv", sep="\t", index=False)
    print(summary.to_string(index=False))
    print(statistics.to_string(index=False))
    print(f"[done] wrote {prefix}_*.tsv")


if __name__ == "__main__":
    main()
