from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rna_stability_elements.models.baselines import regression_metrics
from rna_stability_elements.models.ramht import LABEL_IDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fixed-weight RAMHT/XGBoost prediction blends.")
    parser.add_argument(
        "--ramht-predictions",
        default="data/processed/ramht_2080ti_parallel_predictions.tsv",
    )
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--out-prefix", default="data/processed/ramht_xgboost_blend")
    parser.add_argument(
        "--ramht-weights",
        default="0,0.1,0.2,0.3,0.4,0.5",
        help="Comma-separated fixed RAMHT weights; XGBoost receives 1-weight.",
    )
    parser.add_argument(
        "--selected-ramht-weight",
        type=float,
        default=0.2,
        help="Fixed blend weight to also write as the selected ensemble result.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed = Path(args.processed_dir)
    weights = [float(value) for value in args.ramht_weights.split(",")]
    ramht = pd.read_csv(args.ramht_predictions, sep="\t")
    metric_rows = []
    prediction_frames = []

    for label_id in LABEL_IDS:
        classical = pd.read_csv(
            processed / f"fair_benchmark_classical_{label_id}_predictions.tsv",
            sep="\t",
        )
        xgboost = classical[classical["model"] == "xgboost_full"].copy()
        ramht_label = ramht[ramht["label_id"] == label_id].copy()
        paired = xgboost.merge(
            ramht_label[["gene_id", "split_name", "y_pred"]],
            on=["gene_id", "split_name"],
            how="inner",
            suffixes=("_xgboost", "_ramht"),
            validate="one_to_one",
        )
        for ramht_weight in weights:
            blended = paired.copy()
            blended["y_pred"] = (
                ramht_weight * blended["y_pred_ramht"]
                + (1.0 - ramht_weight) * blended["y_pred_xgboost"]
            )
            blended["residual"] = blended["y_pred"] - blended["y_true"]
            blended["ramht_weight"] = ramht_weight
            blended["xgboost_weight"] = 1.0 - ramht_weight
            blended["model"] = "ramht_xgboost_fixed_blend"
            blended["label_id"] = label_id
            prediction_frames.append(blended)
            for (evaluation, split_name), group in blended.groupby(
                ["evaluation", "split_name"], sort=False
            ):
                row = regression_metrics(group["y_true"], group["y_pred"])
                row.update(
                    {
                        "label_id": label_id,
                        "evaluation": evaluation,
                        "split_name": split_name,
                        "model": "ramht_xgboost_fixed_blend",
                        "ramht_weight": ramht_weight,
                        "xgboost_weight": 1.0 - ramht_weight,
                        "n_test": len(group),
                    }
                )
                metric_rows.append(row)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = (
        metrics.groupby(["label_id", "evaluation", "ramht_weight", "xgboost_weight"], as_index=False)
        .agg(
            n_splits=("split_name", "nunique"),
            pearson_mean=("pearson", "mean"),
            spearman_mean=("spearman", "mean"),
            r2_mean=("r2", "mean"),
            rmse_mean=("rmse", "mean"),
            mae_mean=("mae", "mean"),
        )
    )
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(f"{prefix}_metrics.tsv", sep="\t", index=False)
    predictions.to_csv(f"{prefix}_predictions.tsv", sep="\t", index=False)
    summary.to_csv(f"{prefix}_summary.tsv", sep="\t", index=False)
    selected_summary = summary[summary["ramht_weight"] == args.selected_ramht_weight]
    selected_predictions = predictions[predictions["ramht_weight"] == args.selected_ramht_weight]
    selected_summary.to_csv(f"{prefix}_selected_summary.tsv", sep="\t", index=False)
    selected_predictions.to_csv(f"{prefix}_selected_predictions.tsv", sep="\t", index=False)
    print(summary.to_string(index=False))
    print(f"[done] wrote {prefix}_*.tsv")


if __name__ == "__main__":
    main()
