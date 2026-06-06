from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    processed_dir = root / "data/processed"
    figure_dir = root / "docs/figures"
    feature_tables = pd.read_csv(processed_dir / "parallel_label_feature_tables.tsv", sep="\t")

    comparison = build_model_comparison(processed_dir, feature_tables)
    comparison.to_csv(processed_dir / "parallel_label_model_comparison.tsv", sep="\t", index=False)

    overlap = build_importance_overlap(processed_dir, feature_tables["label_id"].tolist())
    overlap.to_csv(processed_dir / "parallel_label_importance_overlap.tsv", sep="\t", index=False)

    write_model_comparison_figure(comparison, figure_dir)
    print(comparison.to_string(index=False))


def build_model_comparison(processed_dir: Path, feature_tables: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in feature_tables.itertuples(index=False):
        label_id = item.label_id
        features = pd.read_csv(item.feature_path, sep="\t", usecols=["target_label"])
        summary_path = processed_dir / f"parallel_eval_{label_id}_elasticnet_summary.tsv"
        metrics_path = processed_dir / f"parallel_eval_{label_id}_elasticnet_metrics.tsv"
        summary = pd.read_csv(summary_path, sep="\t")
        metrics = pd.read_csv(metrics_path, sep="\t")
        target_std = float(features["target_label"].std())
        target_iqr = float(features["target_label"].quantile(0.75) - features["target_label"].quantile(0.25))
        for row in summary.itertuples(index=False):
            metric_subset = metrics[metrics["evaluation"] == row.evaluation]
            rows.append(
                {
                    "label_id": label_id,
                    "feature_type": item.feature_type,
                    "label_key": item.label_key,
                    "target_column": item.target_column,
                    "evaluation": row.evaluation,
                    "model": row.model,
                    "n_splits": int(row.n_splits),
                    "feature_rows": int(item.feature_rows),
                    "target_std": target_std,
                    "target_iqr": target_iqr,
                    "pearson_median": float(row.pearson_median),
                    "spearman_median": float(row.spearman_median),
                    "r2_median": float(row.r2_median),
                    "rmse_median": float(row.rmse_median),
                    "nrmse_median": float(row.rmse_median / target_std),
                    "pearson_mean": float(row.pearson_mean),
                    "spearman_mean": float(row.spearman_mean),
                    "r2_mean": float(row.r2_mean),
                    "rmse_mean": float(row.rmse_mean),
                    "pearson_split_min": float(metric_subset["pearson"].min()),
                    "pearson_split_max": float(metric_subset["pearson"].max()),
                }
            )
    return pd.DataFrame(rows).sort_values(["evaluation", "label_id"]).reset_index(drop=True)


def build_importance_overlap(processed_dir: Path, label_ids: list[str]) -> pd.DataFrame:
    importance = {}
    for label_id in label_ids:
        path = processed_dir / f"parallel_eval_{label_id}_elasticnet_importance.tsv"
        frame = pd.read_csv(path, sep="\t")
        importance[label_id] = frame.groupby("feature", as_index=False).agg(
            coefficient=("importance", "mean"),
            abs_importance=("importance_abs", "mean"),
        )

    rows = []
    for left, right in combinations(label_ids, 2):
        merged = importance[left].merge(importance[right], on="feature", suffixes=("_left", "_right"))
        signed_corr = float(merged["coefficient_left"].corr(merged["coefficient_right"], method="pearson"))
        abs_rank_corr = float(merged["abs_importance_left"].corr(merged["abs_importance_right"], method="spearman"))
        for top_k in [25, 50, 100, 200]:
            left_top = set(merged.nlargest(top_k, "abs_importance_left")["feature"])
            right_top = set(merged.nlargest(top_k, "abs_importance_right")["feature"])
            intersection = len(left_top & right_top)
            rows.append(
                {
                    "left_label_id": left,
                    "right_label_id": right,
                    "top_k": top_k,
                    "signed_coefficient_pearson": signed_corr,
                    "abs_importance_spearman": abs_rank_corr,
                    "topk_overlap": intersection,
                    "topk_jaccard": intersection / len(left_top | right_top),
                }
            )
    return pd.DataFrame(rows)


def write_model_comparison_figure(comparison: pd.DataFrame, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    order = [
        "gene_sense_late_chase_6h_2h",
        "gene_sense_total_chase_6h_0h",
        "exon_sense_late_chase_6h_2h",
        "exon_sense_total_chase_6h_0h",
    ]
    colors = ["#4477AA", "#CC6677", "#66AA77", "#AA7744"]
    color_map = dict(zip(order, colors))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, evaluation in zip(axes, ["repeated_random", "chromosome_holdout"]):
        data = comparison[comparison["evaluation"] == evaluation].set_index("label_id").reindex(order)
        ax.bar(range(len(data)), data["pearson_median"], color=[color_map[label] for label in data.index])
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(data.index, rotation=25, ha="right")
        ax.set_ylim(0, max(0.65, float(comparison["pearson_median"].max()) + 0.05))
        ax.set_title(evaluation)
        ax.set_ylabel("Pearson median")
    fig.tight_layout()
    fig.savefig(figure_dir / "parallel_label_model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
