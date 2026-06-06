from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEEP_PREFIXES = {
    "region_cnn_quick_cpu": "parallel_deep_region_cnn",
    "sequence_transformer_quick_cpu": "parallel_deep_sequence_transformer",
    "saluki_like_quick_cpu": "parallel_deep_saluki_like",
}

DEEP_GPU_FULL_PREFIXES = {
    "region_cnn_gpu_full": "parallel_deep_gpu_full_region_cnn",
    "sequence_transformer_gpu_full": "parallel_deep_gpu_full_sequence_transformer",
    "saluki_like_gpu_full": "parallel_deep_gpu_full_saluki_like",
}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    processed_dir = root / "data/processed"
    figure_dir = root / "docs/figures"
    feature_tables = pd.read_csv(processed_dir / "parallel_label_feature_tables.tsv", sep="\t")
    metadata = feature_tables.set_index("label_id")[["feature_type", "label_key", "target_column", "feature_rows"]]

    rows = []
    rows.extend(load_full_elasticnet(processed_dir, metadata))
    rows.extend(load_compact_quick(processed_dir))
    rows.extend(load_deep_quick(processed_dir, metadata))
    rows.extend(load_deep_gpu_full(processed_dir, metadata))
    summary = pd.DataFrame(rows).sort_values(["evaluation", "label_id", "pearson_median"], ascending=[True, True, False])
    summary.to_csv(processed_dir / "parallel_model_suite_summary.tsv", sep="\t", index=False)
    write_figure(summary, figure_dir)
    print(summary.to_string(index=False))


def load_full_elasticnet(processed_dir: Path, metadata: pd.DataFrame) -> list[dict[str, object]]:
    path = processed_dir / "parallel_label_model_comparison.tsv"
    if not path.exists():
        return []
    frame = pd.read_csv(path, sep="\t")
    rows = []
    for item in frame.itertuples(index=False):
        rows.append(
            {
                "label_id": item.label_id,
                "feature_type": item.feature_type,
                "label_key": item.label_key,
                "target_column": item.target_column,
                "model": "elasticnet_full",
                "benchmark_type": "full",
                "evaluation": item.evaluation,
                "n_splits": int(item.n_splits),
                "feature_rows": int(item.feature_rows),
                "pearson_median": float(item.pearson_median),
                "spearman_median": float(item.spearman_median),
                "r2_median": float(item.r2_median),
                "rmse_median": float(item.rmse_median),
                "nrmse_median": float(item.nrmse_median),
            }
        )
    return rows


def load_compact_quick(processed_dir: Path) -> list[dict[str, object]]:
    path = processed_dir / "parallel_compact_model_benchmark_summary.tsv"
    if not path.exists():
        return []
    frame = pd.read_csv(path, sep="\t")
    rows = []
    for item in frame.itertuples(index=False):
        rows.append(
            {
                "label_id": item.label_id,
                "feature_type": item.feature_type,
                "label_key": item.label_key,
                "target_column": item.target_column,
                "model": item.model,
                "benchmark_type": "quick_compact",
                "evaluation": item.evaluation,
                "n_splits": int(item.n_splits),
                "feature_rows": pd.NA,
                "pearson_median": float(item.pearson_median),
                "spearman_median": float(item.spearman_median),
                "r2_median": float(item.r2_median),
                "rmse_median": float(item.rmse_median),
                "nrmse_median": pd.NA,
            }
        )
    return rows


def load_deep_quick(processed_dir: Path, metadata: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for label_id, meta in metadata.iterrows():
        for model_name, prefix in DEEP_PREFIXES.items():
            path = processed_dir / f"{prefix}_{label_id}_metrics.tsv"
            if not path.exists():
                continue
            metrics = pd.read_csv(path, sep="\t")
            for item in metrics.itertuples(index=False):
                rows.append(
                    {
                        "label_id": label_id,
                        "feature_type": meta.feature_type,
                        "label_key": meta.label_key,
                        "target_column": meta.target_column,
                        "model": model_name,
                        "benchmark_type": "quick_deep_cpu",
                        "evaluation": item.evaluation,
                        "n_splits": 1,
                        "feature_rows": int(meta.feature_rows),
                        "pearson_median": float(item.pearson),
                        "spearman_median": float(item.spearman),
                        "r2_median": float(item.r2),
                        "rmse_median": float(item.rmse),
                        "nrmse_median": pd.NA,
                    }
                )
    return rows


def load_deep_gpu_full(processed_dir: Path, metadata: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for label_id, meta in metadata.iterrows():
        for model_name, prefix in DEEP_GPU_FULL_PREFIXES.items():
            path = processed_dir / f"{prefix}_{label_id}_metrics.tsv"
            if not path.exists():
                continue
            metrics = pd.read_csv(path, sep="\t")
            for evaluation, group in metrics.groupby("evaluation"):
                rows.append(
                    {
                        "label_id": label_id,
                        "feature_type": meta.feature_type,
                        "label_key": meta.label_key,
                        "target_column": meta.target_column,
                        "model": model_name,
                        "benchmark_type": "full_deep_gpu",
                        "evaluation": evaluation,
                        "n_splits": len(group),
                        "feature_rows": int(meta.feature_rows),
                        "pearson_median": float(group["pearson"].median()),
                        "spearman_median": float(group["spearman"].median()),
                        "r2_median": float(group["r2"].median()),
                        "rmse_median": float(group["rmse"].median()),
                        "nrmse_median": pd.NA,
                    }
                )
    return rows


def write_figure(summary: pd.DataFrame, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    data = summary[
        (summary["evaluation"].isin(["repeated_random", "chromosome_holdout_quick"]))
        & (summary["benchmark_type"].isin(["quick_compact", "quick_deep_cpu", "full_deep_gpu"]))
    ].copy()
    label_order = [
        "gene_sense_late_chase_6h_2h",
        "gene_sense_total_chase_6h_0h",
        "exon_sense_late_chase_6h_2h",
        "exon_sense_total_chase_6h_0h",
    ]
    model_order = [
        "ridge",
        "random_forest_light",
        "xgboost_light",
        "region_cnn_quick_cpu",
        "sequence_transformer_quick_cpu",
        "saluki_like_quick_cpu",
        "region_cnn_gpu_full",
        "sequence_transformer_gpu_full",
        "saluki_like_gpu_full",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    for ax, label_id in zip(axes.flat, label_order):
        subset = data[(data["label_id"] == label_id) & (data["evaluation"] == "repeated_random")]
        values = subset.set_index("model").reindex(model_order)["pearson_median"]
        ax.bar(range(len(values)), values, color="#4477AA")
        ax.set_title(label_id)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(model_order, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Repeated-random Pearson")
        ax.set_ylim(0, max(0.85, float(data["pearson_median"].max()) + 0.05))
    fig.tight_layout()
    fig.savefig(figure_dir / "parallel_model_suite_repeated_random.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
