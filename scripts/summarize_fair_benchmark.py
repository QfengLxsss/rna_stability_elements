from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABEL_ORDER = [
    "gene_sense_late_chase_6h_2h",
    "gene_sense_total_chase_6h_0h",
    "exon_sense_late_chase_6h_2h",
    "exon_sense_total_chase_6h_0h",
]
LABEL_NAMES = {
    "gene_sense_late_chase_6h_2h": "gene sense\n6h / 2h",
    "gene_sense_total_chase_6h_0h": "gene sense\n6h / 0h",
    "exon_sense_late_chase_6h_2h": "exon sense\n6h / 2h",
    "exon_sense_total_chase_6h_0h": "exon sense\n6h / 0h",
}
MODEL_ORDER = [
    "elasticnet_full",
    "random_forest_full",
    "xgboost_full",
    "region_cnn",
    "sequence_transformer",
    "saluki_like",
]
MODEL_NAMES = {
    "elasticnet_full": "ElasticNet",
    "random_forest_full": "RandomForest",
    "xgboost_full": "XGBoost",
    "region_cnn": "Region-CNN",
    "sequence_transformer": "Transformer",
    "saluki_like": "Saluki-like",
}
MODEL_COLORS = {
    "elasticnet_full": "#8C8C8C",
    "random_forest_full": "#6F9E7C",
    "xgboost_full": "#2F6B49",
    "region_cnn": "#8BAED8",
    "sequence_transformer": "#4C78A8",
    "saluki_like": "#D88752",
}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    processed = root / "data/processed"
    figure_dir = root / "docs/figures"
    source_dir = processed / "figure_source_data"
    source_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(processed)
    summary = summarize_metrics(metrics)
    paired = paired_differences(metrics)
    costs = load_costs(processed)
    metrics.to_csv(processed / "fair_benchmark_all_metrics.tsv", sep="\t", index=False)
    summary.to_csv(processed / "fair_benchmark_summary.tsv", sep="\t", index=False)
    paired.to_csv(processed / "fair_benchmark_paired_differences.tsv", sep="\t", index=False)
    costs.to_csv(processed / "fair_benchmark_cost_summary.tsv", sep="\t", index=False)
    metrics.to_csv(source_dir / "fair_benchmark_all_metrics.tsv", sep="\t", index=False)
    costs.to_csv(source_dir / "fair_benchmark_cost_summary.tsv", sep="\t", index=False)

    setup_style()
    write_overview(metrics, costs, figure_dir)
    write_distributions(metrics, figure_dir)
    write_chromosome_heatmap(metrics, figure_dir)
    write_report(root, summary, paired, costs)
    print(summary.to_string(index=False))


def load_metrics(processed: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(processed.glob("fair_benchmark_classical_*_metrics.tsv")):
        frame = pd.read_csv(path, sep="\t")
        frame["model_family"] = "engineered_feature_model"
        frames.append(frame)
    metadata = pd.read_csv(processed / "fair_benchmark_cohort_summary.tsv", sep="\t")
    for item in metadata.itertuples(index=False):
        for model in ["region_cnn", "sequence_transformer", "saluki_like"]:
            path = processed / f"parallel_deep_gpu_full_{model}_{item.label_id}_metrics.tsv"
            frame = pd.read_csv(path, sep="\t")
            frame["label_id"] = item.label_id
            frame["model"] = model
            frame["input_representation"] = "raw_sequence"
            frame["model_family"] = "deep_sequence_model"
            frames.append(frame)
    metrics = pd.concat(frames, ignore_index=True, sort=False)
    return metrics


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in metrics.groupby(["label_id", "model", "evaluation"], sort=False):
        label_id, model, evaluation = keys
        row = {
            "label_id": label_id,
            "model": model,
            "model_display": MODEL_NAMES[model],
            "evaluation": evaluation,
            "n_splits": len(group),
        }
        for metric in ["pearson", "spearman", "r2", "rmse"]:
            values = group[metric].dropna().to_numpy()
            low, high = bootstrap_mean_ci(values)
            row.update(
                {
                    f"{metric}_mean": float(np.mean(values)),
                    f"{metric}_median": float(np.median(values)),
                    f"{metric}_std": float(np.std(values, ddof=0)),
                    f"{metric}_iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
                    f"{metric}_mean_ci_low": low,
                    f"{metric}_mean_ci_high": high,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["evaluation", "label_id", "pearson_median"], ascending=[True, True, False])


def paired_differences(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    reference = metrics[metrics["model"] == "xgboost_full"]
    for (label_id, evaluation), ref_group in reference.groupby(["label_id", "evaluation"]):
        ref = ref_group.set_index("split_name")["pearson"]
        for model in MODEL_ORDER:
            if model == "xgboost_full":
                continue
            test = metrics[
                (metrics["label_id"] == label_id)
                & (metrics["evaluation"] == evaluation)
                & (metrics["model"] == model)
            ].set_index("split_name")["pearson"]
            paired = pd.concat([ref.rename("reference"), test.rename("model")], axis=1).dropna()
            delta = paired["model"] - paired["reference"]
            low, high = bootstrap_mean_ci(delta.to_numpy())
            rows.append(
                {
                    "label_id": label_id,
                    "evaluation": evaluation,
                    "reference_model": "xgboost_full",
                    "model": model,
                    "n_paired_splits": len(delta),
                    "pearson_delta_mean": delta.mean(),
                    "pearson_delta_median": delta.median(),
                    "pearson_delta_std": delta.std(ddof=0),
                    "pearson_delta_mean_ci_low": low,
                    "pearson_delta_mean_ci_high": high,
                    "win_fraction_vs_xgboost": float((delta > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def load_costs(processed: Path) -> pd.DataFrame:
    classical = pd.concat(
        [pd.read_csv(path, sep="\t") for path in sorted(processed.glob("fair_benchmark_classical_*_metrics.tsv"))],
        ignore_index=True,
    )
    classical = classical[classical["split_name"] == "random_repeat_0"].copy()
    classical_cost = classical[
        [
            "label_id",
            "model",
            "fit_wall_seconds",
            "predict_wall_seconds",
            "peak_process_rss_mb",
            "serialized_model_mb",
            "compute_device",
        ]
    ].rename(columns={"fit_wall_seconds": "train_wall_seconds"})
    classical_cost["peak_gpu_memory_mb"] = np.nan
    classical_cost["epochs_trained"] = np.nan
    deep = pd.read_csv(processed / "fair_benchmark_deep_cost_summary.tsv", sep="\t")
    deep_cost = deep[
        ["label_id", "model", "wall_seconds", "peak_gpu_memory_mb", "epochs_trained", "device"]
    ].rename(columns={"wall_seconds": "train_wall_seconds", "device": "compute_device"})
    deep_cost["predict_wall_seconds"] = np.nan
    deep_cost["peak_process_rss_mb"] = np.nan
    deep_cost["serialized_model_mb"] = np.nan
    costs = pd.concat([classical_cost, deep_cost], ignore_index=True, sort=False)
    costs["cost_split"] = "random_repeat_0"
    return costs


def bootstrap_mean_ci(values: np.ndarray, n_bootstrap: int = 5000) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(13)
    samples = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, figure_dir: Path, name: str) -> None:
    for suffix, kwargs in [("png", {"dpi": 300}), ("svg", {}), ("pdf", {})]:
        fig.savefig(figure_dir / f"{name}.{suffix}", bbox_inches="tight", **kwargs)


def write_overview(metrics: pd.DataFrame, costs: pd.DataFrame, figure_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)
    for ax, evaluation, panel in zip(
        axes[:1].flat,
        ["repeated_random", "chromosome_holdout"],
        ["a", "b"],
    ):
        matrix = (
            metrics[metrics["evaluation"] == evaluation]
            .groupby(["model", "label_id"])["pearson"]
            .median()
            .unstack()
            .reindex(index=MODEL_ORDER, columns=LABEL_ORDER)
        )
        image = ax.imshow(matrix, cmap="YlGnBu", vmin=0.35, vmax=0.80, aspect="auto")
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                ax.text(col, row, f"{matrix.iloc[row, col]:.3f}", ha="center", va="center", fontsize=5.8)
        ax.set_xticks(range(4))
        ax.set_xticklabels([LABEL_NAMES[label] for label in LABEL_ORDER])
        ax.set_yticks(range(6))
        ax.set_yticklabels([MODEL_NAMES[model] for model in MODEL_ORDER])
        ax.set_title(evaluation.replace("_", " "))
        ax.text(-0.13, 1.07, panel, transform=ax.transAxes, fontweight="bold", fontsize=9)
        fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02, label="Median Pearson")

    ax = axes[1, 0]
    chromosome = metrics[metrics["evaluation"] == "chromosome_holdout"].copy()
    reference = chromosome[chromosome["model"] == "xgboost_full"].set_index(["label_id", "split_name"])["pearson"]
    positions = np.arange(len(MODEL_ORDER) - 1)
    for label_index, label in enumerate(LABEL_ORDER):
        deltas = []
        for model in [item for item in MODEL_ORDER if item != "xgboost_full"]:
            model_values = chromosome[
                (chromosome["label_id"] == label) & (chromosome["model"] == model)
            ].set_index(["label_id", "split_name"])["pearson"]
            deltas.append((model_values - reference).median())
        ax.plot(positions, deltas, marker="o", linewidth=1, label=LABEL_NAMES[label].replace("\n", " "))
    ax.axhline(0, color="#666666", linestyle="--", linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels([MODEL_NAMES[item] for item in MODEL_ORDER if item != "xgboost_full"], rotation=30, ha="right")
    ax.set_ylabel("Median chromosome Pearson delta vs XGBoost")
    ax.set_title("Paired generalization difference")
    ax.legend(fontsize=5.5)
    ax.text(-0.13, 1.07, "c", transform=ax.transAxes, fontweight="bold", fontsize=9)

    ax = axes[1, 1]
    performance = (
        metrics[metrics["evaluation"] == "repeated_random"].groupby("model")["pearson"].median()
    )
    cost = costs.groupby("model")["train_wall_seconds"].median()
    for model in MODEL_ORDER:
        ax.scatter(cost[model], performance[model], s=38, color=MODEL_COLORS[model])
        ax.text(cost[model] * 1.05, performance[model], MODEL_NAMES[model], fontsize=5.8, va="center")
    ax.set_xscale("log")
    ax.set_xlabel("Median train wall time on random_repeat_0 (s, log scale)")
    ax.set_ylabel("Median repeated-random Pearson across labels")
    ax.set_title("Performance-cost trade-off")
    ax.text(-0.13, 1.07, "d", transform=ax.transAxes, fontweight="bold", fontsize=9)
    save_figure(fig, figure_dir, "fair_benchmark_overview")
    plt.close(fig)


def write_distributions(metrics: pd.DataFrame, figure_dir: Path) -> None:
    data = metrics[metrics["evaluation"] == "chromosome_holdout"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharey=True, constrained_layout=True)
    for ax, label_id, panel in zip(axes.flat, LABEL_ORDER, list("abcd")):
        subset = data[data["label_id"] == label_id]
        values = [subset[subset["model"] == model]["pearson"].to_numpy() for model in MODEL_ORDER]
        box = ax.boxplot(values, patch_artist=True, showfliers=False, widths=0.65)
        for patch, model in zip(box["boxes"], MODEL_ORDER):
            patch.set_facecolor(MODEL_COLORS[model])
            patch.set_alpha(0.85)
        ax.set_xticks(range(1, 7))
        ax.set_xticklabels([MODEL_NAMES[model] for model in MODEL_ORDER], rotation=35, ha="right")
        ax.set_title(LABEL_NAMES[label_id].replace("\n", " "))
        ax.set_ylabel("Chromosome-holdout Pearson")
        ax.text(-0.13, 1.07, panel, transform=ax.transAxes, fontweight="bold", fontsize=9)
    save_figure(fig, figure_dir, "fair_benchmark_split_distributions")
    plt.close(fig)


def write_chromosome_heatmap(metrics: pd.DataFrame, figure_dir: Path) -> None:
    data = metrics[metrics["evaluation"] == "chromosome_holdout"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.1), constrained_layout=True)
    image = None
    for ax, label_id, panel in zip(axes.flat, LABEL_ORDER, list("abcd")):
        matrix = data[data["label_id"] == label_id].pivot(
            index="model", columns="holdout_group", values="pearson"
        ).reindex(index=MODEL_ORDER)
        image = ax.imshow(matrix, cmap="YlGnBu", vmin=0.2, vmax=0.85, aspect="auto")
        ax.set_yticks(range(6))
        ax.set_yticklabels([MODEL_NAMES[model] for model in MODEL_ORDER])
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels(matrix.columns, rotation=90, fontsize=5)
        ax.set_title(LABEL_NAMES[label_id].replace("\n", " "))
        ax.text(-0.13, 1.07, panel, transform=ax.transAxes, fontweight="bold", fontsize=9)
    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02, label="Pearson")
    save_figure(fig, figure_dir, "fair_benchmark_chromosome_heatmap")
    plt.close(fig)


def write_report(root: Path, summary: pd.DataFrame, paired: pd.DataFrame, costs: pd.DataFrame) -> None:
    repeated = summary[summary["evaluation"] == "repeated_random"]
    chromosome = summary[summary["evaluation"] == "chromosome_holdout"]
    lines = [
        "# Fair Model Benchmark",
        "",
        "All six models use the same four label-specific cohorts and the same fixed train, validation, and test gene assignments.",
        "",
        "## Design",
        "",
        "- Four shared cohorts: engineered-feature and raw-sequence tables have identical gene order.",
        "- Fixed splits per label: 3 repeated-random and 23 chromosome-holdout splits.",
        "- Existing GPU-full deep results were reused only after exact per-split test-gene audit.",
        "- Classical models train only on the manifest `train` role; validation and test genes are excluded.",
        "",
        "![Fair benchmark overview](figures/fair_benchmark_overview.png)",
        "",
        "## Best Models by Label",
        "",
        "| Label | Repeated-random best | Pearson | Chromosome-holdout best | Pearson |",
        "| --- | --- | ---: | --- | ---: |",
    ]
    for label in LABEL_ORDER:
        rr = repeated[repeated["label_id"] == label].sort_values("pearson_median", ascending=False).iloc[0]
        ch = chromosome[chromosome["label_id"] == label].sort_values("pearson_median", ascending=False).iloc[0]
        lines.append(
            f"| `{label}` | {rr.model_display} | {rr.pearson_median:.3f} | "
            f"{ch.model_display} | {ch.pearson_median:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Main Conclusions",
            "",
            "1. Full XGBoost leads all four chromosome-holdout tasks and three of four repeated-random tasks; RandomForest narrowly leads the repeated-random median for exon_sense 6h/0h.",
            "2. The label definition remains a larger source of performance variation than model architecture.",
            "3. Deep raw-sequence models are competitive, but do not exceed engineered-feature XGBoost in the current configuration.",
            "4. Chromosome-holdout distributions show that model rankings are generally stable, while absolute difficulty varies by chromosome.",
            "5. The controlled cost benchmark shows that XGBoost is also substantially faster than the current full deep models.",
            "",
            "## Cost Benchmark",
            "",
            "Costs were measured on the fixed `random_repeat_0` split. Deep-model cost runs reproduce the same test genes as the manifest.",
            "",
            "| Model | Median train wall time (s) | Median peak GPU memory (MB) |",
            "| --- | ---: | ---: |",
        ]
    )
    for model in MODEL_ORDER:
        group = costs[costs["model"] == model]
        peak = group["peak_gpu_memory_mb"].median()
        peak_text = "NA" if pd.isna(peak) else f"{peak:.0f}"
        lines.append(f"| {MODEL_NAMES[model]} | {group.train_wall_seconds.median():.1f} | {peak_text} |")
    lines.extend(
        [
            "",
            "## Statistical Outputs",
            "",
            "- `data/processed/fair_benchmark_summary.tsv`: mean, median, standard deviation, IQR, and bootstrap mean CI.",
            "- `data/processed/fair_benchmark_paired_differences.tsv`: paired Pearson differences and win fractions versus XGBoost.",
            "- `data/processed/fair_benchmark_cost_summary.tsv`: controlled computation-cost measurements.",
            "- `data/processed/fair_benchmark_deep_reuse_audit.tsv`: exact split-level deep-result reuse audit.",
            "",
            "## Figures",
            "",
            "- `docs/figures/fair_benchmark_overview.{png,svg,pdf}`",
            "- `docs/figures/fair_benchmark_split_distributions.{png,svg,pdf}`",
            "- `docs/figures/fair_benchmark_chromosome_heatmap.{png,svg,pdf}`",
        ]
    )
    (root / "docs/fair_benchmark_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
