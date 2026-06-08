from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABEL_ORDER = [
    "gene_sense_late_chase_6h_2h",
    "gene_sense_total_chase_6h_0h",
    "exon_sense_late_chase_6h_2h",
    "exon_sense_total_chase_6h_0h",
]
LABEL_NAMES = ["gene 6h/2h", "gene 6h/0h", "exon 6h/2h", "exon 6h/0h"]
MODEL_ORDER = ["region_cnn", "sequence_transformer", "saluki_like"]
CONDITION_ORDER = [
    "raw_all",
    "raw_5utr_only",
    "raw_cds_only",
    "raw_3utr_only",
    "raw_no_5utr",
    "raw_no_cds",
    "raw_no_3utr",
    "raw_all_plus_engineered",
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    processed = root / "data/processed"
    metrics = load_metrics(processed)
    validate_complete(metrics)
    summary = summarize(metrics)
    paired = paired_differences(metrics)
    summary.to_csv(processed / "deep_input_ablation_summary.tsv", sep="\t", index=False)
    paired.to_csv(processed / "deep_input_ablation_paired_differences.tsv", sep="\t", index=False)
    source = processed / "figure_source_data"
    summary.to_csv(source / "deep_input_ablation_summary.tsv", sep="\t", index=False)
    paired.to_csv(source / "deep_input_ablation_paired_differences.tsv", sep="\t", index=False)
    make_figures(root, summary, paired)
    write_report(root, summary, paired)


def load_metrics(processed: Path) -> pd.DataFrame:
    frames = []
    for label in LABEL_ORDER:
        for model in MODEL_ORDER:
            path = processed / f"parallel_deep_gpu_full_{model}_{label}_metrics.tsv"
            frame = pd.read_csv(path, sep="\t")
            frame["label_id"] = label
            frame["model_id"] = model
            frame["input_condition"] = "raw_all"
            frames.append(frame)
    for path in sorted(processed.glob("deep_input_ablation_gpu_full_*_metrics.tsv")):
        frames.append(pd.read_csv(path, sep="\t"))
    return pd.concat(frames, ignore_index=True)


def validate_complete(metrics: pd.DataFrame) -> None:
    counts = metrics.groupby(["label_id", "model_id", "input_condition"])["split_name"].nunique()
    expected = len(LABEL_ORDER) * len(MODEL_ORDER) * len(CONDITION_ORDER)
    if len(counts) != expected or not (counts == 26).all():
        raise ValueError(f"Incomplete deep input-ablation results:\n{counts}")
    new = metrics[metrics["input_condition"] != "raw_all"]
    if not new["manifest_exact_match"].all():
        raise ValueError("One or more new deep input-ablation splits failed manifest audit.")


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in metrics.groupby(
        ["label_id", "model_id", "input_condition", "evaluation"], sort=False
    ):
        row = dict(zip(["label_id", "model_id", "input_condition", "evaluation"], keys))
        row["n_splits"] = len(group)
        row["n_tabular_features"] = int(group["n_tabular_features"].iloc[0])
        row["epochs_median"] = group["epochs_trained"].median()
        for metric in ["pearson", "spearman", "r2", "rmse"]:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = np.mean(values)
            row[f"{metric}_median"] = np.median(values)
            row[f"{metric}_std"] = np.std(values, ddof=1) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def paired_differences(metrics: pd.DataFrame) -> pd.DataFrame:
    reference = metrics[metrics["input_condition"] == "raw_all"][
        ["label_id", "model_id", "evaluation", "split_name", "pearson"]
    ].rename(columns={"pearson": "reference_pearson"})
    joined = metrics.merge(
        reference, on=["label_id", "model_id", "evaluation", "split_name"], how="inner"
    )
    joined["delta"] = joined["pearson"] - joined["reference_pearson"]
    rng = np.random.default_rng(13)
    rows = []
    for keys, group in joined.groupby(
        ["label_id", "model_id", "input_condition", "evaluation"], sort=False
    ):
        delta = group["delta"].to_numpy()
        bootstrap = np.mean(rng.choice(delta, size=(5000, len(delta)), replace=True), axis=1)
        rows.append(
            {
                "label_id": keys[0],
                "model_id": keys[1],
                "input_condition": keys[2],
                "evaluation": keys[3],
                "n_paired_splits": len(delta),
                "pearson_delta_mean": np.mean(delta),
                "pearson_delta_median": np.median(delta),
                "pearson_delta_std": np.std(delta, ddof=1) if len(delta) > 1 else 0.0,
                "pearson_delta_mean_ci_low": np.quantile(bootstrap, 0.025),
                "pearson_delta_mean_ci_high": np.quantile(bootstrap, 0.975),
                "win_fraction_vs_raw_all": np.mean(delta > 0),
            }
        )
    return pd.DataFrame(rows)


def make_figures(root: Path, summary: pd.DataFrame, paired: pd.DataFrame) -> None:
    figure_dir = root / "docs/figures"
    chromosome = summary[summary["evaluation"] == "chromosome_holdout"]
    fig, axes = plt.subplots(3, 4, figsize=(13, 9), sharex=True, constrained_layout=True)
    for row, model in enumerate(MODEL_ORDER):
        for col, label in enumerate(LABEL_ORDER):
            ax = axes[row, col]
            data = (
                chromosome[(chromosome["model_id"] == model) & (chromosome["label_id"] == label)]
                .set_index("input_condition")
                .reindex(CONDITION_ORDER)
            )
            ax.barh(range(len(data)), data["pearson_median"], color="#347f8d")
            ax.invert_yaxis()
            ax.set_xlim(0, 0.9)
            if row == 0:
                ax.set_title(LABEL_NAMES[col])
            if col == 0:
                ax.set_ylabel(model.replace("_", " "))
                ax.set_yticks(range(len(CONDITION_ORDER)))
                ax.set_yticklabels(CONDITION_ORDER)
            else:
                ax.set_yticks([])
            if row == 2:
                ax.set_xlabel("Median Pearson")
    save_figure(fig, figure_dir, "deep_input_ablation_chromosome_holdout")

    data = paired[
        (paired["evaluation"] == "chromosome_holdout")
        & (paired["input_condition"] != "raw_all")
    ]
    table = (
        data.groupby(["model_id", "input_condition"])["pearson_delta_mean"]
        .mean()
        .unstack("model_id")
        .reindex(index=CONDITION_ORDER[1:], columns=MODEL_ORDER)
    )
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    image = ax.imshow(table, aspect="auto", cmap="RdBu_r", vmin=-0.12, vmax=0.12)
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels([item.replace("_", " ") for item in MODEL_ORDER])
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels(table.index)
    ax.set_title("Mean chromosome-holdout Pearson delta versus raw all")
    for row in range(len(table)):
        for col in range(len(MODEL_ORDER)):
            value = table.iloc[row, col]
            ax.text(col, row, f"{value:+.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, shrink=0.8)
    save_figure(fig, figure_dir, "deep_input_ablation_paired_differences")


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    for suffix in ["png", "svg", "pdf"]:
        fig.savefig(directory / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(root: Path, summary: pd.DataFrame, paired: pd.DataFrame) -> None:
    chromosome = summary[summary["evaluation"] == "chromosome_holdout"]
    chromosome_paired = paired[paired["evaluation"] == "chromosome_holdout"]
    condition_effects = (
        chromosome_paired[chromosome_paired["input_condition"] != "raw_all"]
        .groupby("input_condition")
        .agg(
            paired_delta_mean=("pearson_delta_mean", "mean"),
            paired_delta_min=("pearson_delta_mean", "min"),
            paired_delta_max=("pearson_delta_mean", "max"),
            mean_win_fraction=("win_fraction_vs_raw_all", "mean"),
            positive_combinations=("pearson_delta_mean", lambda values: int((values > 0).sum())),
            n_combinations=("pearson_delta_mean", "size"),
        )
        .reindex(CONDITION_ORDER[1:])
    )
    hybrid = condition_effects.loc["raw_all_plus_engineered"]
    no_cds = condition_effects.loc["raw_no_cds"]
    cds_only = condition_effects.loc["raw_cds_only"]
    no_5utr = condition_effects.loc["raw_no_5utr"]
    no_3utr = condition_effects.loc["raw_no_3utr"]
    lines = [
        "# Deep Raw-Sequence Region Ablation and Hybrid Benchmark",
        "",
        "All conditions use the fixed fair-benchmark splits and unchanged model architectures.",
        "The benchmark contains 2,184 new GPU trainings plus 312 reused raw-all trainings.",
        "",
        "![Deep input ablation](figures/deep_input_ablation_chromosome_holdout.png)",
        "",
        "## Main Findings",
        "",
        f"- Hybrid input improves all {int(hybrid.positive_combinations)}/"
        f"{int(hybrid.n_combinations)} model-label combinations, with a mean paired "
        f"chromosome-holdout Pearson gain of {hybrid.paired_delta_mean:+.3f}.",
        f"- Removing CDS has the largest leave-one-region-out penalty "
        f"({no_cds.paired_delta_mean:+.3f}), while CDS-only retains most raw-all performance "
        f"({cds_only.paired_delta_mean:+.3f}).",
        f"- Removing 5'UTR has a smaller average effect ({no_5utr.paired_delta_mean:+.3f}) "
        f"than removing 3'UTR ({no_3utr.paired_delta_mean:+.3f}).",
        "",
        "## Cross-Architecture Condition Effects",
        "",
        "| Condition | Mean paired delta | Min to max delta | Mean split win fraction | Positive combinations |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for condition, row in condition_effects.iterrows():
        lines.append(
            f"| `{condition}` | {row.paired_delta_mean:+.3f} | "
            f"{row.paired_delta_min:+.3f} to {row.paired_delta_max:+.3f} | "
            f"{row.mean_win_fraction:.2f} | "
            f"{int(row.positive_combinations)}/{int(row.n_combinations)} |"
        )
    lines.extend(
        [
            "",
            "![Paired deep input differences](figures/deep_input_ablation_paired_differences.png)",
            "",
        "## Hybrid Effect by Model and Label",
        "",
        "| Label | Model | Raw-all median Pearson | Hybrid median Pearson | Mean paired delta |",
        "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for label in LABEL_ORDER:
        for model in MODEL_ORDER:
            raw = chromosome[
                (chromosome["label_id"] == label)
                & (chromosome["model_id"] == model)
                & (chromosome["input_condition"] == "raw_all")
            ].iloc[0]
            hybrid = chromosome[
                (chromosome["label_id"] == label)
                & (chromosome["model_id"] == model)
                & (chromosome["input_condition"] == "raw_all_plus_engineered")
            ].iloc[0]
            delta = paired[
                (paired["label_id"] == label)
                & (paired["model_id"] == model)
                & (paired["input_condition"] == "raw_all_plus_engineered")
                & (paired["evaluation"] == "chromosome_holdout")
            ].iloc[0]
            lines.append(
                f"| `{label}` | `{model}` | {raw.pearson_median:.3f} | "
                f"{hybrid.pearson_median:.3f} | {delta.pearson_delta_mean:+.3f} |"
            )
    lines.extend(
        [
            "",
            "## Design",
            "",
            "- Removed regions are replaced with empty sequences while original model windows and architecture remain unchanged.",
            "- Existing raw-all GPU-full results are reused; every new split is audited against the fixed manifest.",
            "- Hybrid conditions add all 1,336 engineered sequence features, fitted and standardized using training genes only.",
            "- Summary bars use the median across splits; paired deltas use split-matched mean differences versus raw-all.",
            "- Per-model-label paired tables include bootstrap 95% confidence intervals and win fractions.",
            "",
            "## Outputs",
            "",
            "- `data/processed/deep_input_ablation_summary.tsv`",
            "- `data/processed/deep_input_ablation_paired_differences.tsv`",
            "- `docs/figures/deep_input_ablation_chromosome_holdout.{png,svg,pdf}`",
            "- `docs/figures/deep_input_ablation_paired_differences.{png,svg,pdf}`",
        ]
    )
    (root / "docs/deep_input_ablation_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
