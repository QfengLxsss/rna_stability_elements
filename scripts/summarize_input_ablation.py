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
LABEL_NAMES = {
    "gene_sense_late_chase_6h_2h": "gene 6h/2h",
    "gene_sense_total_chase_6h_0h": "gene 6h/0h",
    "exon_sense_late_chase_6h_2h": "exon 6h/2h",
    "exon_sense_total_chase_6h_0h": "exon 6h/0h",
}
FEATURE_ORDER = [
    "all_regions",
    "full_only",
    "structured_regions",
    "5utr_only",
    "cds_only",
    "3utr_only",
    "utr_only",
    "structured_no_5utr",
    "structured_no_cds",
    "structured_no_3utr",
    "simple_length_composition",
    "length_only",
    "composition_only",
    "motif_only",
    "kmer3_only",
    "kmer4_only",
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    processed = root / "data/processed"
    paths = sorted(processed.glob("input_ablation_xgboost_*_metrics.tsv"))
    if not paths:
        raise FileNotFoundError("No input-ablation metric files found.")
    metrics = pd.concat([pd.read_csv(path, sep="\t") for path in paths], ignore_index=True)
    validate_complete(metrics)
    summary = summarize(metrics)
    paired = paired_differences(metrics)
    summary.to_csv(processed / "input_ablation_summary.tsv", sep="\t", index=False)
    paired.to_csv(processed / "input_ablation_paired_differences.tsv", sep="\t", index=False)
    source = processed / "figure_source_data"
    source.mkdir(parents=True, exist_ok=True)
    summary.to_csv(source / "input_ablation_summary.tsv", sep="\t", index=False)
    paired.to_csv(source / "input_ablation_paired_differences.tsv", sep="\t", index=False)
    make_figures(root, summary, paired)
    write_report(root, summary, paired)
    print(summary.to_string(index=False))


def validate_complete(metrics: pd.DataFrame) -> None:
    expected = {(label, feature) for label in LABEL_ORDER for feature in FEATURE_ORDER}
    observed = set(zip(metrics["label_id"], metrics["feature_set"]))
    missing = sorted(expected - observed)
    if missing:
        raise ValueError(f"Missing label/feature-set runs: {missing}")
    counts = metrics.groupby(["label_id", "feature_set"])["split_name"].nunique()
    if not (counts == 26).all():
        raise ValueError(f"Expected 26 splits per label/feature set:\n{counts[counts != 26]}")


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in metrics.groupby(["label_id", "feature_set", "evaluation"], sort=False):
        row = dict(zip(["label_id", "feature_set", "evaluation"], keys))
        row["n_splits"] = len(group)
        row["n_features"] = int(group["n_features"].iloc[0])
        row["fit_wall_seconds_median"] = group["fit_wall_seconds"].median()
        for metric in ["pearson", "spearman", "r2", "rmse"]:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = np.mean(values)
            row[f"{metric}_median"] = np.median(values)
            row[f"{metric}_std"] = np.std(values, ddof=1) if len(values) > 1 else 0.0
            row[f"{metric}_iqr"] = np.quantile(values, 0.75) - np.quantile(values, 0.25)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_differences(metrics: pd.DataFrame) -> pd.DataFrame:
    reference = metrics[metrics["feature_set"] == "all_regions"][
        ["label_id", "evaluation", "split_name", "pearson"]
    ].rename(columns={"pearson": "reference_pearson"})
    joined = metrics.merge(reference, on=["label_id", "evaluation", "split_name"], how="inner")
    joined["pearson_delta_vs_all"] = joined["pearson"] - joined["reference_pearson"]
    rows = []
    rng = np.random.default_rng(13)
    for keys, group in joined.groupby(["label_id", "feature_set", "evaluation"], sort=False):
        delta = group["pearson_delta_vs_all"].to_numpy()
        bootstrap = np.mean(
            rng.choice(delta, size=(5000, len(delta)), replace=True),
            axis=1,
        )
        rows.append(
            {
                "label_id": keys[0],
                "feature_set": keys[1],
                "evaluation": keys[2],
                "n_paired_splits": len(group),
                "pearson_delta_mean": np.mean(delta),
                "pearson_delta_median": np.median(delta),
                "pearson_delta_std": np.std(delta, ddof=1) if len(delta) > 1 else 0.0,
                "pearson_delta_mean_ci_low": np.quantile(bootstrap, 0.025),
                "pearson_delta_mean_ci_high": np.quantile(bootstrap, 0.975),
                "win_fraction_vs_all": np.mean(delta > 0),
            }
        )
    return pd.DataFrame(rows)


def make_figures(root: Path, summary: pd.DataFrame, paired: pd.DataFrame) -> None:
    figure_dir = root / "docs/figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 10, "axes.labelsize": 9})

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for ax, evaluation in zip(axes.flat, ["repeated_random", "chromosome_holdout"] * 2):
        metric = "pearson_median" if ax in axes[0] else "pearson_delta_median"
        data = summary if ax in axes[0] else paired
        table = (
            data[data["evaluation"] == evaluation]
            .pivot(index="feature_set", columns="label_id", values=metric)
            .reindex(index=FEATURE_ORDER, columns=LABEL_ORDER)
        )
        if ax in axes[0]:
            image = ax.imshow(table, aspect="auto", cmap="viridis", vmin=0, vmax=0.8)
            title = f"{evaluation.replace('_', ' ')}: median Pearson"
        else:
            image = ax.imshow(table, aspect="auto", cmap="RdBu_r", vmin=-0.25, vmax=0.25)
            title = f"{evaluation.replace('_', ' ')}: delta vs all"
        ax.set_title(title)
        ax.set_xticks(range(len(LABEL_ORDER)))
        ax.set_xticklabels([LABEL_NAMES[item] for item in LABEL_ORDER], rotation=35, ha="right")
        ax.set_yticks(range(len(FEATURE_ORDER)))
        ax.set_yticklabels(FEATURE_ORDER)
        fig.colorbar(image, ax=ax, shrink=0.75)
    save_figure(fig, figure_dir, "input_ablation_overview")

    chromosome = summary[summary["evaluation"] == "chromosome_holdout"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 5), sharey=True, constrained_layout=True)
    for ax, label in zip(axes, LABEL_ORDER):
        data = chromosome[chromosome["label_id"] == label].set_index("feature_set").reindex(FEATURE_ORDER)
        ax.barh(range(len(data)), data["pearson_median"], color="#3b7a8f")
        ax.set_title(LABEL_NAMES[label])
        ax.set_xlabel("Median Pearson")
        ax.set_xlim(0, 0.85)
        ax.invert_yaxis()
    axes[0].set_yticks(range(len(FEATURE_ORDER)))
    axes[0].set_yticklabels(FEATURE_ORDER)
    save_figure(fig, figure_dir, "input_ablation_chromosome_holdout")


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    for suffix in ["png", "svg", "pdf"]:
        fig.savefig(directory / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(root: Path, summary: pd.DataFrame, paired: pd.DataFrame) -> None:
    chromosome = summary[summary["evaluation"] == "chromosome_holdout"]
    lines = [
        "# Engineered-Feature Input Ablation",
        "",
        "Full XGBoost was evaluated on the fixed fair-benchmark manifests using 16 interpretable input feature sets.",
        "",
        "![Input ablation overview](figures/input_ablation_overview.png)",
        "",
        "## Best Reduced Input by Label",
        "",
        "| Label | Best reduced set | Median Pearson | Delta versus all regions |",
        "| --- | --- | ---: | ---: |",
    ]
    for label in LABEL_ORDER:
        candidates = chromosome[
            (chromosome["label_id"] == label) & (chromosome["feature_set"] != "all_regions")
        ].sort_values("pearson_median", ascending=False)
        best = candidates.iloc[0]
        delta = paired[
            (paired["label_id"] == label)
            & (paired["feature_set"] == best.feature_set)
            & (paired["evaluation"] == "chromosome_holdout")
        ].iloc[0]
        lines.append(
            f"| `{label}` | `{best.feature_set}` | {best.pearson_median:.3f} | "
            f"{delta.pearson_delta_median:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Main Conclusions",
            "",
            "1. CDS is the dominant individual region for all four labels; removing CDS from the structured representation causes the largest region-level performance loss.",
            "2. Explicit 5'UTR/CDS/3'UTR features outperform full-transcript-only features, showing that region identity carries useful information.",
            "3. Removing 5'UTR has only a small effect, whereas removing 3'UTR causes a moderate loss and removing CDS causes a large loss.",
            "4. K-mer-only inputs retain most of the full model performance; motif-only inputs are insufficient in the current motif panel.",
            "5. For exon_sense 6h/0h, 16 length/composition variables remain highly predictive, supporting the concern that this label contains strong abundance- or processing-linked sequence correlates.",
            "",
            "## Design Notes",
            "",
            "- All runs use the same train, validation, and test assignments as the fair benchmark.",
            "- Region leave-one-out sets exclude `full` features, preventing indirect leakage of the removed region.",
            "- This phase isolates engineered-feature information content using the strongest fair-benchmark model.",
            "- Deep raw-sequence region ablation and sequence-plus-tabular hybrid experiments remain a separate GPU phase.",
            "",
            "## Outputs",
            "",
            "- `data/processed/input_ablation_summary.tsv`",
            "- `data/processed/input_ablation_paired_differences.tsv`",
            "- `docs/figures/input_ablation_overview.{png,svg,pdf}`",
            "- `docs/figures/input_ablation_chromosome_holdout.{png,svg,pdf}`",
        ]
    )
    (root / "docs/input_ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
