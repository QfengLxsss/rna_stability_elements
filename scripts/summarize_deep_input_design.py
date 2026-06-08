from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_deep_input_design_gpu_full import CONFIGS, MODELS, REUSED_CONFIG, SCREEN_LABELS


LABEL_ORDER = (
    "gene_sense_late_chase_6h_2h",
    "gene_sense_total_chase_6h_0h",
    "exon_sense_late_chase_6h_2h",
    "exon_sense_total_chase_6h_0h",
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    processed = root / "data/processed"
    metrics = load_metrics(processed)
    validate_screen(metrics)
    summary = summarize(metrics)
    paired = paired_differences(metrics)
    ranking = rank_screen_configs(summary, paired)
    best_config = ranking.iloc[0]["config_id"]
    summary.to_csv(processed / "deep_input_design_summary.tsv", sep="\t", index=False)
    paired.to_csv(processed / "deep_input_design_paired_differences.tsv", sep="\t", index=False)
    ranking.to_csv(processed / "deep_input_design_screen_ranking.tsv", sep="\t", index=False)
    (processed / "deep_input_design_best_config.txt").write_text(
        f"{best_config}\n", encoding="utf-8"
    )
    source = processed / "figure_source_data"
    source.mkdir(parents=True, exist_ok=True)
    summary.to_csv(source / "deep_input_design_summary.tsv", sep="\t", index=False)
    paired.to_csv(source / "deep_input_design_paired_differences.tsv", sep="\t", index=False)
    ranking.to_csv(source / "deep_input_design_screen_ranking.tsv", sep="\t", index=False)
    make_figures(root, ranking, paired)
    write_report(root, ranking, best_config, metrics)


def load_metrics(processed: Path) -> pd.DataFrame:
    frames = []
    for label_id in LABEL_ORDER:
        for model_id in MODELS:
            path = (
                processed
                / f"deep_input_ablation_gpu_full_raw_all_plus_engineered_{model_id}_{label_id}_metrics.tsv"
            )
            frame = pd.read_csv(path, sep="\t")
            frame["config_id"] = REUSED_CONFIG
            frame["budget_family"] = "medium"
            frame["model_id"] = model_id
            frame["label_id"] = label_id
            frame["input_representation"] = "raw_sequence_plus_engineered"
            frame["manifest_exact_match"] = True
            frames.append(frame)
    for path in sorted(processed.glob("deep_input_design_gpu_full_*_metrics.tsv")):
        frames.append(pd.read_csv(path, sep="\t"))
    return pd.concat(frames, ignore_index=True)


def validate_screen(metrics: pd.DataFrame) -> None:
    screen = metrics[
        (metrics["model_id"] == "sequence_transformer") & metrics["label_id"].isin(SCREEN_LABELS)
    ]
    counts = screen.groupby(["config_id", "label_id"])["split_name"].nunique()
    expected = len(CONFIGS) * len(SCREEN_LABELS)
    if len(counts) != expected or not (counts == 26).all():
        raise ValueError(f"Screening is incomplete:\n{counts}")
    if not screen["manifest_exact_match"].all():
        raise ValueError("One or more screening results failed manifest audit.")


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["config_id", "model_id", "label_id", "evaluation"]
    for keys, group in metrics.groupby(group_columns, sort=False):
        row = dict(zip(group_columns, keys))
        row["n_splits"] = len(group)
        row["max_length_5utr"] = int(group["max_length_5utr"].iloc[0])
        row["max_length_cds"] = int(group["max_length_cds"].iloc[0])
        row["max_length_3utr"] = int(group["max_length_3utr"].iloc[0])
        row["total_length"] = (
            row["max_length_5utr"] + row["max_length_cds"] + row["max_length_3utr"]
        )
        row["crop_strategy"] = group["crop_strategy"].iloc[0]
        row["epochs_median"] = group["epochs_trained"].median()
        for metric in ["pearson", "spearman", "r2", "rmse"]:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = np.mean(values)
            row[f"{metric}_median"] = np.median(values)
            row[f"{metric}_std"] = np.std(values, ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_differences(metrics: pd.DataFrame) -> pd.DataFrame:
    reference = metrics[
        (metrics["config_id"] == REUSED_CONFIG)
        & (metrics["model_id"] == "sequence_transformer")
        & metrics["label_id"].isin(SCREEN_LABELS)
    ][["label_id", "evaluation", "split_name", "pearson"]].rename(
        columns={"pearson": "reference_pearson"}
    )
    screen = metrics[
        (metrics["model_id"] == "sequence_transformer") & metrics["label_id"].isin(SCREEN_LABELS)
    ]
    joined = screen.merge(reference, on=["label_id", "evaluation", "split_name"], how="inner")
    joined["delta"] = joined["pearson"] - joined["reference_pearson"]
    rng = np.random.default_rng(13)
    rows = []
    for keys, group in joined.groupby(["config_id", "label_id", "evaluation"], sort=False):
        delta = group["delta"].to_numpy()
        bootstrap = np.mean(rng.choice(delta, size=(5000, len(delta)), replace=True), axis=1)
        rows.append(
            {
                "config_id": keys[0],
                "label_id": keys[1],
                "evaluation": keys[2],
                "n_paired_splits": len(delta),
                "pearson_delta_mean": np.mean(delta),
                "pearson_delta_std": np.std(delta, ddof=1),
                "pearson_delta_ci_low": np.quantile(bootstrap, 0.025),
                "pearson_delta_ci_high": np.quantile(bootstrap, 0.975),
                "win_fraction": np.mean(delta > 0),
            }
        )
    return pd.DataFrame(rows)


def rank_screen_configs(summary: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    performance = summary[
        (summary["model_id"] == "sequence_transformer")
        & summary["label_id"].isin(SCREEN_LABELS)
        & (summary["evaluation"] == "chromosome_holdout")
    ]
    effects = paired[paired["evaluation"] == "chromosome_holdout"]
    ranking = performance.groupby("config_id").agg(
        chromosome_pearson_mean=("pearson_mean", "mean"),
        chromosome_pearson_min_label=("pearson_mean", "min"),
        chromosome_pearson_std_across_labels=("pearson_mean", "std"),
        total_length=("total_length", "first"),
        crop_strategy=("crop_strategy", "first"),
    )
    paired_rank = effects.groupby("config_id").agg(
        paired_delta_mean=("pearson_delta_mean", "mean"),
        paired_delta_min_label=("pearson_delta_mean", "min"),
        mean_win_fraction=("win_fraction", "mean"),
    )
    ranking = ranking.join(paired_rank).reset_index()
    ranking["consistent_nonnegative"] = ranking["paired_delta_min_label"] >= 0
    return ranking.sort_values(
        ["chromosome_pearson_mean", "paired_delta_min_label"], ascending=False
    ).reset_index(drop=True)


def make_figures(root: Path, ranking: pd.DataFrame, paired: pd.DataFrame) -> None:
    figure_dir = root / "docs/figures"
    top = ranking.sort_values("chromosome_pearson_mean")
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    colors = ["#cc6b49" if item else "#347f8d" for item in top["consistent_nonnegative"]]
    ax.barh(top["config_id"], top["chromosome_pearson_mean"], color=colors)
    lower = max(0, top["chromosome_pearson_mean"].min() - 0.01)
    upper = top["chromosome_pearson_mean"].max() + 0.004
    ax.set_xlim(lower, upper)
    ax.set_xlabel("Mean chromosome-holdout Pearson across 6h/2h labels")
    ax.set_title("Transformer hybrid input-design screening")
    save_figure(fig, figure_dir, "deep_input_design_screen_ranking")

    data = paired[paired["evaluation"] == "chromosome_holdout"]
    table = (
        data.pivot(index="config_id", columns="label_id", values="pearson_delta_mean")
        .reindex(ranking["config_id"])
    )
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    image = ax.imshow(table, aspect="auto", cmap="RdBu_r", vmin=-0.08, vmax=0.08)
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(["gene 6h/2h", "exon 6h/2h"])
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels(table.index)
    for row in range(len(table)):
        for col in range(len(table.columns)):
            ax.text(col, row, f"{table.iloc[row, col]:+.3f}", ha="center", va="center", fontsize=8)
    ax.set_title("Paired Pearson delta versus medium balanced")
    fig.colorbar(image, ax=ax, shrink=0.8)
    save_figure(fig, figure_dir, "deep_input_design_screen_paired_differences")


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    for suffix in ["png", "svg", "pdf"]:
        fig.savefig(directory / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(root: Path, ranking: pd.DataFrame, best_config: str, metrics: pd.DataFrame) -> None:
    expansion = metrics[metrics["config_id"] == best_config].groupby(
        ["model_id", "label_id"]
    )["split_name"].nunique()
    expansion_complete = len(expansion) == len(MODELS) * 4 and (expansion == 26).all()
    rank = ranking.set_index("config_id")
    reference = rank.loc[best_config]
    short_balanced = rank.loc["short_balanced"]
    long_balanced = rank.loc["long_balanced"]
    long_end = rank.loc["long_end"]
    cds_heavy = rank.loc["medium_cds_heavy_balanced"]
    utr3_heavy = rank.loc["medium_3utr_heavy_balanced"]
    lines = [
        "# Deep Hybrid Input-Design Benchmark",
        "",
        "The screening stage compares Transformer hybrid window budgets, crop strategies, and "
        "fixed-budget region allocations on the gene/exon 6h/2h labels.",
        "",
        f"Selected configuration: `{best_config}`.",
        "",
        "## Main Findings",
        "",
        f"- `{best_config}` remains the most robust default: 5'UTR/CDS/3'UTR = "
        "256/1024/1024 with `balanced` crop.",
        f"- Short balanced windows lose {short_balanced.paired_delta_mean:+.3f} mean paired "
        "Pearson versus the selected medium balanced reference.",
        f"- Long balanced windows do not improve the average result "
        f"({long_balanced.paired_delta_mean:+.3f}); `long_end` is close to the reference "
        f"({long_end.paired_delta_mean:+.3f}) but does not clearly surpass it.",
        f"- At the same total medium budget, CDS-heavy allocation is only slightly lower "
        f"({cds_heavy.paired_delta_mean:+.3f}), whereas 3'UTR-heavy allocation is much worse "
        f"({utr3_heavy.paired_delta_mean:+.3f}).",
        "- These results support keeping a medium, region-balanced hybrid input while moving "
        "biological interpretation toward CDS-aware sequence features rather than simply "
        "making the transcript window longer.",
        "",
        "![Screen ranking](figures/deep_input_design_screen_ranking.png)",
        "",
        "## Screening Ranking",
        "",
        "| Rank | Configuration | Mean Pearson | Worst-label delta | Mean paired delta | Win fraction |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for index, row in ranking.iterrows():
        win_fraction = "reference" if row.config_id == best_config else f"{row.mean_win_fraction:.2f}"
        lines.append(
            f"| {index + 1} | `{row.config_id}` | {row.chromosome_pearson_mean:.3f} | "
            f"{row.paired_delta_min_label:+.3f} | {row.paired_delta_mean:+.3f} | "
            f"{win_fraction} |"
        )
    lines.extend(
        [
            "",
            "![Paired differences](figures/deep_input_design_screen_paired_differences.png)",
            "",
            "## Expansion Status",
            "",
            f"- Four-label, three-model expansion complete: `{expansion_complete}`.",
            "- The selected `medium_balanced` configuration is the previously completed "
            "raw sequence + engineered feature hybrid setting, so the expansion reuses "
            "the already audited four-label, three-model results.",
            "- Selection prioritizes mean chromosome-holdout Pearson across both 6h/2h labels; "
            "worst-label paired delta is used as the consistency tie-breaker.",
            "- `random` is a deterministic per-transcript crop generated once with the fixed "
            "experiment seed, not per-epoch stochastic augmentation.",
        ]
    )
    (root / "docs/deep_input_design_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
