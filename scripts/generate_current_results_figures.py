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
DEEP_MODELS = ["region_cnn", "sequence_transformer", "saluki_like"]
DEEP_NAMES = ["Region-CNN", "Transformer", "Saluki-like"]
PALETTE = {
    "gene": "#4C78A8",
    "exon": "#E07B39",
    "late": "#7A9CC6",
    "total": "#D88752",
    "elasticnet": "#8C8C8C",
    "xgboost": "#3E7C59",
    "deep": "#C44E52",
}


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
            "axes.linewidth": 0.7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.frameon": False,
            "legend.fontsize": 6.5,
        }
    )


def save_figure(fig: plt.Figure, figure_dir: Path, name: str) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in [
        ("png", {"dpi": 300}),
        ("svg", {}),
        ("pdf", {}),
    ]:
        fig.savefig(figure_dir / f"{name}.{suffix}", bbox_inches="tight", **kwargs)


def main() -> None:
    setup_style()
    root = Path(__file__).resolve().parents[1]
    processed = root / "data/processed"
    figure_dir = root / "docs/figures"
    source_dir = processed / "figure_source_data"
    source_dir.mkdir(parents=True, exist_ok=True)

    quality = pd.read_csv(processed / "parallel_label_quality_summary.tsv", sep="\t")
    deep = pd.read_csv(processed / "parallel_deep_gpu_full_summary.tsv", sep="\t")
    suite = pd.read_csv(processed / "parallel_model_suite_summary.tsv", sep="\t")

    quality.to_csv(source_dir / "current_results_label_quality.tsv", sep="\t", index=False)
    deep.to_csv(source_dir / "current_results_gpu_full.tsv", sep="\t", index=False)
    pipeline = build_pipeline_comparison(suite)
    pipeline.to_csv(source_dir / "current_results_pipeline_comparison.tsv", sep="\t", index=False)

    write_overview_figure(quality, deep, pipeline, figure_dir)
    write_gpu_full_figure(deep, figure_dir)


def build_pipeline_comparison(suite: pd.DataFrame) -> pd.DataFrame:
    data = suite[suite["evaluation"] == "repeated_random"].copy()
    keep = data[
        data["model"].isin(
            [
                "elasticnet_full",
                "xgboost_light",
                "region_cnn_gpu_full",
                "sequence_transformer_gpu_full",
                "saluki_like_gpu_full",
            ]
        )
    ].copy()
    keep["pipeline"] = keep["model"].map(
        {
            "elasticnet_full": "ElasticNet + engineered features",
            "xgboost_light": "XGBoost-light + engineered features",
            "region_cnn_gpu_full": "Best GPU-full raw sequence",
            "sequence_transformer_gpu_full": "Best GPU-full raw sequence",
            "saluki_like_gpu_full": "Best GPU-full raw sequence",
        }
    )
    deep_best = (
        keep[keep["pipeline"] == "Best GPU-full raw sequence"]
        .sort_values("pearson_median", ascending=False)
        .groupby("label_id", as_index=False)
        .head(1)
    )
    conventional = keep[keep["pipeline"] != "Best GPU-full raw sequence"]
    return pd.concat([conventional, deep_best], ignore_index=True)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def write_overview_figure(
    quality: pd.DataFrame,
    deep: pd.DataFrame,
    pipeline: pd.DataFrame,
    figure_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.7), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.flat

    q = quality.set_index("label_id").reindex(LABEL_ORDER)
    x = np.arange(len(q))
    colors = [PALETTE["gene"], PALETTE["gene"], PALETTE["exon"], PALETTE["exon"]]
    ax_a.bar(x, q["feature_rows"] / 1000, color=colors, width=0.68)
    for index, value in enumerate(q["feature_rows"]):
        ax_a.text(index, value / 1000 + 0.12, f"{value:,}", ha="center", va="bottom", fontsize=6)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([LABEL_NAMES[item] for item in LABEL_ORDER])
    ax_a.set_ylabel("Model-ready genes (thousands)")
    ax_a.set_ylim(0, 11)
    ax_a.set_title("Strict label coverage")
    panel_label(ax_a, "a")

    rr = deep[deep["evaluation"] == "repeated_random"].pivot(
        index="model", columns="label_id", values="pearson_median"
    )
    rr = rr.reindex(index=DEEP_MODELS, columns=LABEL_ORDER)
    image = ax_b.imshow(rr.to_numpy(), cmap="Blues", vmin=0.35, vmax=0.80, aspect="auto")
    for row in range(rr.shape[0]):
        for col in range(rr.shape[1]):
            value = rr.iloc[row, col]
            ax_b.text(col, row, f"{value:.3f}", ha="center", va="center", fontsize=6.2)
    ax_b.set_xticks(np.arange(4))
    ax_b.set_xticklabels([LABEL_NAMES[item] for item in LABEL_ORDER])
    ax_b.set_yticks(np.arange(3))
    ax_b.set_yticklabels(DEEP_NAMES)
    ax_b.set_title("GPU-full raw-sequence performance")
    cbar = fig.colorbar(image, ax=ax_b, fraction=0.04, pad=0.02)
    cbar.set_label("Repeated-random Pearson")
    panel_label(ax_b, "b")

    pipeline_order = [
        "ElasticNet + engineered features",
        "XGBoost-light + engineered features",
        "Best GPU-full raw sequence",
    ]
    pipeline_colors = [PALETTE["elasticnet"], PALETTE["xgboost"], PALETTE["deep"]]
    width = 0.23
    for offset, (name, color) in enumerate(zip(pipeline_order, pipeline_colors)):
        values = (
            pipeline[pipeline["pipeline"] == name]
            .set_index("label_id")
            .reindex(LABEL_ORDER)["pearson_median"]
        )
        ax_c.bar(x + (offset - 1) * width, values, width=width, color=color, label=name)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([LABEL_NAMES[item] for item in LABEL_ORDER])
    ax_c.set_ylabel("Repeated-random Pearson")
    ax_c.set_ylim(0.3, 0.82)
    ax_c.set_title("Input representation and model pipeline")
    ax_c.legend(loc="upper left", fontsize=5.8)
    panel_label(ax_c, "c")

    gpu = deep.pivot_table(
        index=["label_id", "model"], columns="evaluation", values="pearson_median"
    ).reset_index()
    gpu["generalization_gap"] = gpu["repeated_random"] - gpu["chromosome_holdout"]
    positions = np.arange(len(LABEL_ORDER))
    for model, name, marker in zip(DEEP_MODELS, DEEP_NAMES, ["o", "s", "^"]):
        values = gpu[gpu["model"] == model].set_index("label_id").reindex(LABEL_ORDER)
        ax_d.plot(
            positions,
            values["generalization_gap"].to_numpy(),
            marker=marker,
            markersize=4,
            linewidth=1.1,
            label=name,
        )
    ax_d.axhline(0, color="#666666", linewidth=0.7, linestyle="--")
    ax_d.set_xticks(positions)
    ax_d.set_xticklabels([LABEL_NAMES[item] for item in LABEL_ORDER])
    ax_d.set_ylabel("Pearson gap: random - chromosome")
    ax_d.set_title("Chromosome-holdout robustness")
    ax_d.legend(loc="lower left")
    panel_label(ax_d, "d")

    save_figure(fig, figure_dir, "current_results_overview")
    plt.close(fig)


def write_gpu_full_figure(deep: pd.DataFrame, figure_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True, constrained_layout=True)
    x = np.arange(len(LABEL_ORDER))
    width = 0.23
    model_colors = ["#7A9CC6", "#4C78A8", "#D88752"]
    for ax, evaluation, title in zip(
        axes,
        ["repeated_random", "chromosome_holdout"],
        ["Repeated-random evaluation", "Chromosome-holdout evaluation"],
    ):
        subset = deep[deep["evaluation"] == evaluation]
        for index, (model, name, color) in enumerate(zip(DEEP_MODELS, DEEP_NAMES, model_colors)):
            values = subset[subset["model"] == model].set_index("label_id").reindex(LABEL_ORDER)
            ax.bar(x + (index - 1) * width, values["pearson_median"], width, color=color, label=name)
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL_NAMES[item] for item in LABEL_ORDER])
        ax.set_ylim(0.3, 0.82)
        ax.set_title(title)
        ax.set_ylabel("Median Pearson across splits")
    axes[0].legend(loc="upper left")
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    save_figure(fig, figure_dir, "gpu_full_model_comparison")
    plt.close(fig)


if __name__ == "__main__":
    main()
