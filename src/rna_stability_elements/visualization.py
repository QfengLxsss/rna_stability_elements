from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch


PALETTE = {
    "ink": "#202124",
    "muted": "#667085",
    "grid": "#D0D5DD",
    "blue": "#2F80ED",
    "teal": "#00A6A6",
    "green": "#27AE60",
    "amber": "#F2A541",
    "red": "#D64550",
    "purple": "#7B61FF",
    "slate": "#344054",
    "panel": "#F7F9FC",
}


def make_progress_figures(
    *,
    processed_dir: str | Path = "data/processed",
    out_dir: str | Path = "docs/figures",
) -> dict[str, Path]:
    """Create the current project progress figure set."""
    processed_dir = Path(processed_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_project_results(processed_dir)

    paths = {
        "progress_dashboard": out_dir / "progress_dashboard.png",
        "data_funnel": out_dir / "data_funnel.png",
        "target_distribution": out_dir / "target_distribution.png",
        "target_agreement": out_dir / "target_agreement.png",
        "replicate_qc": out_dir / "replicate_qc.png",
        "sequence_landscape": out_dir / "sequence_landscape.png",
        "baseline_performance": out_dir / "baseline_performance.png",
        "model_leaderboard_overview": out_dir / "progress_model_leaderboard_overview.png",
        "model_family_generalization": out_dir / "progress_model_family_generalization.png",
        "grammar_overview": out_dir / "progress_grammar_overview.png",
    }
    plot_progress_dashboard(data, paths["progress_dashboard"])
    plot_data_funnel(data, paths["data_funnel"])
    plot_target_distribution(data, paths["target_distribution"])
    plot_target_agreement(data, paths["target_agreement"])
    plot_replicate_qc(data, paths["replicate_qc"])
    plot_sequence_landscape(data, paths["sequence_landscape"])
    plot_baseline_performance(data, paths["baseline_performance"])
    plot_model_leaderboard_overview(data, paths["model_leaderboard_overview"])
    plot_model_family_generalization(data, paths["model_family_generalization"])
    plot_grammar_overview(data, paths["grammar_overview"])
    return paths


def write_progress_report(
    *,
    out: str | Path = "docs/progress_visual_report.md",
    figure_dir: str | Path = "docs/figures",
    processed_dir: str | Path = "data/processed",
) -> Path:
    """Write a compact Chinese visual progress report referencing generated figures."""
    figure_dir = Path(figure_dir)
    processed_dir = Path(processed_dir)
    data = load_project_results(processed_dir)
    leaderboard = data["model_leaderboard"]
    chromosome_best = _best_leaderboard_row(leaderboard, "chromosome_holdout")
    repeated_best = _best_leaderboard_row(leaderboard, "repeated_random")
    master = data["master"]
    sequence_status = master["sequence_status"].value_counts().to_dict() if "sequence_status" in master else {}
    best_name = chromosome_best.get("display_model", "compact XGBoost-GPU")
    best_chromosome = float(chromosome_best.get("pearson_mean", 0.496))
    best_repeated = float(repeated_best.get("pearson_mean", 0.512))

    text = f"""# RNA Stability Elements: 可视化进展报告

## 当前做到哪里了

项目已经完成第一阶段的主要闭环：ENCODE BrU/BruChase-seq 数据整理、gene-level stability target、replicate QC、GENCODE v29 representative transcript 序列接入、compact sequence features、严格评估、统一模型 leaderboard，以及第一版 RNA stability grammar interpretation。

- 建模基因数: {len(master):,}
- GENCODE v29 sequence mapped genes: {sequence_status.get("mapped", 0):,}
- 当前主标签: `gene_sense` consensus median of `log2_stability_6h_2h`
- 当前最佳严格模型: {best_name}，repeated random Pearson = {best_repeated:.3f}，chromosome holdout Pearson = {best_chromosome:.3f}
- 最新模型和语法解释: `docs/rna_stability_grammar_interpretation_report.md`

## 图 1. 阶段一进度总览

![Progress dashboard](figures/progress_dashboard.png)

## 图 2. 全模型 Leaderboard

![Model leaderboard overview](figures/progress_model_leaderboard_overview.png)

## 图 3. 模型家族与泛化

![Model family generalization](figures/progress_model_family_generalization.png)

## 图 4. 第一版 RNA Stability Grammar

![Grammar overview](figures/progress_grammar_overview.png)

## 图 5. 数据漏斗与标签构建

![Data funnel](figures/data_funnel.png)

## 图 6. Target 分布与标签一致性

![Target distribution](figures/target_distribution.png)

![Target agreement](figures/target_agreement.png)

## 图 7. Replicate QC 与序列 landscape

![Replicate QC](figures/replicate_qc.png)

![Sequence landscape](figures/sequence_landscape.png)

## 目前结论

1. `gene_sense` 与 `exon_sense` target 在 gene x cell_line 和 consensus 层面均高度一致，说明标签不是单一 quantification 定义偶然造成的。
2. 10,907 个 consensus genes 已全部映射到 GENCODE v29 representative transcript，序列侧建模链路已经打通。
3. 严格评估显示 compact k-mer / motif / composition XGBoost-GPU 是当前最强模型，chromosome holdout Pearson 约 0.496。
4. Conv-tokenized Transformer 是当前最好的 deep sequence model，chromosome holdout Pearson 约 0.470；pretrained Nucleotide Transformer embedding 与 compact features 融合后接近但没有超过 compact XGBoost。
5. 第一版解释结果支持 CDS 和 3'UTR k-mer grammar 是当前最主要的可预测信号。下一步应优先做 in silico perturbation、motif clustering 和 residual analysis，而不是单纯继续堆模型。
"""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out


def load_project_results(processed_dir: Path) -> dict[str, object]:
    return {
        "exon_consensus": pd.read_csv(processed_dir / "stability_consensus_targets_exon_sense.tsv", sep="\t"),
        "gene_consensus": pd.read_csv(processed_dir / "stability_consensus_targets_gene_sense.tsv", sep="\t"),
        "target_comparison": pd.read_csv(processed_dir / "target_comparison_exon_vs_gene_sense.tsv", sep="\t"),
        "replicate_experiment": pd.read_csv(processed_dir / "qc_replicate_experiment_gene_sense.tsv", sep="\t"),
        "master": pd.read_csv(processed_dir / "modeling_master_with_sequences.tsv", sep="\t"),
        "sequence_features": _read_feature_summary(processed_dir / "sequence_model_features.tsv"),
        "baseline_metrics": load_baseline_metrics(processed_dir),
        "model_leaderboard": _read_optional_tsv(processed_dir / "model_leaderboard.tsv"),
        "grammar_groups": _read_optional_tsv(processed_dir / "rna_stability_feature_group_importance.tsv"),
        "grammar_features": _read_optional_tsv(processed_dir / "rna_stability_sequence_grammar.tsv"),
    }


def _read_optional_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def _best_leaderboard_row(leaderboard: pd.DataFrame, evaluation: str) -> dict[str, object]:
    if leaderboard.empty or "evaluation" not in leaderboard or "rank_within_evaluation" not in leaderboard:
        return {}
    subset = leaderboard[leaderboard["evaluation"] == evaluation].copy()
    if subset.empty:
        return {}
    subset = subset.sort_values(["rank_within_evaluation", "pearson_mean"], ascending=[True, False])
    return subset.iloc[0].to_dict()


def load_baseline_metrics(processed_dir: Path) -> list[dict[str, object]]:
    metrics = []
    for name in ["ridge", "elasticnet", "random_forest"]:
        path = processed_dir / f"baseline_sequence_{name}_metrics.json"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            row = json.load(handle)
        metrics.append(row)
    return metrics


def plot_progress_dashboard(data: dict[str, object], out: Path) -> None:
    master = data["master"]
    chromosome_best = _best_leaderboard_row(data["model_leaderboard"], "chromosome_holdout")
    best_model = chromosome_best.get("display_model", "xgboost_gpu (all)")
    best_pearson = float(chromosome_best.get("pearson_mean", 0.496))
    cards = [
        ("Cell lines", "16", "ENCODE BrU/BruChase"),
        ("Target rows", "150k", "gene_sense gene x cell line"),
        ("Consensus genes", f"{len(master):,}", "context-agnostic labels"),
        ("Sequence mapped", f"{int((master['sequence_status'] == 'mapped').sum()):,}", "GENCODE v29 transcripts"),
        ("Feature table", "1,346", "region k-mer + motifs"),
        ("Best chr Pearson", f"{best_pearson:.3f}", str(best_model)),
    ]
    steps = [
        ("ENCODE discovery", True),
        ("Target QC", True),
        ("Consensus label", True),
        ("GENCODE sequence", True),
        ("Baseline signal", True),
        ("Strict holdouts", True),
        ("Interpretation", True),
    ]

    fig = plt.figure(figsize=(13.5, 7.5), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.055, 0.91, "RNA Stability Elements", fontsize=25, fontweight="bold", color=PALETTE["ink"])
    ax.text(
        0.055,
        0.865,
        "Phase 1 progress: from ENCODE BrU/BruChase labels to strict sequence grammar",
        fontsize=12.5,
        color=PALETTE["muted"],
    )

    for index, (title, value, subtitle) in enumerate(cards):
        x = 0.055 + (index % 3) * 0.305
        y = 0.64 - (index // 3) * 0.22
        _rounded_panel(ax, x, y, 0.27, 0.15)
        ax.text(x + 0.025, y + 0.105, title, fontsize=10.5, color=PALETTE["muted"])
        ax.text(x + 0.025, y + 0.055, value, fontsize=24, fontweight="bold", color=PALETTE["blue"])
        ax.text(x + 0.025, y + 0.025, subtitle, fontsize=9.5, color=PALETTE["slate"])

    x0, y0 = 0.08, 0.18
    x1 = 0.92
    ax.plot([x0, x1], [y0, y0], color=PALETTE["grid"], lw=2)
    for index, (label, done) in enumerate(steps):
        x = x0 + index * (x1 - x0) / (len(steps) - 1)
        color = PALETTE["green"] if done else PALETTE["grid"]
        ax.scatter([x], [y0], s=320, color=color, edgecolors="white", linewidths=2, zorder=3)
        ax.text(x, y0 + 0.055, str(index + 1), ha="center", va="center", fontsize=10, color="white")
        ax.text(x, y0 - 0.065, label, ha="center", va="top", fontsize=9, color=PALETTE["slate"])

    ax.text(0.055, 0.065, "Next: in silico perturbation, motif clustering, and residual analysis", fontsize=11, color=PALETTE["muted"])
    save_figure(fig, out)


def plot_data_funnel(data: dict[str, object], out: Path) -> None:
    master = data["master"]
    mapped = int((master["sequence_status"] == "mapped").sum()) if "sequence_status" in master else len(master)
    feature_columns = 1346
    compact_numeric = _leaderboard_feature_count(data["model_leaderboard"], "compact", fallback=1336)
    hybrid_features = _leaderboard_feature_count(data["model_leaderboard"], "hybrid", fallback=5176)
    lm_dims = max(0, hybrid_features - compact_numeric)

    fig = plt.figure(figsize=(14.5, 7.8), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.055, 0.91, "Data and Label Construction", fontsize=23, fontweight="bold", color=PALETTE["ink"])
    ax.text(
        0.055,
        0.865,
        "Counts are grouped by meaning; feature dimensions are shown separately from biological sample/gene counts.",
        fontsize=11.5,
        color=PALETTE["muted"],
    )

    acquisition = [
        ("Cell lines", "16", "ENCODE BrU/BruChase"),
        ("Experiments", "48", "16 cell lines x 0h/2h/6h"),
        ("Gene quant TSV", "96", "2 biological replicates each"),
    ]
    targets = [
        ("Gene x cell-line rows", "150,233", "gene_sense stability target"),
        ("Consensus genes", f"{len(master):,}", "median across cell lines"),
        ("Mapped transcripts", f"{mapped:,}", "GENCODE v29 representative"),
    ]
    features = [
        ("Compact table", f"{feature_columns:,}", "metadata + target included"),
        ("Numeric compact features", f"{compact_numeric:,}", "strict XGBoost input"),
        ("NT region embedding", f"{lm_dims:,}", "5'UTR + CDS + 3'UTR"),
    ]

    ax.text(0.06, 0.775, "1. ENCODE acquisition", fontsize=13.5, fontweight="bold", color=PALETTE["slate"])
    ax.text(0.06, 0.465, "2. Target and sequence table", fontsize=13.5, fontweight="bold", color=PALETTE["slate"])
    ax.text(0.67, 0.775, "3. Feature representations", fontsize=13.5, fontweight="bold", color=PALETTE["slate"])

    _draw_card_row(ax, acquisition, x0=0.06, y=0.61, width=0.16, gap=0.025, color=PALETTE["blue"])
    _draw_card_row(ax, targets, x0=0.06, y=0.30, width=0.16, gap=0.025, color=PALETTE["green"])
    _draw_card_column(ax, features, x=0.68, y0=0.57, width=0.24, height=0.135, gap=0.034)

    _draw_flow_arrow(ax, 0.235, 0.58, 0.235, 0.49)
    _draw_flow_arrow(ax, 0.42, 0.58, 0.42, 0.49)
    _draw_flow_arrow(ax, 0.58, 0.39, 0.66, 0.62)

    ax.text(0.06, 0.18, "Main modeling unit: one row per gene after consensus target construction and transcript mapping.", fontsize=11, color=PALETTE["muted"])
    ax.text(0.68, 0.18, "Feature dimensions are model inputs, not additional genes or samples.", fontsize=11, color=PALETTE["muted"])
    save_figure(fig, out)


def plot_target_distribution(data: dict[str, object], out: Path) -> None:
    exon = data["exon_consensus"]["stability_consensus_median"]
    gene = data["gene_consensus"]["stability_consensus_median"]
    fig, ax = plt.subplots(figsize=(10.5, 6), facecolor="white")
    bins = np.linspace(min(exon.min(), gene.min()), max(exon.max(), gene.max()), 55)
    ax.hist(exon, bins=bins, alpha=0.48, density=True, label="exon_sense consensus", color=PALETTE["blue"])
    ax.hist(gene, bins=bins, alpha=0.58, density=True, label="gene_sense consensus", color=PALETTE["amber"])
    ax.axvline(gene.median(), color=PALETTE["amber"], lw=2)
    ax.axvline(exon.median(), color=PALETTE["blue"], lw=2)
    _style_axes(ax)
    ax.set_title("Consensus Stability Target Distribution", loc="left", fontsize=17, fontweight="bold")
    ax.set_xlabel("median log2 stability 6h / 2h")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=10)
    save_figure(fig, out)


def plot_target_agreement(data: dict[str, object], out: Path) -> None:
    exon = data["exon_consensus"][["gene_id", "stability_consensus_median"]]
    gene = data["gene_consensus"][["gene_id", "stability_consensus_median"]]
    merged = exon.merge(gene, on="gene_id", suffixes=("_exon", "_gene"))
    pearson = merged["stability_consensus_median_exon"].corr(merged["stability_consensus_median_gene"], method="pearson")
    spearman = merged["stability_consensus_median_exon"].corr(merged["stability_consensus_median_gene"], method="spearman")
    fig, ax = plt.subplots(figsize=(7.2, 6.6), facecolor="white")
    ax.hexbin(
        merged["stability_consensus_median_exon"],
        merged["stability_consensus_median_gene"],
        gridsize=58,
        cmap=_density_cmap(),
        mincnt=1,
        linewidths=0,
    )
    lim = [
        min(merged["stability_consensus_median_exon"].min(), merged["stability_consensus_median_gene"].min()),
        max(merged["stability_consensus_median_exon"].max(), merged["stability_consensus_median_gene"].max()),
    ]
    ax.plot(lim, lim, color=PALETTE["red"], lw=1.5, linestyle="--")
    _style_axes(ax)
    ax.set_title("Target Agreement: exon_sense vs gene_sense", loc="left", fontsize=15.5, fontweight="bold")
    ax.set_xlabel("exon_sense consensus")
    ax.set_ylabel("gene_sense consensus")
    ax.text(
        0.04,
        0.96,
        f"shared genes = {len(merged):,}\nPearson = {pearson:.3f}\nSpearman = {spearman:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.45", "fc": "white", "ec": PALETTE["grid"]},
    )
    save_figure(fig, out)


def plot_replicate_qc(data: dict[str, object], out: Path) -> None:
    rep = data["replicate_experiment"].copy()
    pivot = rep.pivot(index="cell_line", columns="time_h", values="replicate_pearson_mean")
    flag_counts = data["master"]["replicate_qc_flag"].value_counts().reindex(["pass", "review", "high_discordance"]).fillna(0)

    fig, (ax_heat, ax_bar) = plt.subplots(1, 2, figsize=(13, 6.8), gridspec_kw={"width_ratios": [1.45, 0.9]}, facecolor="white")
    im = ax_heat.imshow(pivot.values, aspect="auto", vmin=0.7, vmax=1.0, cmap=_qc_cmap())
    ax_heat.set_xticks(np.arange(len(pivot.columns)))
    ax_heat.set_xticklabels([f"{int(value)}h" for value in pivot.columns], fontsize=10)
    ax_heat.set_yticks(np.arange(len(pivot.index)))
    ax_heat.set_yticklabels(pivot.index, fontsize=8.5)
    ax_heat.set_title("Replicate Pearson by Cell Line and Time", loc="left", fontsize=15, fontweight="bold")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.values[i, j]
            ax_heat.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color="white" if value < 0.86 else PALETTE["ink"])
    cbar = fig.colorbar(im, ax=ax_heat, shrink=0.76)
    cbar.outline.set_visible(False)
    cbar.set_label("Pearson", fontsize=9)

    colors = [PALETTE["green"], PALETTE["amber"], PALETTE["red"]]
    ax_bar.bar(flag_counts.index, flag_counts.values, color=colors, width=0.62)
    _style_axes(ax_bar)
    ax_bar.set_title("Gene-Level Replicate QC Flags", loc="left", fontsize=15, fontweight="bold")
    ax_bar.set_ylabel("genes")
    for index, value in enumerate(flag_counts.values):
        ax_bar.text(index, value + max(flag_counts.values) * 0.02, f"{int(value):,}", ha="center", fontsize=10, fontweight="bold")
    save_figure(fig, out)


def plot_sequence_landscape(data: dict[str, object], out: Path) -> None:
    master = data["master"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), facecolor="white")
    axes = axes.ravel()
    panels = [
        ("length_3utr", "target_label", "3'UTR length vs stability", "3'UTR length", "target"),
        ("gc_3utr", "target_label", "3'UTR GC vs stability", "3'UTR GC fraction", "target"),
        ("length_full", "target_label", "Transcript length vs stability", "full transcript length", "target"),
        ("gc_full", "target_label", "Full transcript GC vs stability", "full GC fraction", "target"),
    ]
    for ax, (xcol, ycol, title, xlabel, ylabel) in zip(axes, panels):
        x = master[xcol].clip(upper=master[xcol].quantile(0.99)) if "length" in xcol else master[xcol]
        ax.hexbin(x, master[ycol], gridsize=45, cmap=_density_cmap(), mincnt=1, linewidths=0)
        _style_axes(ax)
        ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    fig.suptitle("Sequence Landscape Around the Consensus Stability Label", x=0.06, y=0.99, ha="left", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, out)


def plot_baseline_performance(data: dict[str, object], out: Path) -> None:
    metrics = pd.DataFrame(data["baseline_metrics"])
    metrics["model_label"] = metrics["model"].replace({"random_forest": "RandomForest", "elasticnet": "ElasticNet", "ridge": "Ridge"})
    metrics = metrics.sort_values("pearson")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), facecolor="white")
    plot_specs = [("pearson", "Pearson", PALETTE["blue"]), ("r2", "R2", PALETTE["teal"]), ("rmse", "RMSE", PALETTE["amber"])]
    for ax, (column, title, color) in zip(axes, plot_specs):
        ax.barh(metrics["model_label"], metrics[column], color=color, height=0.55)
        min_value = float(min(metrics[column].min(), 0))
        max_value = float(max(metrics[column].max(), 0))
        span = max(max_value - min_value, 0.1)
        ax.set_xlim(min_value - 0.18 * span, max_value + 0.18 * span)
        _style_axes(ax)
        ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
        for y, value in enumerate(metrics[column]):
            x = value + (0.03 * span if value >= 0 else -0.03 * span)
            ax.text(x, y, f"{value:.3f}", va="center", ha="left" if value >= 0 else "right", fontsize=10, fontweight="bold")
    fig.suptitle("Sequence Baseline Performance", x=0.045, y=1.02, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.045, -0.02, "Temporary deterministic 80/20 split; use strict holdouts before biological interpretation.", fontsize=10, color=PALETTE["muted"])
    fig.tight_layout()
    save_figure(fig, out)


def plot_model_leaderboard_overview(data: dict[str, object], out: Path) -> None:
    leaderboard = data["model_leaderboard"]
    if leaderboard.empty:
        _empty_figure("Model leaderboard is not available", out)
        return
    families = {"compact": PALETTE["blue"], "deep": PALETTE["teal"], "hybrid": PALETTE["amber"], "lm": PALETTE["purple"]}
    titles = [("chromosome_holdout", "Chromosome holdout"), ("repeated_random", "Repeated random")]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7), facecolor="white", sharex=True)
    for ax, (evaluation, title) in zip(axes, titles):
        subset = (
            leaderboard[leaderboard["evaluation"] == evaluation]
            .sort_values("pearson_mean", ascending=False)
            .drop_duplicates("display_model")
            .head(9)
            .sort_values("pearson_mean", ascending=True)
        )
        colors = [families.get(str(family), PALETTE["slate"]) for family in subset["model_family"]]
        ax.barh(subset["display_model"], subset["pearson_mean"], color=colors, height=0.62)
        _style_axes(ax)
        ax.set_title(title, loc="left", fontsize=15.5, fontweight="bold", color=PALETTE["ink"])
        ax.set_xlabel("Pearson correlation")
        ax.set_xlim(0.32, 0.535)
        for index, value in enumerate(subset["pearson_mean"]):
            ax.text(value + 0.004, index, f"{value:.3f}", va="center", fontsize=9.5, fontweight="bold", color=PALETTE["ink"])
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in families.values()]
    fig.legend(handles, ["Compact", "Deep sequence", "Hybrid", "LM embedding"], frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Strict Model Leaderboard Covers Compact, Deep, LM, and Hybrid Models", x=0.04, y=1.02, ha="left", fontsize=18, fontweight="bold")
    fig.text(0.04, -0.055, "Each panel shows the top unique model names under the corresponding strict evaluation protocol.", fontsize=10.5, color=PALETTE["muted"])
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    save_figure(fig, out)


def plot_model_family_generalization(data: dict[str, object], out: Path) -> None:
    leaderboard = data["model_leaderboard"]
    if leaderboard.empty:
        _empty_figure("Model family summary is not available", out)
        return
    families = {"compact": PALETTE["blue"], "deep": PALETTE["teal"], "hybrid": PALETTE["amber"], "lm": PALETTE["purple"]}
    best_family = (
        leaderboard.groupby(["evaluation", "model_family"], as_index=False)["pearson_mean"]
        .max()
        .pivot(index="model_family", columns="evaluation", values="pearson_mean")
        .reindex(["compact", "hybrid", "deep", "lm"])
    )
    paired = leaderboard.pivot_table(index=["display_model", "model_family"], columns="evaluation", values="pearson_mean", aggfunc="max").reset_index()
    paired = paired.dropna(subset=["chromosome_holdout", "repeated_random"], how="any")
    paired = paired.sort_values("chromosome_holdout", ascending=False).head(12)

    fig, (ax_bar, ax_scatter) = plt.subplots(1, 2, figsize=(14.5, 6.5), facecolor="white", gridspec_kw={"width_ratios": [0.9, 1.1]})
    x = np.arange(len(best_family.index))
    width = 0.36
    ax_bar.bar(x - width / 2, best_family["repeated_random"], width=width, color=[families.get(f, PALETTE["slate"]) for f in best_family.index], alpha=0.62, label="Repeated random")
    ax_bar.bar(x + width / 2, best_family["chromosome_holdout"], width=width, color=[families.get(f, PALETTE["slate"]) for f in best_family.index], alpha=1.0, label="Chromosome holdout")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(["Compact", "Hybrid", "Deep", "LM"], rotation=0)
    ax_bar.set_ylim(0.38, 0.535)
    ax_bar.set_ylabel("Best Pearson")
    ax_bar.set_title("Best Model in Each Family", loc="left", fontsize=15, fontweight="bold")
    _style_axes(ax_bar)
    ax_bar.legend(frameon=False, fontsize=9.5, loc="upper right")
    for xpos, family in enumerate(best_family.index):
        for offset, evaluation in [(-width / 2, "repeated_random"), (width / 2, "chromosome_holdout")]:
            value = best_family.loc[family, evaluation]
            if pd.notna(value):
                ax_bar.text(xpos + offset, value + 0.004, f"{value:.3f}", ha="center", fontsize=8.5, color=PALETTE["ink"])

    for family, group in paired.groupby("model_family"):
        ax_scatter.scatter(group["repeated_random"], group["chromosome_holdout"], s=95, color=families.get(family, PALETTE["slate"]), label=family, alpha=0.92, edgecolor="white", linewidth=0.9)
    lim = (0.40, 0.525)
    ax_scatter.plot(lim, lim, linestyle="--", color=PALETTE["grid"], lw=1.5)
    for _, row in paired.head(6).iterrows():
        label = _short_model_label(str(row["display_model"]))
        ax_scatter.text(row["repeated_random"] + 0.002, row["chromosome_holdout"] - 0.001, label, fontsize=8.5, color=PALETTE["slate"])
    ax_scatter.set_xlim(lim)
    ax_scatter.set_ylim(lim)
    ax_scatter.set_xlabel("Repeated random Pearson")
    ax_scatter.set_ylabel("Chromosome holdout Pearson")
    ax_scatter.set_title("Generalization Gap by Model", loc="left", fontsize=15, fontweight="bold")
    _style_axes(ax_scatter)
    ax_scatter.legend(frameon=False, fontsize=9, loc="lower right")

    fig.suptitle("Model Families Show a Small but Real Holdout Gap", x=0.04, y=1.02, ha="left", fontsize=18, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, out)


def plot_grammar_overview(data: dict[str, object], out: Path) -> None:
    groups = data["grammar_groups"]
    features = data["grammar_features"]
    if groups.empty or features.empty:
        _empty_figure("Grammar tables are not available", out)
        return
    top_groups = groups.sort_values("xgb_importance_sum", ascending=False).head(10).sort_values("xgb_importance_sum")
    stable = (
        features[(features["grammar_type"] == "kmer") & (features["direction"] == "stabilizing")]
        .sort_values(["coef_mean", "xgb_importance"], ascending=False)
        .head(8)
    )
    unstable = (
        features[(features["grammar_type"] == "kmer") & (features["direction"] == "destabilizing")]
        .sort_values(["coef_mean", "xgb_importance"], ascending=True)
        .head(8)
    )

    fig, (ax_group, ax_token) = plt.subplots(1, 2, figsize=(14.5, 6.7), facecolor="white", gridspec_kw={"width_ratios": [1.05, 1.0]})
    labels = [f"{row.region} {row.grammar_type}" for row in top_groups.itertuples()]
    colors = [PALETTE["blue"] if row.grammar_type == "kmer" else PALETTE["amber"] if row.grammar_type == "composition" else PALETTE["teal"] for row in top_groups.itertuples()]
    ax_group.barh(labels, top_groups["xgb_importance_sum"], color=colors, height=0.62)
    _style_axes(ax_group)
    ax_group.set_title("Feature Group Importance", loc="left", fontsize=15, fontweight="bold")
    ax_group.set_xlabel("XGBoost importance sum")
    for index, value in enumerate(top_groups["xgb_importance_sum"]):
        ax_group.text(value + 0.004, index, f"{value:.3f}", va="center", fontsize=9, fontweight="bold")

    token_rows = pd.concat([stable.assign(plot_direction="stabilizing"), unstable.assign(plot_direction="destabilizing")], ignore_index=True)
    token_rows["signed_score"] = token_rows["coef_mean"].fillna(0)
    token_rows = token_rows.sort_values("signed_score")
    token_labels = [f"{row.region}:{row.grammar_token}" for row in token_rows.itertuples()]
    token_colors = [PALETTE["green"] if value > 0 else PALETTE["red"] for value in token_rows["signed_score"]]
    ax_token.barh(token_labels, token_rows["signed_score"], color=token_colors, height=0.56)
    ax_token.axvline(0, color=PALETTE["grid"], lw=1.2)
    _style_axes(ax_token)
    ax_token.set_title("Candidate k-mer Direction", loc="left", fontsize=15, fontweight="bold")
    ax_token.set_xlabel("ElasticNet coefficient mean")
    ax_token.set_xlim(-0.065, 0.06)
    for index, value in enumerate(token_rows["signed_score"]):
        if value > 0:
            ax_token.text(value + 0.003, index, f"{value:.3f}", va="center", ha="left", fontsize=8.5, color=PALETTE["ink"])
        else:
            ax_token.text(value + 0.004, index, f"{value:.3f}", va="center", ha="left", fontsize=8.5, color="white")

    fig.suptitle("First-Pass RNA Stability Grammar: CDS and 3'UTR k-mers Dominate", x=0.04, y=1.02, ha="left", fontsize=18, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, out)


def _draw_card_row(
    ax: plt.Axes,
    cards: list[tuple[str, str, str]],
    *,
    x0: float,
    y: float,
    width: float,
    gap: float,
    color: str,
) -> None:
    for index, (title, value, subtitle) in enumerate(cards):
        x = x0 + index * (width + gap)
        _draw_metric_card(ax, x=x, y=y, width=width, height=0.15, title=title, value=value, subtitle=subtitle, color=color)
        if index < len(cards) - 1:
            _draw_flow_arrow(ax, x + width + 0.004, y + 0.075, x + width + gap - 0.006, y + 0.075)


def _draw_card_column(
    ax: plt.Axes,
    cards: list[tuple[str, str, str]],
    *,
    x: float,
    y0: float,
    width: float,
    height: float,
    gap: float,
) -> None:
    colors = [PALETTE["purple"], PALETTE["blue"], PALETTE["amber"]]
    for index, (title, value, subtitle) in enumerate(cards):
        y = y0 - index * (height + gap)
        _draw_metric_card(ax, x=x, y=y, width=width, height=height, title=title, value=value, subtitle=subtitle, color=colors[index % len(colors)])


def _draw_metric_card(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    value: str,
    subtitle: str,
    color: str,
) -> None:
    _rounded_panel(ax, x, y, width, height)
    ax.text(x + 0.018, y + height - 0.043, title, fontsize=10.5, color=PALETTE["muted"])
    ax.text(x + 0.018, y + height - 0.095, value, fontsize=21, fontweight="bold", color=color)
    ax.text(x + 0.018, y + 0.018, subtitle, fontsize=8.8, color=PALETTE["slate"])


def _draw_flow_arrow(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float) -> None:
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops={"arrowstyle": "->", "lw": 1.6, "color": PALETTE["grid"], "shrinkA": 0, "shrinkB": 0},
    )


def _leaderboard_feature_count(leaderboard: pd.DataFrame, family: str, *, fallback: int) -> int:
    if leaderboard.empty or "model_family" not in leaderboard or "n_features_median" not in leaderboard:
        return fallback
    subset = leaderboard[(leaderboard["model_family"] == family) & leaderboard["n_features_median"].notna()]
    if subset.empty:
        return fallback
    return int(round(float(subset["n_features_median"].max())))


def _read_feature_summary(path: Path) -> pd.DataFrame:
    columns = [
        "gene_id",
        "gene_symbol",
        "target_label",
        "full_length",
        "5utr_length",
        "cds_length",
        "3utr_length",
        "full_gc_fraction",
        "3utr_gc_fraction",
    ]
    existing = pd.read_csv(path, sep="\t", nrows=0).columns
    return pd.read_csv(path, sep="\t", usecols=[column for column in columns if column in existing])


def _short_model_label(label: str) -> str:
    replacements = {
        "xgboost_gpu (all)": "XGBoost",
        "Hybrid xgboost_gpu": "Hybrid",
        "Conv-tokenized Transformer": "Transformer",
        "Region CNN tuned": "CNN",
        "Saluki-like CNN+GRU": "Saluki-like",
        "random_forest (all)": "RF",
        "elasticnet (all)": "ElasticNet",
        "NT frozen embedding xgboost_gpu": "NT LM",
    }
    return replacements.get(label, label[:18])


def _empty_figure(message: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.5), facecolor="white")
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=14, color=PALETTE["muted"])
    save_figure(fig, out)


def save_figure(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["grid"])
    ax.spines["bottom"].set_color(PALETTE["grid"])
    ax.tick_params(colors=PALETTE["slate"], labelsize=9)
    ax.grid(True, axis="y", color=PALETTE["grid"], alpha=0.45, linewidth=0.8)


def _rounded_panel(ax: plt.Axes, x: float, y: float, width: float, height: float) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1,
        edgecolor="#E4E7EC",
        facecolor=PALETTE["panel"],
    )
    ax.add_patch(patch)


def _density_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("rse_density", ["#F2F4F7", "#9ADBE8", "#2F80ED", "#1D3557"])


def _qc_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("rse_qc", [PALETTE["red"], PALETTE["amber"], PALETTE["green"]])
