from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLOR = {
    "compact": "#2F80ED",
    "deep": "#00A6A6",
    "lm": "#7B61FF",
    "hybrid": "#F2A541",
    "grid": "#D0D5DD",
    "ink": "#202124",
    "muted": "#667085",
    "stable": "#27AE60",
    "unstable": "#D64550",
}


def write_leaderboard_and_grammar_report(
    *,
    processed_dir: str | Path = "data/processed",
    figure_dir: str | Path = "docs/figures",
    report_out: str | Path = "docs/rna_stability_grammar_interpretation_report.md",
) -> dict[str, Path]:
    processed_dir = Path(processed_dir)
    figure_dir = Path(figure_dir)
    report_out = Path(report_out)
    figure_dir.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)

    leaderboard = build_model_leaderboard(processed_dir)
    grammar = build_sequence_grammar_tables(processed_dir)

    leaderboard_path = processed_dir / "model_leaderboard.tsv"
    grammar_path = processed_dir / "rna_stability_sequence_grammar.tsv"
    group_path = processed_dir / "rna_stability_feature_group_importance.tsv"
    leaderboard.to_csv(leaderboard_path, sep="\t", index=False)
    grammar["features"].to_csv(grammar_path, sep="\t", index=False)
    grammar["groups"].to_csv(group_path, sep="\t", index=False)

    figures = {
        "leaderboard_random": figure_dir / "model_leaderboard_repeated_random.png",
        "leaderboard_chromosome": figure_dir / "model_leaderboard_chromosome_holdout.png",
        "feature_groups": figure_dir / "grammar_feature_group_importance.png",
        "kmer_direction": figure_dir / "grammar_kmer_direction.png",
    }
    plot_leaderboard(leaderboard, evaluation="repeated_random", out=figures["leaderboard_random"])
    plot_leaderboard(leaderboard, evaluation="chromosome_holdout", out=figures["leaderboard_chromosome"])
    plot_feature_groups(grammar["groups"], out=figures["feature_groups"])
    plot_kmer_direction(grammar["features"], out=figures["kmer_direction"])
    write_grammar_report(report_out, leaderboard=leaderboard, grammar=grammar)
    return {
        "leaderboard": leaderboard_path,
        "grammar": grammar_path,
        "feature_group_table": group_path,
        "report": report_out,
        **figures,
    }


def build_model_leaderboard(processed_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    specs = [
        ("strict_sequence_model_evaluation_summary.tsv", "Compact feature baseline", "compact"),
        ("region_cnn_tuned_strong_random5_summary.tsv", "Region CNN tuned", "deep"),
        ("region_cnn_tuned_strong_chromosome_summary.tsv", "Region CNN tuned", "deep"),
        ("saluki_like_random5_metrics.tsv", "Saluki-like CNN+GRU", "deep"),
        ("saluki_like_chromosome_summary.tsv", "Saluki-like CNN+GRU", "deep"),
        ("sequence_transformer_short_hybrid_random5_summary.tsv", "Conv-tokenized Transformer", "deep"),
        ("sequence_transformer_short_hybrid_chromosome_summary.tsv", "Conv-tokenized Transformer", "deep"),
        ("rna_bert_pilot_summary.tsv", "RNA BERT-style k-mer", "deep"),
        ("rna_lm_nucleotide_transformer_head_comparison_summary.tsv", "Nucleotide Transformer LM", "lm"),
        ("rna_lm_nucleotide_transformer_hybrid_xgboost_summary.tsv", "LM + compact hybrid", "hybrid"),
        ("rna_lm_nucleotide_transformer_hybrid_region_pca_mlp_summary.tsv", "LM + compact hybrid", "hybrid"),
    ]
    for filename, source, family in specs:
        path = processed_dir / filename
        if not path.exists():
            continue
        frame = read_metric_table(path)
        if frame.empty:
            continue
        frame["source_file"] = filename
        frame["model_source"] = source
        frame["model_family"] = family
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    board = pd.concat(rows, ignore_index=True)
    board["display_model"] = board.apply(display_model_name, axis=1)
    board = board.sort_values(["evaluation", "pearson_mean"], ascending=[True, False]).reset_index(drop=True)
    board["rank_within_evaluation"] = board.groupby("evaluation")["pearson_mean"].rank(method="first", ascending=False).astype(int)
    return board


def read_metric_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if {"evaluation", "pearson_mean", "spearman_mean"}.issubset(df.columns):
        keep = [
            column
            for column in [
                "experiment",
                "evaluation",
                "model",
                "feature_set",
                "n_splits",
                "n_features_median",
                "pearson_mean",
                "spearman_mean",
                "r2_mean",
                "rmse_mean",
            ]
            if column in df.columns
        ]
        return df[keep].copy()
    if {"evaluation", "pearson", "spearman"}.issubset(df.columns):
        group_cols = ["evaluation"]
        if "model" in df:
            group_cols.append("model")
        if "feature_set" in df:
            group_cols.append("feature_set")
        return summarize_metrics(df, group_cols)
    if "stat" in df.columns and "mean" in set(df["stat"].astype(str)):
        row = df[df["stat"].astype(str) == "mean"].iloc[0].to_dict()
        evaluation = "chromosome_holdout" if "chromosome" in path.name else "repeated_random"
        return pd.DataFrame(
            [
                {
                    "evaluation": evaluation,
                    "model": path.stem,
                    "feature_set": "all",
                    "n_splits": np.nan,
                    "pearson_mean": row.get("pearson", np.nan),
                    "spearman_mean": row.get("spearman", np.nan),
                    "r2_mean": row.get("r2", np.nan),
                    "rmse_mean": row.get("rmse", np.nan),
                }
            ]
        )
    if {"model_variant", "pearson", "spearman"}.issubset(df.columns):
        out = df.rename(columns={"model_variant": "model", "pearson": "pearson_mean", "spearman": "spearman_mean", "r2": "r2_mean", "rmse": "rmse_mean"}).copy()
        out["evaluation"] = "single_split"
        out["feature_set"] = "all"
        out["n_splits"] = 1
        return out[["evaluation", "model", "feature_set", "n_splits", "pearson_mean", "spearman_mean", "r2_mean", "rmse_mean"]]
    return pd.DataFrame()


def summarize_metrics(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_splits"] = len(group)
        for metric in ["pearson", "spearman", "r2", "rmse"]:
            if metric in group:
                row[f"{metric}_mean"] = float(group[metric].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def display_model_name(row: pd.Series) -> str:
    source = str(row.get("model_source", ""))
    model = str(row.get("model", ""))
    feature_set = str(row.get("feature_set", ""))
    if source == "Compact feature baseline":
        return f"{model} ({feature_set})"
    if source == "Nucleotide Transformer LM":
        experiment = str(row.get("experiment", ""))
        return f"NT frozen embedding {model}"
    if source == "LM + compact hybrid":
        return f"Hybrid {model}"
    return source if source else model


def build_sequence_grammar_tables(processed_dir: Path) -> dict[str, pd.DataFrame]:
    xgb = pd.read_csv(processed_dir / "strict_sequence_xgboost_gpu_importance_summary.tsv", sep="\t")
    elastic = pd.read_csv(processed_dir / "compact_elasticnet_for_interpretation_importance.tsv", sep="\t")
    elastic_summary = (
        elastic.groupby(["feature", "feature_group"], as_index=False)
        .agg(
            coef_mean=("importance", "mean"),
            coef_abs_mean=("importance_abs", "mean"),
            coef_std=("importance", "std"),
            n_splits=("importance", "size"),
        )
    )
    xgb_summary = xgb.rename(columns={"importance_abs_mean": "xgb_importance"})[
        ["feature", "feature_group", "xgb_importance", "importance_mean", "n_splits"]
    ].rename(columns={"n_splits": "xgb_n_splits"})
    merged = xgb_summary.merge(elastic_summary, on=["feature", "feature_group"], how="outer")
    parsed = merged["feature"].map(parse_sequence_feature).apply(pd.Series)
    features = pd.concat([merged, parsed], axis=1)
    features["direction"] = np.where(features["coef_mean"] > 0, "stabilizing", "destabilizing")
    features.loc[features["coef_mean"].isna(), "direction"] = "importance_only"
    features["rank_score"] = features["xgb_importance"].fillna(0) * features["coef_abs_mean"].fillna(features["coef_abs_mean"].median())
    features = features.sort_values(["rank_score", "xgb_importance", "coef_abs_mean"], ascending=False).reset_index(drop=True)
    groups = summarize_feature_groups(features)
    return {"features": features, "groups": groups}


def parse_sequence_feature(feature: str) -> dict[str, str]:
    parts = str(feature).split("_")
    region = parts[0] if parts else "unknown"
    if "_kmer_" in feature:
        token = feature.rsplit("_kmer_", 1)[-1]
        kind = "kmer"
    elif "_motif_" in feature:
        token = feature.rsplit("_motif_", 1)[-1].replace("_count", "")
        kind = "motif"
    elif feature.endswith("_gc_fraction"):
        token = "GC_fraction"
        kind = "composition"
    elif feature.endswith("_au_fraction"):
        token = "AU_fraction"
        kind = "composition"
    elif feature.endswith("_u_fraction"):
        token = "U_fraction"
        kind = "composition"
    elif feature.endswith("_length"):
        token = "length"
        kind = "length"
    else:
        token = feature
        kind = "other"
    return {"region": region, "grammar_type": kind, "grammar_token": token}


def summarize_feature_groups(features: pd.DataFrame) -> pd.DataFrame:
    return (
        features.groupby(["region", "grammar_type"], as_index=False)
        .agg(
            xgb_importance_sum=("xgb_importance", "sum"),
            coef_abs_sum=("coef_abs_mean", "sum"),
            n_features=("feature", "nunique"),
        )
        .sort_values("xgb_importance_sum", ascending=False)
        .reset_index(drop=True)
    )


def plot_leaderboard(leaderboard: pd.DataFrame, *, evaluation: str, out: Path) -> None:
    data = leaderboard[leaderboard["evaluation"] == evaluation].copy()
    data = data.sort_values("pearson_mean", ascending=False).head(14)
    if data.empty:
        return
    colors = [COLOR.get(family, COLOR["muted"]) for family in data["model_family"]]
    fig, ax = plt.subplots(figsize=(11.5, 6.5), facecolor="white")
    y = np.arange(len(data))
    ax.barh(y, data["pearson_mean"], color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(data["display_model"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Pearson correlation")
    ax.set_title(f"Model Leaderboard: {evaluation}", loc="left", fontsize=16, fontweight="bold")
    ax.grid(axis="x", color=COLOR["grid"], linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i, value in enumerate(data["pearson_mean"]):
        ax.text(value + 0.006, i, f"{value:.3f}", va="center", fontsize=9, color=COLOR["ink"])
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_feature_groups(groups: pd.DataFrame, *, out: Path) -> None:
    data = groups.head(14).copy()
    labels = data["region"] + " " + data["grammar_type"]
    fig, ax = plt.subplots(figsize=(10.5, 6.2), facecolor="white")
    y = np.arange(len(data))
    ax.barh(y, data["xgb_importance_sum"], color=COLOR["compact"])
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Summed XGBoost importance")
    ax.set_title("Sequence Grammar Importance by Region", loc="left", fontsize=16, fontweight="bold")
    ax.grid(axis="x", color=COLOR["grid"], linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_kmer_direction(features: pd.DataFrame, *, out: Path) -> None:
    data = features[features["grammar_type"] == "kmer"].dropna(subset=["coef_mean"]).copy()
    stable = data.sort_values("coef_mean", ascending=False).head(12)
    unstable = data.sort_values("coef_mean", ascending=True).head(12)
    plot = pd.concat([unstable, stable], ignore_index=True)
    labels = plot["region"] + " " + plot["grammar_token"]
    colors = np.where(plot["coef_mean"] > 0, COLOR["stable"], COLOR["unstable"])
    fig, ax = plt.subplots(figsize=(10.5, 7), facecolor="white")
    y = np.arange(len(plot))
    ax.barh(y, plot["coef_mean"], color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color=COLOR["ink"], linewidth=1)
    ax.set_xlabel("ElasticNet coefficient mean")
    ax.set_title("Candidate Stabilizing and Destabilizing k-mers", loc="left", fontsize=16, fontweight="bold")
    ax.grid(axis="x", color=COLOR["grid"], linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def write_grammar_report(report_out: Path, *, leaderboard: pd.DataFrame, grammar: dict[str, pd.DataFrame]) -> None:
    chrom = leaderboard[leaderboard["evaluation"] == "chromosome_holdout"].sort_values("pearson_mean", ascending=False).head(8)
    random = leaderboard[leaderboard["evaluation"] == "repeated_random"].sort_values("pearson_mean", ascending=False).head(8)
    features = grammar["features"]
    groups = grammar["groups"]
    top_stable = features[(features["grammar_type"] == "kmer") & (features["coef_mean"] > 0)].sort_values("coef_mean", ascending=False).head(12)
    top_unstable = features[(features["grammar_type"] == "kmer") & (features["coef_mean"] < 0)].sort_values("coef_mean").head(12)
    top_xgb = features.sort_values("xgb_importance", ascending=False).head(15)

    text = f"""# RNA Stability Grammar Interpretation Report

## 1. 统一模型 Leaderboard

### Chromosome Holdout Top Models

{markdown_table(chrom[["display_model", "model_family", "pearson_mean", "spearman_mean", "r2_mean", "rmse_mean"]])}

### Repeated Random Top Models

{markdown_table(random[["display_model", "model_family", "pearson_mean", "spearman_mean", "r2_mean", "rmse_mean"]])}

![Chromosome holdout leaderboard](figures/model_leaderboard_chromosome_holdout.png)

![Repeated random leaderboard](figures/model_leaderboard_repeated_random.png)

## 2. 当前模型结论

- 当前总体最强仍是 compact k-mer/motif/composition XGBoost，chromosome holdout Pearson 约 0.496。
- Hybrid XGBoost 将 Nucleotide Transformer embedding 与 compact features 融合后，chromosome holdout Pearson 约 0.494，接近但没有超过 compact XGBoost。
- 当前最佳深度序列模型是 Conv-tokenized Transformer，chromosome holdout Pearson 约 0.470。
- LM-only XGBoost 约 0.442，说明 pretrained embedding 有信号，但不足以替代显式 k-mer grammar。

## 3. 第一版通用 RNA Stability Grammar

### Feature Group Importance

{markdown_table(groups.head(12))}

![Feature group importance](figures/grammar_feature_group_importance.png)

### XGBoost Top Sequence Features

{markdown_table(top_xgb[["feature", "region", "grammar_type", "grammar_token", "xgb_importance", "coef_mean"]])}

### Candidate Stabilizing k-mers

{markdown_table(top_stable[["feature", "region", "grammar_token", "coef_mean", "xgb_importance"]])}

### Candidate Destabilizing k-mers

{markdown_table(top_unstable[["feature", "region", "grammar_token", "coef_mean", "xgb_importance"]])}

![k-mer direction](figures/grammar_kmer_direction.png)

## 4. Biological Reading

第一版解释支持一个相对稳健的方向：RNA stability 的可预测信号主要来自局部序列语法，尤其是 CDS 和 3'UTR 的 k-mer / composition。5'UTR 也有信号，但贡献较弱。Nucleotide Transformer embedding 中 CDS 区域贡献最大，与 compact feature 的 CDS / 3'UTR k-mer 重要性互相印证。

这份报告仍然是统计解释，不等于机制验证。下一步应把 top k-mer 聚类为 motif grammar，并结合 in silico mutagenesis / saliency 在 deep sequence model 上验证这些候选语法是否真的改变预测稳定性。
"""
    report_out.write_text(text, encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows available._"
    formatted = frame.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
    columns = list(formatted.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in formatted.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)
