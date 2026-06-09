from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = (
    "gene_sense_late_chase_6h_2h",
    "gene_sense_total_chase_6h_0h",
    "exon_sense_late_chase_6h_2h",
    "exon_sense_total_chase_6h_0h",
)
LABEL_DISPLAY = {
    "gene_sense_late_chase_6h_2h": "gene 6h/2h",
    "gene_sense_total_chase_6h_0h": "gene 6h/0h",
    "exon_sense_late_chase_6h_2h": "exon 6h/2h",
    "exon_sense_total_chase_6h_0h": "exon 6h/0h",
}
METADATA_COLUMNS = {
    "gene_id",
    "gene_symbol",
    "canonical_transcript_id",
    "chromosome",
    "strand",
    "gene_biotype",
    "transcript_biotype",
    "sequence_status",
    "replicate_qc_flag",
    "target_feature_type",
    "target_label_key",
    "target_label_id",
    "target_label",
}
REGION_ORDER = ["5utr", "cds", "3utr", "full"]
CLASS_ORDER = ["length", "composition", "kmer3", "kmer4", "motif"]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    processed = root / "data/processed"
    source = processed / "figure_source_data"
    source.mkdir(parents=True, exist_ok=True)

    feature_correlations = compute_feature_correlations(processed)
    elasticnet = load_elasticnet_importances(processed)
    region_class = summarize_region_class_signal(feature_correlations, elasticnet)
    stable_features = summarize_cross_label_features(feature_correlations, elasticnet)
    motif_summary = summarize_motif_signal(feature_correlations, elasticnet)
    simple_signal = summarize_simple_signal(feature_correlations)

    feature_correlations.to_csv(
        processed / "biological_feature_target_correlations.tsv", sep="\t", index=False
    )
    region_class.to_csv(processed / "biological_region_feature_signal.tsv", sep="\t", index=False)
    stable_features.to_csv(
        processed / "biological_cross_label_candidate_features.tsv", sep="\t", index=False
    )
    motif_summary.to_csv(processed / "biological_motif_signal.tsv", sep="\t", index=False)
    simple_signal.to_csv(processed / "biological_length_composition_signal.tsv", sep="\t", index=False)

    for name, frame in [
        ("biological_region_feature_signal.tsv", region_class),
        ("biological_cross_label_candidate_features.tsv", stable_features),
        ("biological_motif_signal.tsv", motif_summary),
        ("biological_length_composition_signal.tsv", simple_signal),
    ]:
        frame.to_csv(source / name, sep="\t", index=False)

    make_figures(root, region_class, stable_features, motif_summary, simple_signal)
    write_report(root, processed, region_class, stable_features, motif_summary, simple_signal)


def compute_feature_correlations(processed: Path) -> pd.DataFrame:
    rows = []
    for label_id in LABELS:
        path = processed / f"parallel_sequence_model_features_{label_id}.tsv"
        data = pd.read_csv(path, sep="\t")
        numeric_columns = [
            column
            for column in data.select_dtypes(include=[np.number]).columns
            if column not in METADATA_COLUMNS
        ]
        ranked_features = data[numeric_columns].rank(axis=0)
        ranked_target = data["target_label"].rank()
        correlations = ranked_features.corrwith(ranked_target)
        for feature, value in correlations.items():
            parsed = parse_feature(feature)
            rows.append(
                {
                    "label_id": label_id,
                    "label_display": LABEL_DISPLAY[label_id],
                    "feature": feature,
                    "spearman": float(value) if pd.notna(value) else np.nan,
                    "spearman_abs": abs(float(value)) if pd.notna(value) else np.nan,
                    **parsed,
                }
            )
    return pd.DataFrame(rows).dropna(subset=["spearman"])


def load_elasticnet_importances(processed: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(processed.glob("parallel_eval_*_elasticnet_importance.tsv")):
        label_id = strip_prefix_suffix(
            path.name, "parallel_eval_", "_elasticnet_importance.tsv"
        )
        frame = pd.read_csv(path, sep="\t")
        frame["label_id"] = label_id
        frame["label_display"] = LABEL_DISPLAY[label_id]
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    parsed = data["feature"].apply(parse_feature).apply(pd.Series)
    data = pd.concat([data, parsed], axis=1)
    data["coefficient_sign"] = np.sign(data["importance"])
    return data


def summarize_region_class_signal(
    correlations: pd.DataFrame, elasticnet: pd.DataFrame
) -> pd.DataFrame:
    corr_rows = []
    for keys, group in correlations.groupby(["label_id", "region", "feature_class"], dropna=False):
        ranked = group.sort_values("spearman_abs", ascending=False)
        corr_rows.append(
            {
                "label_id": keys[0],
                "label_display": LABEL_DISPLAY[keys[0]],
                "region": keys[1],
                "feature_class": keys[2],
                "n_features": len(group),
                "mean_abs_spearman": group["spearman_abs"].mean(),
                "top20_mean_abs_spearman": ranked["spearman_abs"].head(20).mean(),
                "max_abs_spearman": group["spearman_abs"].max(),
            }
        )
    output = pd.DataFrame(corr_rows)
    if elasticnet.empty:
        output["mean_abs_elasticnet_coefficient"] = np.nan
        return output
    elastic = (
        elasticnet.groupby(["label_id", "region", "feature_class"], dropna=False)
        .agg(mean_abs_elasticnet_coefficient=("importance_abs", "mean"))
        .reset_index()
    )
    return output.merge(elastic, on=["label_id", "region", "feature_class"], how="left")


def summarize_cross_label_features(
    correlations: pd.DataFrame, elasticnet: pd.DataFrame
) -> pd.DataFrame:
    pivot = correlations.pivot_table(index="feature", columns="label_id", values="spearman")
    parsed = pd.DataFrame([parse_feature(feature) for feature in pivot.index], index=pivot.index)
    signs = np.sign(pivot)
    positive_fraction = (signs > 0).sum(axis=1) / signs.notna().sum(axis=1)
    negative_fraction = (signs < 0).sum(axis=1) / signs.notna().sum(axis=1)
    out = parsed.reset_index(drop=True).copy()
    out["feature"] = pivot.index
    out["n_labels"] = pivot.notna().sum(axis=1).to_numpy()
    out["mean_spearman"] = pivot.mean(axis=1).to_numpy()
    out["mean_abs_spearman"] = pivot.abs().mean(axis=1).to_numpy()
    out["max_abs_spearman"] = pivot.abs().max(axis=1).to_numpy()
    out["sign_concordance"] = np.maximum(positive_fraction, negative_fraction).to_numpy()
    out["dominant_direction"] = np.where(positive_fraction >= negative_fraction, "positive", "negative")
    for label_id in LABELS:
        out[f"spearman_{label_id}"] = pivot[label_id].to_numpy()
    if not elasticnet.empty:
        elastic = (
            elasticnet.groupby("feature")
            .agg(
                mean_elasticnet_coefficient=("importance", "mean"),
                mean_abs_elasticnet_coefficient=("importance_abs", "mean"),
            )
            .reset_index()
        )
        out = out.merge(elastic, on="feature", how="left")
    return out.sort_values(["mean_abs_spearman", "sign_concordance"], ascending=False).reset_index(
        drop=True
    )


def summarize_motif_signal(correlations: pd.DataFrame, elasticnet: pd.DataFrame) -> pd.DataFrame:
    motifs = correlations[correlations["feature_class"] == "motif"].copy()
    rows = []
    for keys, group in motifs.groupby(["label_id", "region", "element"], dropna=False):
        rows.append(
            {
                "label_id": keys[0],
                "label_display": LABEL_DISPLAY[keys[0]],
                "region": keys[1],
                "motif": keys[2],
                "n_features": len(group),
                "mean_spearman": group["spearman"].mean(),
                "mean_abs_spearman": group["spearman_abs"].mean(),
                "max_abs_spearman": group["spearman_abs"].max(),
            }
        )
    out = pd.DataFrame(rows)
    if not elasticnet.empty:
        elastic_motifs = elasticnet[elasticnet["feature_class"] == "motif"]
        elastic = (
            elastic_motifs.groupby(["label_id", "region", "element"], dropna=False)
            .agg(mean_abs_elasticnet_coefficient=("importance_abs", "mean"))
            .reset_index()
            .rename(columns={"element": "motif"})
        )
        out = out.merge(elastic, on=["label_id", "region", "motif"], how="left")
    return out.sort_values(["mean_abs_spearman", "max_abs_spearman"], ascending=False).reset_index(
        drop=True
    )


def summarize_simple_signal(correlations: pd.DataFrame) -> pd.DataFrame:
    simple = correlations[correlations["feature_class"].isin(["length", "composition"])].copy()
    return simple.sort_values("spearman_abs", ascending=False).reset_index(drop=True)


def parse_feature(feature: str) -> dict[str, object]:
    region = "other"
    remainder = feature
    for candidate in ["5utr", "3utr", "cds", "full"]:
        prefix = f"{candidate}_"
        if feature.startswith(prefix):
            region = candidate
            remainder = feature[len(prefix) :]
            break
    feature_class = "other"
    element = remainder
    kmer_size = np.nan
    measurement = ""
    if remainder == "length":
        feature_class = "length"
        element = "length"
    elif remainder in {"gc_fraction", "au_fraction", "u_fraction"}:
        feature_class = "composition"
        element = remainder.replace("_fraction", "")
    elif remainder.startswith("kmer_"):
        kmer = remainder[len("kmer_") :]
        feature_class = f"kmer{len(kmer)}"
        element = kmer
        kmer_size = len(kmer)
    elif remainder.startswith("motif_"):
        motif = remainder[len("motif_") :]
        if motif.endswith("_count"):
            motif = motif[: -len("_count")]
            measurement = "count"
        elif motif.endswith("_per_kb"):
            motif = motif[: -len("_per_kb")]
            measurement = "per_kb"
        feature_class = "motif"
        element = motif
    return {
        "region": region,
        "feature_class": feature_class,
        "element": element,
        "kmer_size": kmer_size,
        "measurement": measurement,
    }


def strip_prefix_suffix(text: str, prefix: str, suffix: str) -> str:
    if text.startswith(prefix):
        text = text[len(prefix) :]
    if text.endswith(suffix):
        text = text[: -len(suffix)]
    return text


def make_figures(
    root: Path,
    region_class: pd.DataFrame,
    stable_features: pd.DataFrame,
    motif_summary: pd.DataFrame,
    simple_signal: pd.DataFrame,
) -> None:
    figure_dir = root / "docs/figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_region_class_heatmap(figure_dir, region_class)
    plot_top_feature_heatmap(figure_dir, stable_features)
    plot_simple_signal_heatmap(figure_dir, simple_signal)
    plot_motif_signal_heatmap(figure_dir, motif_summary)


def plot_region_class_heatmap(figure_dir: Path, region_class: pd.DataFrame) -> None:
    data = (
        region_class.groupby(["region", "feature_class"])["top20_mean_abs_spearman"]
        .mean()
        .unstack()
        .reindex(index=REGION_ORDER, columns=CLASS_ORDER)
    )
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    image = ax.imshow(data.fillna(0), cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(["5'UTR", "CDS", "3'UTR", "full"])
    for row in range(len(data.index)):
        for col in range(len(data.columns)):
            value = data.iloc[row, col]
            if pd.notna(value):
                ax.text(col, row, f"{value:.3f}", ha="center", va="center", fontsize=8)
    ax.set_title("Region x feature-class target association")
    fig.colorbar(image, ax=ax, shrink=0.8, label="Top-20 mean |Spearman|")
    save_figure(fig, figure_dir, "biological_region_feature_signal")


def plot_top_feature_heatmap(figure_dir: Path, stable_features: pd.DataFrame) -> None:
    top = stable_features.head(25).copy()
    matrix = top[[f"spearman_{label}" for label in LABELS]].to_numpy()
    labels = [
        f"{row.region}:{row.element}" if row.feature_class.startswith("kmer") else row.feature
        for row in top.itertuples()
    ]
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    image = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-0.22, vmax=0.22)
    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels([LABEL_DISPLAY[label] for label in LABELS], rotation=35, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Top cross-label sequence-feature associations")
    fig.colorbar(image, ax=ax, shrink=0.8, label="Spearman with stability proxy")
    save_figure(fig, figure_dir, "biological_top_feature_heatmap")


def plot_simple_signal_heatmap(figure_dir: Path, simple_signal: pd.DataFrame) -> None:
    keep = simple_signal.copy()
    keep["feature_label"] = keep["region"] + ":" + keep["element"]
    order = (
        keep.groupby("feature_label")["spearman_abs"]
        .mean()
        .sort_values(ascending=False)
        .head(24)
        .index
    )
    table = (
        keep[keep["feature_label"].isin(order)]
        .pivot_table(index="feature_label", columns="label_id", values="spearman")
        .reindex(index=order, columns=LABELS)
    )
    fig, ax = plt.subplots(figsize=(7.5, 7), constrained_layout=True)
    image = ax.imshow(table, cmap="RdBu_r", aspect="auto", vmin=-0.35, vmax=0.35)
    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels([LABEL_DISPLAY[label] for label in LABELS], rotation=35, ha="right")
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index, fontsize=8)
    ax.set_title("Length and base-composition associations")
    fig.colorbar(image, ax=ax, shrink=0.8, label="Spearman with stability proxy")
    save_figure(fig, figure_dir, "biological_length_composition_signal")


def plot_motif_signal_heatmap(figure_dir: Path, motif_summary: pd.DataFrame) -> None:
    motif_summary = motif_summary.copy()
    motif_summary["feature_label"] = motif_summary["region"] + ":" + motif_summary["motif"]
    order = (
        motif_summary.groupby("feature_label")["mean_abs_spearman"]
        .mean()
        .sort_values(ascending=False)
        .head(24)
        .index
    )
    table = (
        motif_summary[motif_summary["feature_label"].isin(order)]
        .pivot_table(index="feature_label", columns="label_id", values="mean_spearman")
        .reindex(index=order, columns=LABELS)
    )
    fig, ax = plt.subplots(figsize=(7.5, 7), constrained_layout=True)
    image = ax.imshow(table, cmap="RdBu_r", aspect="auto", vmin=-0.20, vmax=0.20)
    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels([LABEL_DISPLAY[label] for label in LABELS], rotation=35, ha="right")
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index, fontsize=8)
    ax.set_title("Seed motif and AU-rich feature associations")
    fig.colorbar(image, ax=ax, shrink=0.8, label="Mean Spearman")
    save_figure(fig, figure_dir, "biological_motif_signal")


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    for suffix in ["png", "svg", "pdf"]:
        fig.savefig(directory / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(
    root: Path,
    processed: Path,
    region_class: pd.DataFrame,
    stable_features: pd.DataFrame,
    motif_summary: pd.DataFrame,
    simple_signal: pd.DataFrame,
) -> None:
    input_ablation = pd.read_csv(processed / "input_ablation_paired_differences.tsv", sep="\t")
    deep_ablation = pd.read_csv(processed / "deep_input_ablation_paired_differences.tsv", sep="\t")
    chromosome_input = input_ablation[input_ablation["evaluation"] == "chromosome_holdout"]
    chromosome_deep = deep_ablation[deep_ablation["evaluation"] == "chromosome_holdout"]
    input_effects = chromosome_input.groupby("feature_set")["pearson_delta_mean"].mean()
    deep_effects = chromosome_deep.groupby("input_condition")["pearson_delta_mean"].mean()

    top_region = (
        region_class.groupby("region")["top20_mean_abs_spearman"]
        .mean()
        .sort_values(ascending=False)
    )
    top_class = (
        region_class.groupby("feature_class")["top20_mean_abs_spearman"]
        .mean()
        .sort_values(ascending=False)
    )
    top_features = stable_features.head(12)
    strongest_simple = simple_signal.iloc[0]
    strongest_motif = motif_summary.iloc[0]

    lines = [
        "# Biological Interpretation of Sequence Signals",
        "",
        "This report summarizes interpretable evidence from engineered-feature correlations, "
        "ElasticNet coefficients, XGBoost input ablations, and deep raw-sequence region ablations.",
        "",
        "## Main Biological Readout",
        "",
        f"- In univariate feature-target associations, the strongest regional signal is "
        f"`{top_region.index[0]}`, followed by `{top_region.index[1]}`. This mostly reflects "
        "GC-rich and C/G-rich k-mer features, so it should be read as an association map, "
        "not as a causal region ranking.",
        f"- Predictive ablations give the stronger region-level evidence: removing CDS is the "
        f"most damaging deep-sequence perturbation "
        f"({deep_effects.get('raw_no_cds', np.nan):+.3f} mean paired Pearson), and "
        f"`structured_no_cds` is the most damaging engineered-feature leave-one-region-out "
        f"condition ({input_effects.get('structured_no_cds', np.nan):+.3f}).",
        f"- The strongest interpretable feature class by association is `{top_class.index[0]}`. "
        "The signal is therefore better described as distributed sequence grammar "
        "than as one small hand-picked motif panel.",
        f"- `cds_only` remains close to the full engineered-feature model "
        f"({input_effects.get('cds_only', np.nan):+.3f} mean paired Pearson versus all regions), "
        "which supports CDS as the main compact sequence-information carrier.",
        f"- The strongest simple length/composition association is "
        f"`{strongest_simple.feature}` in `{strongest_simple.label_display}` "
        f"(Spearman {strongest_simple.spearman:+.3f}), which is especially important when "
        "interpreting the highly predictable exon-sense 6h/0h label.",
        f"- The strongest current motif-panel association is `{strongest_motif.region}:"
        f"{strongest_motif.motif}` in `{strongest_motif.label_display}` "
        f"(mean Spearman {strongest_motif.mean_spearman:+.3f}). Motif-only ablation remains weak, "
        "so these motif hits should be treated as hypotheses rather than final mechanisms.",
        "",
        "![Region feature signal](figures/biological_region_feature_signal.png)",
        "",
        "## Candidate Cross-Label Features",
        "",
        "| Rank | Feature | Region | Class | Direction | Mean abs Spearman | Sign concordance |",
        "| ---: | --- | --- | --- | --- | ---: | ---: |",
    ]
    for index, row in top_features.iterrows():
        lines.append(
            f"| {index + 1} | `{row.feature}` | `{row.region}` | `{row.feature_class}` | "
            f"{row.dominant_direction} | {row.mean_abs_spearman:.3f} | "
            f"{row.sign_concordance:.2f} |"
        )
    lines.extend(
        [
            "",
            "![Top feature heatmap](figures/biological_top_feature_heatmap.png)",
            "",
            "## Interpretation",
            "",
            "- Positive Spearman values mean the feature is associated with higher 6h retention "
            "relative to 2h or 0h, i.e. a higher stability proxy. Negative values mean the "
            "feature is associated with lower retention.",
            "- CDS-dominant signal is compatible with codon-usage, amino-acid/codon composition, "
            "ribosome-linked decay, mRNA surveillance, and coding-region RBP binding hypotheses. "
            "The current experiment does not distinguish these mechanisms yet.",
            "- Strong length/composition signal, especially for exon-sense 6h/0h, supports the "
            "existing caution that this label may include processing, mature-RNA retention, or "
            "abundance-linked correlates in addition to degradation.",
            "- The current motif panel is intentionally small. Weak motif-only performance does "
            "not rule out RBP or miRNA mechanisms; it mainly says this small panel is not a "
            "sufficient representation of the predictive grammar.",
            "",
            "![Length composition signal](figures/biological_length_composition_signal.png)",
            "",
            "![Motif signal](figures/biological_motif_signal.png)",
            "",
            "## Most Important Remaining Analyses",
            "",
            "1. Run SHAP or permutation importance for XGBoost to separate correlated k-mers, "
            "length, and GC effects.",
            "2. Run in-silico mutagenesis on the Transformer hybrid model for high-confidence "
            "genes to localize sequence positions, not only feature families.",
            "3. Add codon-aware features such as codon frequency, amino-acid composition, CAI/tAI, "
            "stop-codon context, upstream/downstream codon windows, and codon-pair statistics.",
            "4. Expand the motif library with RBP motifs and miRNA seeds, then evaluate motif "
            "families rather than individual toy motifs.",
            "5. Test whether candidate signals remain after controlling for expression level, "
            "transcript length, CDS length, GC, and gene biotype.",
            "",
            "## Outputs",
            "",
            "- `data/processed/biological_feature_target_correlations.tsv`",
            "- `data/processed/biological_region_feature_signal.tsv`",
            "- `data/processed/biological_cross_label_candidate_features.tsv`",
            "- `data/processed/biological_motif_signal.tsv`",
            "- `data/processed/biological_length_composition_signal.tsv`",
            "- `docs/figures/biological_region_feature_signal.{png,svg,pdf}`",
            "- `docs/figures/biological_top_feature_heatmap.{png,svg,pdf}`",
            "- `docs/figures/biological_length_composition_signal.{png,svg,pdf}`",
            "- `docs/figures/biological_motif_signal.{png,svg,pdf}`",
        ]
    )
    (root / "docs/biological_interpretation_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
