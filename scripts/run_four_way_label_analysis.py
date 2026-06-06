from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from rna_stability_elements.target_quality import (
    TARGET_DEFINITIONS,
    aggregate_replicate_paired_targets,
    build_replicate_paired_targets,
    build_strict_consensus,
    load_signal_for_feature_type,
)


FEATURE_TYPES = ["gene_sense", "exon_sense"]
FEATURE_MANIFESTS = {
    "gene_sense": "encode_gene_quant_files.tsv",
    "exon_sense": "encode_genic_feature_files.tsv",
}
PSEUDOCOUNT = 0.1
MIN_SIGNAL_0H = 0.5
MIN_SIGNAL_2H = 0.5
MIN_CELL_LINES = 8
MAX_REPLICATE_LOG2_SPAN = 1.0


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    processed_dir = root / "data/processed"
    figure_dir = root / "docs/figures"
    base_feature_path = processed_dir / "sequence_model_features.tsv"
    base_features = pd.read_csv(base_feature_path, sep="\t")

    all_quality_rows = []
    all_consistency_rows = []
    all_signal_corr_rows = []
    all_feature_rows = []

    for feature_type in FEATURE_TYPES:
        print(f"[four-way] loading {feature_type} signal")
        manifest_path = root / "data/interim" / FEATURE_MANIFESTS[feature_type]
        signal = load_signal_for_feature_type(manifest_path, feature_type=feature_type)
        write_sample_pca(signal, feature_type=feature_type, processed_dir=processed_dir, figure_dir=figure_dir)

        replicate_targets = build_replicate_paired_targets(
            signal,
            pseudocount=PSEUDOCOUNT,
            min_signal_0h=MIN_SIGNAL_0H,
            min_signal_2h=MIN_SIGNAL_2H,
        )
        replicate_path = processed_dir / f"replicate_paired_targets_{feature_type}.tsv"
        replicate_targets.to_csv(replicate_path, sep="\t", index=False)

        targets = aggregate_replicate_paired_targets(
            replicate_targets,
            max_replicate_log2_span=MAX_REPLICATE_LOG2_SPAN,
        )
        target_path = processed_dir / f"robust_stability_targets_{feature_type}.tsv"
        targets.to_csv(target_path, sep="\t", index=False)

        for definition in TARGET_DEFINITIONS:
            label_id = f"{feature_type}_{definition.key}"
            consensus = build_strict_consensus(
                targets,
                target_column=definition.column,
                min_cell_lines=MIN_CELL_LINES,
                pass_only=True,
            )
            consensus["feature_type"] = feature_type
            consensus["label_key"] = definition.key
            consensus["label_id"] = label_id
            consensus_path = processed_dir / f"robust_consensus_{label_id}.tsv"
            consensus.to_csv(consensus_path, sep="\t", index=False)

            feature_table = make_label_feature_table(
                base_features=base_features,
                consensus=consensus,
                feature_type=feature_type,
                label_key=definition.key,
                label_id=label_id,
            )
            feature_path = processed_dir / f"parallel_sequence_model_features_{label_id}.tsv"
            feature_table.to_csv(feature_path, sep="\t", index=False)
            all_feature_rows.append(
                {
                    "label_id": label_id,
                    "feature_type": feature_type,
                    "label_key": definition.key,
                    "target_column": definition.column,
                    "consensus_rows": int(len(consensus)),
                    "feature_rows": int(len(feature_table)),
                    "feature_path": str(feature_path),
                }
            )

            all_quality_rows.append(
                quality_summary_row(
                    targets=targets,
                    consensus=consensus,
                    feature_table=feature_table,
                    feature_type=feature_type,
                    definition=definition,
                    label_id=label_id,
                )
            )
            all_consistency_rows.extend(
                cross_cell_consistency_rows(
                    targets=targets,
                    feature_type=feature_type,
                    definition=definition,
                    label_id=label_id,
                )
            )
            all_signal_corr_rows.extend(
                signal_correlation_rows(
                    targets=targets,
                    feature_type=feature_type,
                    definition=definition,
                    label_id=label_id,
                )
            )

    pd.DataFrame(all_quality_rows).to_csv(
        processed_dir / "parallel_label_quality_summary.tsv", sep="\t", index=False
    )
    pd.DataFrame(all_consistency_rows).to_csv(
        processed_dir / "parallel_label_cross_cell_consistency.tsv", sep="\t", index=False
    )
    pd.DataFrame(all_signal_corr_rows).to_csv(
        processed_dir / "parallel_label_signal_correlations.tsv", sep="\t", index=False
    )
    pd.DataFrame(all_feature_rows).to_csv(
        processed_dir / "parallel_label_feature_tables.tsv", sep="\t", index=False
    )
    write_parallel_label_qc_figure(processed_dir=processed_dir, figure_dir=figure_dir)


def make_label_feature_table(
    *,
    base_features: pd.DataFrame,
    consensus: pd.DataFrame,
    feature_type: str,
    label_key: str,
    label_id: str,
) -> pd.DataFrame:
    labels = consensus[["gene_id", "stability_consensus_median"]].rename(
        columns={"stability_consensus_median": "target_label"}
    )
    table = base_features.drop(columns=["target_label"], errors="ignore").merge(
        labels,
        on="gene_id",
        how="inner",
    )
    table["target_feature_type"] = feature_type
    table["target_label_key"] = label_key
    table["target_label_id"] = label_id
    preferred = [
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
    ]
    columns = [column for column in preferred if column in table]
    columns.extend([column for column in table.columns if column not in columns])
    return table[columns]


def quality_summary_row(
    *,
    targets: pd.DataFrame,
    consensus: pd.DataFrame,
    feature_table: pd.DataFrame,
    feature_type: str,
    definition,
    label_id: str,
) -> dict[str, float | int | str]:
    column = definition.column
    flag_column = f"{column}_quality_flag"
    values = targets[column].dropna()
    pass_values = targets.loc[targets[flag_column] == "pass", column].dropna()
    flags = targets[flag_column].value_counts(dropna=False)
    return {
        "label_id": label_id,
        "feature_type": feature_type,
        "label_key": definition.key,
        "target_column": column,
        "gene_cell_rows_available": int(values.size),
        "gene_cell_rows_pass": int(pass_values.size),
        "pass_fraction": float(values.size and pass_values.size / values.size),
        "single_replicate_after_signal_filter": int(flags.get("single_replicate_after_signal_filter", 0)),
        "high_replicate_target_span": int(flags.get("high_replicate_target_span", 0)),
        "target_mean_pass": float(pass_values.mean()),
        "target_std_pass": float(pass_values.std()),
        "target_iqr_pass": iqr(pass_values),
        "target_abs_gt_2_fraction_pass": float((pass_values.abs() > 2).mean()),
        "consensus_rows": int(len(consensus)),
        "feature_rows": int(len(feature_table)),
        "consensus_target_std": float(consensus["stability_consensus_median"].std()),
        "consensus_target_iqr": iqr(consensus["stability_consensus_median"]),
    }


def cross_cell_consistency_rows(
    *,
    targets: pd.DataFrame,
    feature_type: str,
    definition,
    label_id: str,
) -> list[dict[str, float | int | str]]:
    column = definition.column
    data = targets[targets[f"{column}_quality_flag"] == "pass"].copy()
    pivot = data.pivot_table(index="gene_id", columns="cell_line", values=column, aggfunc="mean")
    rows = []
    for cell_line in pivot.columns:
        other = pivot.drop(columns=[cell_line]).median(axis=1, skipna=True)
        pair = pd.DataFrame({"cell_line_value": pivot[cell_line], "other_cell_line_median": other}).dropna()
        rows.append(
            {
                "label_id": label_id,
                "feature_type": feature_type,
                "label_key": definition.key,
                "target_column": column,
                "cell_line": cell_line,
                "n_genes": int(len(pair)),
                "pearson": safe_corr(pair, "cell_line_value", "other_cell_line_median", "pearson"),
                "spearman": safe_corr(pair, "cell_line_value", "other_cell_line_median", "spearman"),
            }
        )
    return rows


def signal_correlation_rows(
    *,
    targets: pd.DataFrame,
    feature_type: str,
    definition,
    label_id: str,
) -> list[dict[str, float | str]]:
    column = definition.column
    data = targets[targets[f"{column}_quality_flag"] == "pass"].copy()
    rows = []
    for signal_column in ["signal_0h_median", "signal_2h_median", "signal_6h_median"]:
        data[f"log2_{signal_column}"] = np.log2(data[signal_column] + PSEUDOCOUNT)
        for method in ["pearson", "spearman"]:
            rows.append(
                {
                    "label_id": label_id,
                    "feature_type": feature_type,
                    "label_key": definition.key,
                    "target_column": column,
                    "signal_column": signal_column,
                    "method": method,
                    "correlation": safe_corr(data, column, f"log2_{signal_column}", method),
                }
            )
    return rows


def write_sample_pca(
    signal: pd.DataFrame,
    *,
    feature_type: str,
    processed_dir: Path,
    figure_dir: Path,
    n_variable_genes: int = 5000,
) -> None:
    if signal.empty:
        return
    samples = signal.copy()
    samples["sample_id"] = (
        samples["cell_line"].astype(str)
        + "_"
        + samples["time_h"].astype(float).astype(int).astype(str)
        + "h_rep"
        + samples["biological_replicates"].astype(str)
    )
    matrix = samples.pivot_table(
        index="sample_id",
        columns="gene_id",
        values="rpkm",
        aggfunc="mean",
    )
    matrix = np.log2(matrix + PSEUDOCOUNT)
    variable = matrix.var(axis=0).sort_values(ascending=False).head(n_variable_genes).index
    matrix = matrix[variable]
    if matrix.empty:
        return
    matrix = matrix.fillna(matrix.median(axis=0))
    scaled = StandardScaler().fit_transform(matrix)
    pcs = PCA(n_components=5, random_state=13).fit_transform(scaled)
    metadata = samples[["sample_id", "cell_line", "time_h", "biological_replicates"]].drop_duplicates()
    pca = pd.DataFrame(pcs, index=matrix.index, columns=[f"PC{i}" for i in range(1, 6)]).reset_index()
    pca = pca.merge(metadata, on="sample_id", how="left")
    pca["feature_type"] = feature_type
    pca.to_csv(processed_dir / f"sample_signal_pca_{feature_type}.tsv", sep="\t", index=False)

    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    for time_h, group in pca.groupby("time_h"):
        ax.scatter(group["PC1"], group["PC2"], s=38, alpha=0.82, label=f"{int(time_h)}h")
    ax.set_title(f"Sample signal PCA ({feature_type})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False, title="time")
    fig.tight_layout()
    fig.savefig(figure_dir / f"sample_signal_pca_{feature_type}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_parallel_label_qc_figure(*, processed_dir: Path, figure_dir: Path) -> None:
    quality = pd.read_csv(processed_dir / "parallel_label_quality_summary.tsv", sep="\t")
    consistency = pd.read_csv(processed_dir / "parallel_label_cross_cell_consistency.tsv", sep="\t")
    signal_corr = pd.read_csv(processed_dir / "parallel_label_signal_correlations.tsv", sep="\t")

    order = [
        "gene_sense_late_chase_6h_2h",
        "gene_sense_total_chase_6h_0h",
        "exon_sense_late_chase_6h_2h",
        "exon_sense_total_chase_6h_0h",
    ]
    colors = ["#4477AA", "#CC6677", "#66AA77", "#AA7744"]
    color_map = dict(zip(order, colors))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    ax = axes[0, 0]
    q = quality.set_index("label_id").reindex(order)
    ax.bar(np.arange(len(q)), q["consensus_rows"], color=[color_map[i] for i in q.index])
    ax.set_xticks(np.arange(len(q)))
    ax.set_xticklabels(q.index, rotation=20, ha="right")
    ax.set_ylabel("Consensus genes")
    ax.set_title("Strict pass-only label coverage")

    ax = axes[0, 1]
    ax.bar(np.arange(len(q)), q["pass_fraction"], color=[color_map[i] for i in q.index])
    ax.set_xticks(np.arange(len(q)))
    ax.set_xticklabels(q.index, rotation=20, ha="right")
    ax.set_ylabel("Gene-cell pass fraction")
    ax.set_ylim(0, 1)
    ax.set_title("Replicate/denominator QC pass rate")

    ax = axes[1, 0]
    c = consistency.groupby("label_id")["spearman"].median().reindex(order)
    ax.bar(np.arange(len(c)), c.values, color=[color_map[i] for i in c.index])
    ax.set_xticks(np.arange(len(c)))
    ax.set_xticklabels(c.index, rotation=20, ha="right")
    ax.set_ylabel("Median Spearman")
    ax.set_title("Cell line vs other-cell median")

    ax = axes[1, 1]
    subset = signal_corr[
        (signal_corr["method"] == "spearman")
        & (signal_corr["signal_column"].isin(["signal_0h_median", "signal_2h_median"]))
    ]
    width = 0.18
    x = np.arange(2)
    for index, label_id in enumerate(order):
        vals = subset[subset["label_id"] == label_id].set_index("signal_column").reindex(
            ["signal_0h_median", "signal_2h_median"]
        )
        ax.bar(x + (index - 1.5) * width, vals["correlation"], width=width, color=color_map[label_id], label=label_id)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["0h", "2h"])
    ax.set_ylabel("Spearman")
    ax.set_title("Coupling to input signal")
    ax.legend(frameon=False, fontsize=7, ncol=1)

    fig.tight_layout()
    fig.savefig(figure_dir / "parallel_label_qc_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def safe_corr(frame: pd.DataFrame, left: str, right: str, method: str) -> float:
    data = frame[[left, right]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 3:
        return float("nan")
    return float(data[left].corr(data[right], method=method))


def iqr(values: pd.Series) -> float:
    return float(values.quantile(0.75) - values.quantile(0.25))


if __name__ == "__main__":
    main()
