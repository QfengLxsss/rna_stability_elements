from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rna_stability_elements.analysis import build_consensus_targets
from rna_stability_elements.quant import load_gene_signal_table


LABELS = {
    "late_chase_6h_2h": "log2_stability_6h_2h",
    "total_chase_6h_0h": "log2_stability_6h_0h",
}


def _safe_corr(frame: pd.DataFrame, left: str, right: str, method: str) -> float:
    data = frame[[left, right]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 3:
        return float("nan")
    return float(data[left].corr(data[right], method=method))


def _iqr(values: pd.Series) -> float:
    return float(values.quantile(0.75) - values.quantile(0.25))


def write_label_quality_tables(
    *,
    targets_path: Path,
    manifest_path: Path,
    processed_dir: Path,
    feature_table_path: Path,
    min_signal_0h: float = 0.5,
    pseudocount: float = 0.1,
) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    targets = pd.read_csv(targets_path, sep="\t")

    distribution_rows = []
    signal_corr_rows = []
    consistency_rows = []
    consensus_tables: dict[str, pd.DataFrame] = {}

    for label_name, column in LABELS.items():
        values = targets[column].dropna()
        distribution_rows.append(
            {
                "label_name": label_name,
                "target_column": column,
                "n": int(values.size),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "iqr": _iqr(values),
                "median": float(values.median()),
                "p01": float(values.quantile(0.01)),
                "p05": float(values.quantile(0.05)),
                "p95": float(values.quantile(0.95)),
                "p99": float(values.quantile(0.99)),
                "abs_gt_2_fraction": float((values.abs() > 2).mean()),
            }
        )

        corr_frame = targets[["signal_0h", "signal_2h", "signal_6h", column]].copy()
        for signal_column in ["signal_0h", "signal_2h", "signal_6h"]:
            corr_frame[f"log2_{signal_column}"] = np.log2(corr_frame[signal_column] + pseudocount)
            for method in ["pearson", "spearman"]:
                signal_corr_rows.append(
                    {
                        "label_name": label_name,
                        "target_column": column,
                        "signal_column": signal_column,
                        "method": method,
                        "correlation": _safe_corr(corr_frame, column, f"log2_{signal_column}", method),
                    }
                )

        pivot = targets.pivot_table(index="gene_id", columns="cell_line", values=column, aggfunc="mean")
        for cell_line in pivot.columns:
            other_median = pivot.drop(columns=[cell_line]).median(axis=1, skipna=True)
            pair = pd.DataFrame({"cell_line_value": pivot[cell_line], "other_cell_line_median": other_median}).dropna()
            consistency_rows.append(
                {
                    "label_name": label_name,
                    "target_column": column,
                    "cell_line": cell_line,
                    "n_genes": int(len(pair)),
                    "pearson": float(pair["cell_line_value"].corr(pair["other_cell_line_median"], method="pearson")),
                    "spearman": float(pair["cell_line_value"].corr(pair["other_cell_line_median"], method="spearman")),
                }
            )

        consensus = build_consensus_targets(targets, target_column=column, min_cell_lines=8)
        consensus_tables[label_name] = consensus
        consensus.to_csv(processed_dir / f"consensus_for_{column}.tsv", sep="\t", index=False)

    pd.DataFrame(distribution_rows).to_csv(
        processed_dir / "target_label_choice_distribution.tsv", sep="\t", index=False
    )
    pd.DataFrame(signal_corr_rows).to_csv(
        processed_dir / "target_label_choice_signal_correlations.tsv", sep="\t", index=False
    )
    pd.DataFrame(consistency_rows).to_csv(
        processed_dir / "target_label_choice_cross_cell_consistency.tsv", sep="\t", index=False
    )

    replicate_metrics = _replicate_paired_target_metrics(
        manifest_path=manifest_path,
        min_signal_0h=min_signal_0h,
        pseudocount=pseudocount,
    )
    replicate_metrics.to_csv(
        processed_dir / "target_label_choice_replicate_paired_metrics.tsv", sep="\t", index=False
    )

    _write_parallel_feature_tables(
        feature_table_path=feature_table_path,
        consensus_tables=consensus_tables,
        processed_dir=processed_dir,
    )
    _write_summary_figure(processed_dir)


def _replicate_paired_target_metrics(
    *,
    manifest_path: Path,
    min_signal_0h: float,
    pseudocount: float,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path, sep="\t")
    signal = load_gene_signal_table(manifest, feature_type="gene_sense", value_column="rpkm")
    signal["biological_replicates"] = signal["biological_replicates"].astype(str)
    wide = signal.pivot_table(
        index=["gene_id", "gene_symbol", "cell_line", "biological_replicates"],
        columns="time_h",
        values="rpkm",
        aggfunc="mean",
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={0.0: "signal_0h", 2.0: "signal_2h", 6.0: "signal_6h"})
    wide = wide[wide["signal_0h"].fillna(0) >= min_signal_0h].copy()
    wide["log2_stability_6h_2h"] = np.log2((wide["signal_6h"] + pseudocount) / (wide["signal_2h"] + pseudocount))
    wide["log2_stability_6h_0h"] = np.log2((wide["signal_6h"] + pseudocount) / (wide["signal_0h"] + pseudocount))

    rows = []
    for label_name, column in LABELS.items():
        pivot = wide.pivot_table(
            index=["gene_id", "gene_symbol", "cell_line"],
            columns="biological_replicates",
            values=column,
            aggfunc="mean",
        )
        replicate_columns = [c for c in ["1", "2"] if c in pivot.columns]
        if len(replicate_columns) != 2:
            continue
        pair = pivot[replicate_columns].dropna().rename(columns={"1": "replicate_1", "2": "replicate_2"})
        pair["delta"] = pair["replicate_2"] - pair["replicate_1"]
        rows.append(_replicate_metric_row(label_name, column, "ALL", pair))
        for cell_line, group in pair.reset_index().groupby("cell_line"):
            rows.append(_replicate_metric_row(label_name, column, cell_line, group))
    return pd.DataFrame(rows)


def _replicate_metric_row(label_name: str, column: str, cell_line: str, pair: pd.DataFrame) -> dict[str, float | str | int]:
    return {
        "label_name": label_name,
        "target_column": column,
        "cell_line": cell_line,
        "n_genes": int(len(pair)),
        "pearson": float(pair["replicate_1"].corr(pair["replicate_2"], method="pearson")),
        "spearman": float(pair["replicate_1"].corr(pair["replicate_2"], method="spearman")),
        "median_abs_delta": float(pair["delta"].abs().median()),
        "delta_iqr": _iqr(pair["delta"]),
    }


def _write_parallel_feature_tables(
    *,
    feature_table_path: Path,
    consensus_tables: dict[str, pd.DataFrame],
    processed_dir: Path,
) -> None:
    features = pd.read_csv(feature_table_path, sep="\t")
    metadata_columns = [column for column in features.columns if not _is_feature_value_column(column)]
    feature_columns = [column for column in features.columns if column not in metadata_columns]
    for label_name, consensus in consensus_tables.items():
        table = features[metadata_columns + feature_columns].drop(columns=["target_label"], errors="ignore")
        labels = consensus[["gene_id", "stability_consensus_median"]].rename(
            columns={"stability_consensus_median": "target_label"}
        )
        table = table.merge(labels, on="gene_id", how="inner")
        ordered = metadata_columns.copy()
        if "target_label" not in ordered:
            insert_at = ordered.index("replicate_qc_flag") + 1 if "replicate_qc_flag" in ordered else len(ordered)
            ordered.insert(insert_at, "target_label")
        ordered = [column for column in ordered if column in table]
        ordered.extend([column for column in table.columns if column not in ordered])
        table = table[ordered]
        table.to_csv(
            processed_dir / f"sequence_model_features_{label_name}.tsv",
            sep="\t",
            index=False,
        )


def _is_feature_value_column(column: str) -> bool:
    if column == "target_label":
        return False
    prefixes = (
        "full_",
        "5utr_",
        "cds_",
        "3utr_",
        "lm_",
    )
    return column.startswith(prefixes)


def _write_summary_figure(processed_dir: Path) -> None:
    distribution = pd.read_csv(processed_dir / "target_label_choice_distribution.tsv", sep="\t")
    consistency = pd.read_csv(processed_dir / "target_label_choice_cross_cell_consistency.tsv", sep="\t")
    replicate = pd.read_csv(processed_dir / "target_label_choice_replicate_paired_metrics.tsv", sep="\t")
    signal_corr = pd.read_csv(processed_dir / "target_label_choice_signal_correlations.tsv", sep="\t")

    colors = {"late_chase_6h_2h": "#4477AA", "total_chase_6h_0h": "#CC6677"}
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))

    ax = axes[0, 0]
    x = np.arange(len(distribution))
    ax.bar(x, distribution["std"], color=[colors[v] for v in distribution["label_name"]])
    ax.set_xticks(x)
    ax.set_xticklabels(distribution["label_name"], rotation=15, ha="right")
    ax.set_ylabel("Target std")
    ax.set_title("Target dynamic range")

    ax = axes[0, 1]
    plot = consistency.groupby("label_name")["spearman"].median().reindex(colors.keys())
    ax.bar(np.arange(len(plot)), plot.values, color=[colors[v] for v in plot.index])
    ax.set_xticks(np.arange(len(plot)))
    ax.set_xticklabels(plot.index, rotation=15, ha="right")
    ax.set_ylabel("Median Spearman")
    ax.set_title("Cell line vs other-cell median")

    ax = axes[1, 0]
    all_rep = replicate[replicate["cell_line"] == "ALL"].set_index("label_name").reindex(colors.keys())
    ax.bar(np.arange(len(all_rep)), all_rep["spearman"], color=[colors[v] for v in all_rep.index])
    ax.set_xticks(np.arange(len(all_rep)))
    ax.set_xticklabels(all_rep.index, rotation=15, ha="right")
    ax.set_ylabel("Spearman")
    ax.set_title("Replicate-paired target reliability")

    ax = axes[1, 1]
    subset = signal_corr[(signal_corr["method"] == "spearman") & signal_corr["signal_column"].isin(["signal_0h", "signal_2h"])]
    for offset, label_name in [(-0.18, "late_chase_6h_2h"), (0.18, "total_chase_6h_0h")]:
        vals = subset[subset["label_name"] == label_name].set_index("signal_column").reindex(["signal_0h", "signal_2h"])
        pos = np.arange(len(vals)) + offset
        ax.bar(pos, vals["correlation"], width=0.34, color=colors[label_name], label=label_name)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(["signal_0h", "signal_2h"])
    ax.set_ylabel("Spearman")
    ax.set_title("Coupling to denominator/baseline signal")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    out = processed_dir.parent.parent / "docs" / "figures" / "target_label_choice_qc.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    write_label_quality_tables(
        targets_path=root / "data/processed/stability_targets_gene_sense.tsv",
        manifest_path=root / "data/interim/encode_gene_quant_files.tsv",
        processed_dir=root / "data/processed",
        feature_table_path=root / "data/processed/sequence_model_features.tsv",
    )


if __name__ == "__main__":
    main()
