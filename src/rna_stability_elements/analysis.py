from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from rna_stability_elements.quant import load_gene_signal_table

DEFAULT_TARGET_COLUMNS = [
    "log2_stability_2h_0h",
    "log2_stability_6h_2h",
    "log2_stability_6h_0h",
]


def summarize_targets(
    targets: pd.DataFrame,
    *,
    target_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize stability target coverage and cross-cell-line variability."""
    target_columns = target_columns or [c for c in DEFAULT_TARGET_COLUMNS if c in targets]
    missing = [column for column in target_columns if column not in targets]
    if missing:
        raise ValueError(f"Target columns missing from input: {missing}")

    cell_agg = {
        "gene_id": pd.NamedAgg(column="gene_id", aggfunc="nunique"),
        "signal_0h_median": pd.NamedAgg(column="signal_0h", aggfunc="median"),
        "signal_2h_median": pd.NamedAgg(column="signal_2h", aggfunc="median"),
        "signal_6h_median": pd.NamedAgg(column="signal_6h", aggfunc="median"),
    }
    for column in target_columns:
        cell_agg[f"{column}_median"] = pd.NamedAgg(column=column, aggfunc="median")
        cell_agg[f"{column}_iqr"] = pd.NamedAgg(column=column, aggfunc=_iqr)
    cell_summary = (
        targets.groupby("cell_line", as_index=False)
        .agg(**cell_agg)
        .rename(columns={"gene_id": "n_genes"})
        .sort_values("cell_line")
    )

    gene_agg = {
        "cell_line": pd.NamedAgg(column="cell_line", aggfunc="nunique"),
        "signal_0h_mean": pd.NamedAgg(column="signal_0h", aggfunc="mean"),
    }
    for column in target_columns:
        gene_agg[f"{column}_mean"] = pd.NamedAgg(column=column, aggfunc="mean")
        gene_agg[f"{column}_std"] = pd.NamedAgg(column=column, aggfunc="std")
        gene_agg[f"{column}_min"] = pd.NamedAgg(column=column, aggfunc="min")
        gene_agg[f"{column}_max"] = pd.NamedAgg(column=column, aggfunc="max")
        gene_agg[f"{column}_range"] = pd.NamedAgg(column=column, aggfunc=lambda x: x.max() - x.min())
    gene_summary = (
        targets.groupby(["gene_id", "gene_symbol"], as_index=False)
        .agg(**gene_agg)
        .rename(columns={"cell_line": "n_cell_lines"})
    )
    sort_column = (
        "log2_stability_6h_2h_std"
        if "log2_stability_6h_2h_std" in gene_summary
        else f"{target_columns[0]}_std"
    )
    gene_summary = gene_summary.sort_values(sort_column, ascending=False).reset_index(drop=True)
    return cell_summary, gene_summary


def write_target_summaries(
    targets_path: str | Path,
    *,
    cell_out: str | Path,
    gene_out: str | Path,
    target_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = pd.read_csv(targets_path, sep="\t")
    cell_summary, gene_summary = summarize_targets(targets, target_columns=target_columns)
    Path(cell_out).parent.mkdir(parents=True, exist_ok=True)
    Path(gene_out).parent.mkdir(parents=True, exist_ok=True)
    cell_summary.to_csv(cell_out, sep="\t", index=False)
    gene_summary.to_csv(gene_out, sep="\t", index=False)
    return cell_summary, gene_summary


def build_consensus_targets(
    targets: pd.DataFrame,
    *,
    target_column: str = "log2_stability_6h_2h",
    min_cell_lines: int = 8,
) -> pd.DataFrame:
    """Collapse gene x cell-line targets to one context-agnostic target per gene."""
    required = {"gene_id", "gene_symbol", "cell_line", target_column}
    missing = required - set(targets.columns)
    if missing:
        raise ValueError(f"Targets missing required columns: {sorted(missing)}")

    data = targets.dropna(subset=[target_column]).copy()
    grouped = (
        data.groupby(["gene_id", "gene_symbol"], as_index=False)
        .agg(
            n_cell_lines=("cell_line", "nunique"),
            stability_consensus_median=(target_column, "median"),
            stability_consensus_mean=(target_column, "mean"),
            stability_consensus_std=(target_column, "std"),
            stability_consensus_iqr=(target_column, _iqr),
            stability_consensus_min=(target_column, "min"),
            stability_consensus_max=(target_column, "max"),
            signal_0h_median=("signal_0h", "median"),
            signal_2h_median=("signal_2h", "median"),
            signal_6h_median=("signal_6h", "median"),
        )
        .copy()
    )
    grouped["target_source_column"] = target_column
    grouped["label_quality_flag"] = "pass"
    grouped.loc[grouped["n_cell_lines"] < min_cell_lines, "label_quality_flag"] = "low_cell_line_coverage"
    grouped = grouped[grouped["n_cell_lines"] >= min_cell_lines].copy()
    return grouped.sort_values(["gene_id"]).reset_index(drop=True)


def write_consensus_targets(
    targets_path: str | Path,
    *,
    out: str | Path,
    target_column: str = "log2_stability_6h_2h",
    min_cell_lines: int = 8,
) -> pd.DataFrame:
    targets = pd.read_csv(targets_path, sep="\t")
    consensus = build_consensus_targets(
        targets,
        target_column=target_column,
        min_cell_lines=min_cell_lines,
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    consensus.to_csv(out, sep="\t", index=False)
    return consensus


def compare_target_sets(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_name: str = "left",
    right_name: str = "right",
    target_column: str = "log2_stability_6h_2h",
) -> pd.DataFrame:
    """Compare two gene x cell-line target tables on shared rows."""
    keys = ["gene_id", "gene_symbol", "cell_line"]
    required = set(keys + [target_column])
    missing_left = required - set(left.columns)
    missing_right = required - set(right.columns)
    if missing_left:
        raise ValueError(f"Left targets missing required columns: {sorted(missing_left)}")
    if missing_right:
        raise ValueError(f"Right targets missing required columns: {sorted(missing_right)}")

    merged = left[keys + [target_column]].merge(
        right[keys + [target_column]],
        on=keys,
        suffixes=(f"_{left_name}", f"_{right_name}"),
        how="inner",
    )
    left_col = f"{target_column}_{left_name}"
    right_col = f"{target_column}_{right_name}"
    merged["delta"] = merged[right_col] - merged[left_col]

    rows = []
    for cell_line, group in merged.groupby("cell_line"):
        rows.append(_comparison_row(cell_line, group, left_col, right_col))
    rows.append(_comparison_row("ALL", merged, left_col, right_col))
    summary = pd.DataFrame(rows)
    return summary.sort_values("cell_line").reset_index(drop=True)


def write_target_comparison(
    left_path: str | Path,
    right_path: str | Path,
    *,
    out: str | Path,
    left_name: str = "left",
    right_name: str = "right",
    target_column: str = "log2_stability_6h_2h",
) -> pd.DataFrame:
    left = pd.read_csv(left_path, sep="\t")
    right = pd.read_csv(right_path, sep="\t")
    summary = compare_target_sets(
        left,
        right,
        left_name=left_name,
        right_name=right_name,
        target_column=target_column,
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, sep="\t", index=False)
    return summary


def build_replicate_qc(
    signal: pd.DataFrame,
    *,
    value_column: str = "rpkm",
    pseudocount: float = 0.1,
    min_mean_signal: float = 0.5,
    max_log2_range: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize biological replicate agreement before target construction."""
    required = {"gene_id", "gene_symbol", "cell_line", "time_h", "file_accession", value_column}
    missing = required - set(signal.columns)
    if missing:
        raise ValueError(f"Signal table missing required columns: {sorted(missing)}")

    grouped = (
        signal.groupby(["gene_id", "gene_symbol", "cell_line", "time_h"], as_index=False)
        .agg(
            n_replicates=("file_accession", "nunique"),
            mean_signal=(value_column, "mean"),
            std_signal=(value_column, "std"),
            min_signal=(value_column, "min"),
            max_signal=(value_column, "max"),
            mean_raw_count=("raw_count", "mean"),
        )
        .copy()
    )
    grouped["replicate_cv"] = grouped["std_signal"] / grouped["mean_signal"].replace(0, np.nan)
    grouped["replicate_log2_range"] = np.log2(
        (grouped["max_signal"] + pseudocount) / (grouped["min_signal"] + pseudocount)
    )
    grouped["replicate_quality_flag"] = "pass"
    grouped.loc[grouped["n_replicates"] < 2, "replicate_quality_flag"] = "missing_replicate"
    grouped.loc[grouped["mean_signal"] < min_mean_signal, "replicate_quality_flag"] = "low_signal"
    grouped.loc[
        (grouped["n_replicates"] >= 2)
        & (grouped["mean_signal"] >= min_mean_signal)
        & (grouped["replicate_log2_range"] > max_log2_range),
        "replicate_quality_flag",
    ] = "high_replicate_discordance"

    experiment_rows = []
    for (cell_line, time_h), group in signal.groupby(["cell_line", "time_h"]):
        experiment_rows.append(_replicate_experiment_row(group, cell_line, time_h, value_column, min_mean_signal))
    experiment_qc = pd.DataFrame(experiment_rows).sort_values(["cell_line", "time_h"]).reset_index(drop=True)
    gene_qc = grouped.sort_values(["cell_line", "time_h", "gene_id"]).reset_index(drop=True)
    return gene_qc, experiment_qc


def write_replicate_qc(
    manifest_path: str | Path,
    *,
    gene_out: str | Path,
    experiment_out: str | Path,
    feature_type: str = "gene_sense",
    value_column: str = "rpkm",
    pseudocount: float = 0.1,
    min_mean_signal: float = 0.5,
    max_log2_range: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(manifest_path, sep="\t")
    signal = load_gene_signal_table(
        manifest,
        feature_type=feature_type,
        value_column=value_column,
    )
    gene_qc, experiment_qc = build_replicate_qc(
        signal,
        value_column=value_column,
        pseudocount=pseudocount,
        min_mean_signal=min_mean_signal,
        max_log2_range=max_log2_range,
    )
    Path(gene_out).parent.mkdir(parents=True, exist_ok=True)
    Path(experiment_out).parent.mkdir(parents=True, exist_ok=True)
    gene_qc.to_csv(gene_out, sep="\t", index=False)
    experiment_qc.to_csv(experiment_out, sep="\t", index=False)
    return gene_qc, experiment_qc


def build_modeling_master_table(
    consensus: pd.DataFrame,
    *,
    replicate_qc: pd.DataFrame | None = None,
    target_label_column: str = "stability_consensus_median",
    target_label_name: str = "gene_sense_log2_stability_6h_2h_consensus_median",
    source_dataset: str = "ENCODE_Ljungman_BrU_BruChase_gene_sense",
) -> pd.DataFrame:
    """Build the first-stage context-agnostic modeling table."""
    required = {
        "gene_id",
        "gene_symbol",
        target_label_column,
        "stability_consensus_mean",
        "stability_consensus_std",
        "stability_consensus_iqr",
        "n_cell_lines",
        "label_quality_flag",
    }
    missing = required - set(consensus.columns)
    if missing:
        raise ValueError(f"Consensus table missing required columns: {sorted(missing)}")

    table = consensus.copy()
    table.insert(0, "sample_id", table["gene_id"])
    table["target_label"] = table[target_label_column]
    table["target_label_name"] = target_label_name
    table["source_dataset"] = source_dataset
    table["sequence_status"] = "pending_gencode_mapping"

    if replicate_qc is not None:
        replicate_summary = summarize_replicate_qc_by_gene(replicate_qc)
        table = table.merge(replicate_summary, on="gene_id", how="left")

    preferred = [
        "sample_id",
        "gene_id",
        "gene_symbol",
        "target_label",
        "target_label_name",
        "source_dataset",
        "sequence_status",
        "label_quality_flag",
        "n_cell_lines",
        "stability_consensus_median",
        "stability_consensus_mean",
        "stability_consensus_std",
        "stability_consensus_iqr",
        "stability_consensus_min",
        "stability_consensus_max",
        "signal_0h_median",
        "signal_2h_median",
        "signal_6h_median",
        "target_source_column",
    ]
    columns = [column for column in preferred if column in table.columns]
    columns.extend([column for column in table.columns if column not in columns])
    return table[columns].sort_values("gene_id").reset_index(drop=True)


def summarize_replicate_qc_by_gene(replicate_qc: pd.DataFrame) -> pd.DataFrame:
    required = {"gene_id", "replicate_quality_flag", "replicate_log2_range", "replicate_cv", "mean_signal"}
    missing = required - set(replicate_qc.columns)
    if missing:
        raise ValueError(f"Replicate QC table missing required columns: {sorted(missing)}")

    data = replicate_qc.copy()
    data["replicate_pass"] = data["replicate_quality_flag"] == "pass"
    data["replicate_high_discordance"] = data["replicate_quality_flag"] == "high_replicate_discordance"
    summary = (
        data.groupby("gene_id", as_index=False)
        .agg(
            replicate_qc_observations=("replicate_quality_flag", "size"),
            replicate_pass_fraction=("replicate_pass", "mean"),
            replicate_high_discordance_fraction=("replicate_high_discordance", "mean"),
            replicate_log2_range_median=("replicate_log2_range", "median"),
            replicate_log2_range_max=("replicate_log2_range", "max"),
            replicate_cv_median=("replicate_cv", "median"),
            replicate_cv_max=("replicate_cv", "max"),
            replicate_mean_signal_median=("mean_signal", "median"),
        )
        .copy()
    )
    summary["replicate_qc_flag"] = "pass"
    summary.loc[summary["replicate_pass_fraction"] < 0.5, "replicate_qc_flag"] = "review"
    summary.loc[summary["replicate_high_discordance_fraction"] > 0.25, "replicate_qc_flag"] = "high_discordance"
    return summary


def write_modeling_master_table(
    consensus_path: str | Path,
    *,
    out: str | Path,
    replicate_qc_path: str | Path | None = None,
    target_label_column: str = "stability_consensus_median",
    target_label_name: str = "gene_sense_log2_stability_6h_2h_consensus_median",
    source_dataset: str = "ENCODE_Ljungman_BrU_BruChase_gene_sense",
) -> pd.DataFrame:
    consensus = pd.read_csv(consensus_path, sep="\t")
    replicate_qc = pd.read_csv(replicate_qc_path, sep="\t") if replicate_qc_path else None
    table = build_modeling_master_table(
        consensus,
        replicate_qc=replicate_qc,
        target_label_column=target_label_column,
        target_label_name=target_label_name,
        source_dataset=source_dataset,
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, sep="\t", index=False)
    return table


def _comparison_row(cell_line: str, group: pd.DataFrame, left_col: str, right_col: str) -> dict[str, float | str | int]:
    pearson = group[left_col].corr(group[right_col], method="pearson")
    spearman = group[left_col].corr(group[right_col], method="spearman")
    return {
        "cell_line": cell_line,
        "n_shared_rows": int(len(group)),
        "pearson": float(pearson) if pd.notna(pearson) else float("nan"),
        "spearman": float(spearman) if pd.notna(spearman) else float("nan"),
        "delta_median": float(group["delta"].median()),
        "delta_iqr": _iqr(group["delta"]),
        "left_median": float(group[left_col].median()),
        "right_median": float(group[right_col].median()),
    }


def _replicate_experiment_row(
    group: pd.DataFrame,
    cell_line: str,
    time_h: float,
    value_column: str,
    min_mean_signal: float,
) -> dict[str, float | str | int]:
    pivot = group.pivot_table(
        index=["gene_id", "gene_symbol"],
        columns="file_accession",
        values=value_column,
        aggfunc="mean",
    )
    replicate_ids = list(pivot.columns)
    pair_rows = []
    for left, right in combinations(replicate_ids, 2):
        pair = pivot[[left, right]].dropna()
        pair = pair[pair.mean(axis=1) >= min_mean_signal]
        if len(pair) == 0:
            continue
        pair_rows.append(
            {
                "n_shared_genes": int(len(pair)),
                "pearson": pair[left].corr(pair[right], method="pearson"),
                "spearman": pair[left].corr(pair[right], method="spearman"),
            }
        )
    if pair_rows:
        pairs = pd.DataFrame(pair_rows)
        n_shared = int(pairs["n_shared_genes"].median())
        pearson = float(pairs["pearson"].mean())
        spearman = float(pairs["spearman"].mean())
    else:
        n_shared = 0
        pearson = float("nan")
        spearman = float("nan")

    return {
        "cell_line": cell_line,
        "time_h": float(time_h),
        "n_replicates": int(len(replicate_ids)),
        "n_replicate_pairs": int(len(pair_rows)),
        "n_shared_genes_median": n_shared,
        "replicate_pearson_mean": pearson,
        "replicate_spearman_mean": spearman,
    }


def _iqr(values: pd.Series) -> float:
    return float(values.quantile(0.75) - values.quantile(0.25))
