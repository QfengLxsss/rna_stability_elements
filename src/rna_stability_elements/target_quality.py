from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rna_stability_elements.analysis import build_consensus_targets
from rna_stability_elements.quant import load_gene_signal_table


@dataclass(frozen=True)
class TargetDefinition:
    key: str
    column: str
    numerator: str
    denominator: str
    min_denominator_signal: float


TARGET_DEFINITIONS = [
    TargetDefinition(
        key="late_chase_6h_2h",
        column="log2_stability_6h_2h",
        numerator="signal_6h",
        denominator="signal_2h",
        min_denominator_signal=0.5,
    ),
    TargetDefinition(
        key="total_chase_6h_0h",
        column="log2_stability_6h_0h",
        numerator="signal_6h",
        denominator="signal_0h",
        min_denominator_signal=0.5,
    ),
]


def load_signal_for_feature_type(
    manifest_path: str | Path,
    *,
    feature_type: str,
    value_column: str = "rpkm",
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path, sep="\t")
    signal = load_gene_signal_table(
        manifest,
        feature_type=feature_type,
        value_column=value_column,
    )
    signal["biological_replicates"] = signal["biological_replicates"].astype(str)
    signal["feature_type"] = feature_type
    return signal


def build_replicate_paired_targets(
    signal: pd.DataFrame,
    *,
    pseudocount: float = 0.1,
    min_signal_0h: float = 0.5,
    min_signal_2h: float = 0.5,
) -> pd.DataFrame:
    """Build paired-replicate target values before collapsing replicate evidence.

    The original target builder averages biological replicates before taking time-point ratios.
    This function instead pairs replicate 1 across 0h/2h/6h and replicate 2 across 0h/2h/6h,
    then computes ratios per replicate. Low denominator rows are retained but flagged, so
    downstream consensus tables can use stricter pass-only labels.
    """
    required = {
        "gene_id",
        "gene_symbol",
        "cell_line",
        "time_h",
        "biological_replicates",
        "raw_count",
        "rpkm",
    }
    missing = required - set(signal.columns)
    if missing:
        raise ValueError(f"Signal table missing required columns: {sorted(missing)}")

    grouped = (
        signal.groupby(
            ["gene_id", "gene_symbol", "cell_line", "biological_replicates", "time_h"],
            as_index=False,
        )
        .agg(signal=("rpkm", "mean"), raw_count=("raw_count", "mean"))
        .copy()
    )
    wide = grouped.pivot_table(
        index=["gene_id", "gene_symbol", "cell_line", "biological_replicates"],
        columns="time_h",
        values="signal",
        aggfunc="mean",
    ).reset_index()
    wide.columns.name = None
    for time_h in (0.0, 2.0, 6.0):
        if time_h not in wide.columns:
            wide[time_h] = np.nan
    wide = wide.rename(columns={0.0: "signal_0h", 2.0: "signal_2h", 6.0: "signal_6h"})
    for column in ["signal_0h", "signal_2h", "signal_6h"]:
        wide[column] = pd.to_numeric(wide[column], errors="coerce")

    wide["initial_signal_pass"] = wide["signal_0h"].fillna(0) >= min_signal_0h
    for definition in TARGET_DEFINITIONS:
        if definition.column == "log2_stability_6h_2h":
            denominator_pass = wide[definition.denominator].fillna(0) >= min_signal_2h
        else:
            denominator_pass = wide[definition.denominator].fillna(0) >= definition.min_denominator_signal
        complete = wide[[definition.numerator, definition.denominator]].notna().all(axis=1)
        wide[f"{definition.column}_denominator_pass"] = denominator_pass
        wide[f"{definition.column}_replicate_input_pass"] = wide["initial_signal_pass"] & denominator_pass & complete
        wide[definition.column] = np.log2(
            (wide[definition.numerator] + pseudocount)
            / (wide[definition.denominator] + pseudocount)
        )
    return wide.sort_values(["cell_line", "gene_id", "biological_replicates"]).reset_index(drop=True)


def aggregate_replicate_paired_targets(
    replicate_targets: pd.DataFrame,
    *,
    max_replicate_log2_span: float = 1.0,
) -> pd.DataFrame:
    """Collapse paired replicate targets to gene x cell-line labels with label-specific QC."""
    keys = ["gene_id", "gene_symbol", "cell_line"]
    signal_summary = (
        replicate_targets.groupby(keys, as_index=False)
        .agg(
            signal_0h_median=("signal_0h", "median"),
            signal_2h_median=("signal_2h", "median"),
            signal_6h_median=("signal_6h", "median"),
            n_replicates_total=("biological_replicates", "nunique"),
        )
        .copy()
    )
    output = signal_summary
    for definition in TARGET_DEFINITIONS:
        pass_column = f"{definition.column}_replicate_input_pass"
        data = replicate_targets[replicate_targets[pass_column]].dropna(subset=[definition.column]).copy()
        if data.empty:
            continue
        aggregate = (
            data.groupby(keys, as_index=False)
            .agg(
                **{
                    definition.column: (definition.column, "median"),
                    f"{definition.column}_replicate_mean": (definition.column, "mean"),
                    f"{definition.column}_replicate_std": (definition.column, "std"),
                    f"{definition.column}_replicate_min": (definition.column, "min"),
                    f"{definition.column}_replicate_max": (definition.column, "max"),
                    f"{definition.column}_n_replicates": ("biological_replicates", "nunique"),
                    f"{definition.column}_signal_0h_median": ("signal_0h", "median"),
                    f"{definition.column}_signal_2h_median": ("signal_2h", "median"),
                    f"{definition.column}_signal_6h_median": ("signal_6h", "median"),
                }
            )
            .copy()
        )
        aggregate[f"{definition.column}_replicate_span"] = (
            aggregate[f"{definition.column}_replicate_max"]
            - aggregate[f"{definition.column}_replicate_min"]
        )
        flag = np.full(len(aggregate), "pass", dtype=object)
        flag[aggregate[f"{definition.column}_n_replicates"] < 2] = "single_replicate_after_signal_filter"
        flag[aggregate[f"{definition.column}_replicate_span"] > max_replicate_log2_span] = (
            "high_replicate_target_span"
        )
        aggregate[f"{definition.column}_quality_flag"] = flag
        output = output.merge(aggregate, on=keys, how="left")
    return output.sort_values(["cell_line", "gene_id"]).reset_index(drop=True)


def build_strict_consensus(
    targets: pd.DataFrame,
    *,
    target_column: str,
    min_cell_lines: int = 8,
    pass_only: bool = True,
) -> pd.DataFrame:
    """Build consensus using label-specific target QC flags when available."""
    data = targets.copy()
    for time_label in ["0h", "2h", "6h"]:
        expected = f"signal_{time_label}"
        median_column = f"signal_{time_label}_median"
        if expected not in data and median_column in data:
            data[expected] = data[median_column]
    flag_column = f"{target_column}_quality_flag"
    if pass_only and flag_column in data:
        data.loc[data[flag_column] != "pass", target_column] = np.nan
    consensus = build_consensus_targets(
        data,
        target_column=target_column,
        min_cell_lines=min_cell_lines,
    )
    consensus["consensus_quality_mode"] = "pass_only" if pass_only else "all_available"
    return consensus
