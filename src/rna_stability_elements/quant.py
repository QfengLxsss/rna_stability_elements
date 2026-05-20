from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

GENIC_COLUMNS = [
    "chrom",
    "start",
    "end",
    "gene_id_symbol",
    "score",
    "strand",
    "gene_span",
    "feature_type",
    "reserved",
    "feature_length",
    "raw_count",
    "rpm",
    "rpkm",
]


def read_genic_quantification(
    path: str | Path,
    *,
    feature_type: str | None = "exon_sense",
    value_column: str = "rpkm",
) -> pd.DataFrame:
    """Read ENCODE Bru-seq genic feature quantification TSV files.

    ENCODE files for this pipeline are headerless bed-like TSVs with 13 columns.
    The last three numeric columns are exposed as raw_count, rpm, and rpkm.
    """
    frame = pd.read_csv(path, sep="\t", header=None, names=GENIC_COLUMNS, comment="#")
    if feature_type is not None:
        frame = frame[frame["feature_type"] == feature_type].copy()
    gene_parts = frame["gene_id_symbol"].astype(str).str.split("/", n=1, expand=True)
    frame["gene_id"] = gene_parts[0].str.replace(r"\.\d+$", "", regex=True)
    frame["gene_symbol"] = gene_parts[1].fillna(gene_parts[0]) if gene_parts.shape[1] > 1 else gene_parts[0]
    numeric_columns = [
        "start",
        "end",
        "score",
        "gene_span",
        "reserved",
        "feature_length",
        "raw_count",
        "rpm",
        "rpkm",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if value_column not in frame:
        raise ValueError(f"Unknown value_column={value_column!r}; available columns: {list(frame)}")
    return frame


def aggregate_gene_signal(
    frame: pd.DataFrame,
    *,
    value_column: str = "rpkm",
) -> pd.DataFrame:
    """Aggregate rows to one signal per gene."""
    grouped = (
        frame.groupby(["gene_id", "gene_symbol"], as_index=False)
        .agg(
            raw_count=("raw_count", "sum"),
            feature_length=("feature_length", "sum"),
            value=(value_column, "sum"),
        )
        .rename(columns={"value": value_column})
    )
    return grouped


def load_gene_signal_table(
    file_manifest: pd.DataFrame,
    *,
    feature_type: str = "exon_sense",
    value_column: str = "rpkm",
) -> pd.DataFrame:
    """Read all local quantification files from a manifest into a long signal table."""
    required = {"cell_line", "time_h", "experiment_accession", "file_accession", "local_path"}
    missing = required - set(file_manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    rows: list[pd.DataFrame] = []
    selected = file_manifest[file_manifest["local_path"].astype(str).str.len() > 0].copy()
    if selected.empty:
        raise ValueError("No files with local_path found in manifest.")

    for item in selected.itertuples(index=False):
        path = Path(item.local_path)
        if not path.exists():
            raise FileNotFoundError(path)
        frame = read_genic_quantification(
            path,
            feature_type=feature_type,
            value_column=value_column,
        )
        genes = aggregate_gene_signal(frame, value_column=value_column)
        genes["cell_line"] = item.cell_line
        genes["time_h"] = float(item.time_h)
        genes["experiment_accession"] = item.experiment_accession
        genes["file_accession"] = item.file_accession
        if hasattr(item, "biological_replicates"):
            genes["biological_replicates"] = getattr(item, "biological_replicates")
        rows.append(genes)

    return pd.concat(rows, ignore_index=True)


def build_stability_targets(
    signal: pd.DataFrame,
    *,
    value_column: str = "rpkm",
    pseudocount: float = 0.1,
    min_signal_0h: float = 0.5,
    min_cell_lines_per_gene: int = 1,
) -> pd.DataFrame:
    """Build relative stability targets from 0h, 2h, and 6h gene signals."""
    required = {"gene_id", "gene_symbol", "cell_line", "time_h", value_column}
    missing = required - set(signal.columns)
    if missing:
        raise ValueError(f"Signal table missing required columns: {sorted(missing)}")

    replicated = (
        signal.groupby(["gene_id", "gene_symbol", "cell_line", "time_h"], as_index=False)
        .agg(signal=(value_column, "mean"), raw_count=("raw_count", "mean"))
        .copy()
    )
    wide = replicated.pivot_table(
        index=["gene_id", "gene_symbol", "cell_line"],
        columns="time_h",
        values="signal",
        aggfunc="mean",
    ).reset_index()
    wide.columns.name = None
    for time_h in (0.0, 2.0, 6.0):
        if time_h not in wide.columns:
            wide[time_h] = np.nan
    wide = wide.rename(columns={0.0: "signal_0h", 2.0: "signal_2h", 6.0: "signal_6h"})

    for column in ("signal_0h", "signal_2h", "signal_6h"):
        wide[column] = pd.to_numeric(wide[column], errors="coerce")

    keep = wide["signal_0h"].fillna(0) >= min_signal_0h
    targets = wide.loc[keep].copy()
    targets["log2_stability_2h_0h"] = _log2_ratio(
        targets["signal_2h"], targets["signal_0h"], pseudocount
    )
    targets["log2_stability_6h_2h"] = _log2_ratio(
        targets["signal_6h"], targets["signal_2h"], pseudocount
    )
    targets["log2_stability_6h_0h"] = _log2_ratio(
        targets["signal_6h"], targets["signal_0h"], pseudocount
    )
    targets["decay_rate_0_2h"] = _decay_rate(targets["signal_0h"], targets["signal_2h"], 2.0, pseudocount)
    targets["decay_rate_2_6h"] = _decay_rate(targets["signal_2h"], targets["signal_6h"], 4.0, pseudocount)
    targets["half_life_0_2h"] = _half_life(targets["decay_rate_0_2h"])
    targets["half_life_2_6h"] = _half_life(targets["decay_rate_2_6h"])

    if min_cell_lines_per_gene > 1:
        counts = targets.groupby("gene_id")["cell_line"].nunique()
        keep_genes = counts[counts >= min_cell_lines_per_gene].index
        targets = targets[targets["gene_id"].isin(keep_genes)].copy()

    return targets.sort_values(["cell_line", "gene_id"]).reset_index(drop=True)


def build_targets_from_manifest(
    manifest_path: str | Path,
    *,
    feature_type: str = "exon_sense",
    value_column: str = "rpkm",
    pseudocount: float = 0.1,
    min_signal_0h: float = 0.5,
    min_cell_lines_per_gene: int = 1,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path, sep="\t")
    signal = load_gene_signal_table(
        manifest,
        feature_type=feature_type,
        value_column=value_column,
    )
    return build_stability_targets(
        signal,
        value_column=value_column,
        pseudocount=pseudocount,
        min_signal_0h=min_signal_0h,
        min_cell_lines_per_gene=min_cell_lines_per_gene,
    )


def _log2_ratio(numerator: Iterable[float], denominator: Iterable[float], pseudocount: float) -> pd.Series:
    return np.log2((pd.Series(numerator, dtype=float) + pseudocount) / (pd.Series(denominator, dtype=float) + pseudocount))


def _decay_rate(start: Iterable[float], end: Iterable[float], delta_h: float, pseudocount: float) -> pd.Series:
    start_series = pd.Series(start, dtype=float) + pseudocount
    end_series = pd.Series(end, dtype=float) + pseudocount
    return -np.log(end_series / start_series) / delta_h


def _half_life(decay_rate: Iterable[float]) -> pd.Series:
    rate = pd.Series(decay_rate, dtype=float)
    half_life = np.log(2) / rate
    half_life[rate <= 0] = np.nan
    return half_life
