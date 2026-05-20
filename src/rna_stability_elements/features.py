from __future__ import annotations

import gzip
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd


def read_fasta(path: str | Path) -> dict[str, str]:
    """Read a FASTA file into a name -> sequence dictionary."""
    sequences: dict[str, list[str]] = {}
    current: str | None = None
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current = line[1:].split()[0]
                sequences[current] = []
            elif current is not None:
                sequences[current].append(line.upper())
    return {name: "".join(parts) for name, parts in sequences.items()}


def sequence_feature_table(
    fasta_path: str | Path,
    *,
    ks: Iterable[int] = (3, 4, 5, 6),
    motifs: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Compute simple sequence, k-mer, and motif-density features."""
    sequences = read_fasta(fasta_path)
    rows = [
        sequence_features(name, seq, ks=ks, motifs=motifs or {})
        for name, seq in sequences.items()
    ]
    return pd.DataFrame(rows)


def compact_sequence_model_features(
    sequence_table: pd.DataFrame,
    *,
    target_column: str = "target_label",
    regions: Iterable[str] = ("full", "5utr", "cds", "3utr"),
    ks: Iterable[int] = (3, 4),
    motifs: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build a compact numeric feature table from region sequence columns."""
    required = {"gene_id", target_column}
    missing = required - set(sequence_table.columns)
    if missing:
        raise ValueError(f"Sequence table missing required columns: {sorted(missing)}")

    metadata_columns = [
        "gene_id",
        "gene_symbol",
        "canonical_transcript_id",
        "chromosome",
        "strand",
        "gene_biotype",
        "transcript_biotype",
        "sequence_status",
        "replicate_qc_flag",
        target_column,
    ]
    rows = []
    for item in sequence_table.itertuples(index=False):
        row = {
            column: getattr(item, column)
            for column in metadata_columns
            if column in sequence_table.columns
        }
        for region in regions:
            column = f"sequence_{region}"
            if column not in sequence_table.columns:
                continue
            sequence = getattr(item, column)
            if not isinstance(sequence, str):
                sequence = ""
            features = sequence_features(region, sequence, ks=ks, motifs=motifs or {})
            for key, value in features.items():
                if key == "sequence_id":
                    continue
                row[f"{region}_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def write_compact_sequence_model_features(
    sequence_table_path: str | Path,
    *,
    out: str | Path,
    target_column: str = "target_label",
    regions: Iterable[str] = ("full", "5utr", "cds", "3utr"),
    ks: Iterable[int] = (3, 4),
    motifs: dict[str, str] | None = None,
) -> pd.DataFrame:
    sequence_table = pd.read_csv(sequence_table_path, sep="\t")
    features = compact_sequence_model_features(
        sequence_table,
        target_column=target_column,
        regions=regions,
        ks=ks,
        motifs=motifs,
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out, sep="\t", index=False)
    return features


def merge_feature_tables(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    key: str = "gene_id",
    target_column: str = "target_label",
    right_prefix: str = "",
    how: str = "inner",
) -> pd.DataFrame:
    """Merge two gene-level feature tables while keeping one metadata/target copy."""
    if how not in {"inner", "left", "right", "outer"}:
        raise ValueError("how must be one of: inner, left, right, outer")
    if key not in left or key not in right:
        raise ValueError(f"Both feature tables must contain key column {key!r}.")
    if left[key].duplicated().any() or right[key].duplicated().any():
        raise ValueError(f"Feature tables must be unique by {key!r}.")
    metadata_columns = {
        key,
        "gene_symbol",
        "canonical_transcript_id",
        "chromosome",
        "strand",
        "gene_biotype",
        "transcript_biotype",
        "sequence_status",
        "replicate_qc_flag",
        target_column,
    }
    right_feature_columns = [column for column in right.columns if column not in metadata_columns]
    if right_prefix:
        rename = {
            column: f"{right_prefix}{column}"
            for column in right_feature_columns
            if not column.startswith(right_prefix)
        }
        right = right.rename(columns=rename)
        right_feature_columns = [rename.get(column, column) for column in right_feature_columns]
    merged = left.merge(right[[key] + right_feature_columns], on=key, how=how)
    return merged


def write_merged_feature_table(
    left_path: str | Path,
    right_path: str | Path,
    *,
    out: str | Path,
    key: str = "gene_id",
    target_column: str = "target_label",
    right_prefix: str = "",
    how: str = "inner",
) -> pd.DataFrame:
    left = pd.read_csv(left_path, sep="\t")
    right = pd.read_csv(right_path, sep="\t")
    merged = merge_feature_tables(
        left,
        right,
        key=key,
        target_column=target_column,
        right_prefix=right_prefix,
        how=how,
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, sep="\t", index=False)
    return merged


def sequence_features(
    sequence_id: str,
    sequence: str,
    *,
    ks: Iterable[int] = (3, 4, 5, 6),
    motifs: dict[str, str] | None = None,
) -> dict[str, float | str]:
    seq = normalize_rna(sequence)
    length = len(seq)
    counts = Counter(seq)
    denom = max(length, 1)
    features: dict[str, float | str] = {
        "sequence_id": sequence_id,
        "length": float(length),
        "gc_fraction": (counts["G"] + counts["C"]) / denom,
        "au_fraction": (counts["A"] + counts["U"]) / denom,
        "u_fraction": counts["U"] / denom,
    }
    for k in ks:
        features.update(kmer_frequencies(seq, k))
    for name, motif in (motifs or {}).items():
        motif_seq = normalize_rna(motif)
        count = count_overlapping(seq, motif_seq)
        features[f"motif_{name}_count"] = float(count)
        features[f"motif_{name}_per_kb"] = 1000.0 * count / max(length, 1)
    return features


def normalize_rna(sequence: str) -> str:
    return "".join(base for base in sequence.upper().replace("T", "U") if base in {"A", "C", "G", "U"})


def kmer_frequencies(sequence: str, k: int) -> dict[str, float]:
    denom = max(len(sequence) - k + 1, 1)
    counts: Counter[str] = Counter(sequence[index : index + k] for index in range(denom))
    alphabet = "ACGU"
    output: dict[str, float] = {}
    for kmer in _all_kmers(alphabet, k):
        output[f"kmer_{kmer}"] = counts[kmer] / denom
    return output


def count_overlapping(sequence: str, motif: str) -> int:
    if not motif:
        return 0
    count = 0
    start = 0
    while True:
        index = sequence.find(motif, start)
        if index < 0:
            return count
        count += 1
        start = index + 1


def _all_kmers(alphabet: str, k: int) -> Iterable[str]:
    if k == 0:
        yield ""
        return
    for prefix in _all_kmers(alphabet, k - 1):
        for base in alphabet:
            yield prefix + base
