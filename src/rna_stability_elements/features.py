from __future__ import annotations

import gzip
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd


RNA_GENETIC_CODE = {
    "UUU": "F",
    "UUC": "F",
    "UUA": "L",
    "UUG": "L",
    "UCU": "S",
    "UCC": "S",
    "UCA": "S",
    "UCG": "S",
    "UAU": "Y",
    "UAC": "Y",
    "UAA": "*",
    "UAG": "*",
    "UGU": "C",
    "UGC": "C",
    "UGA": "*",
    "UGG": "W",
    "CUU": "L",
    "CUC": "L",
    "CUA": "L",
    "CUG": "L",
    "CCU": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAU": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGU": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "AUU": "I",
    "AUC": "I",
    "AUA": "I",
    "AUG": "M",
    "ACU": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAU": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGU": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GUU": "V",
    "GUC": "V",
    "GUA": "V",
    "GUG": "V",
    "GCU": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAU": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGU": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}

SYNONYMOUS_CODONS: dict[str, list[str]] = {}
for _codon, _aa in RNA_GENETIC_CODE.items():
    SYNONYMOUS_CODONS.setdefault(_aa, []).append(_codon)

AMINO_ACIDS = sorted(set(RNA_GENETIC_CODE.values()))


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


def codon_feature_table(
    sequence_table: pd.DataFrame,
    *,
    target_column: str = "target_label",
    cds_column: str = "sequence_cds",
) -> pd.DataFrame:
    """Build CDS codon, amino-acid, and reading-frame features."""
    required = {"gene_id", target_column, cds_column}
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
        row.update(codon_features(getattr(item, cds_column)))
        rows.append(row)
    return pd.DataFrame(rows)


def write_codon_feature_table(
    sequence_table_path: str | Path,
    *,
    out: str | Path,
    target_column: str = "target_label",
    cds_column: str = "sequence_cds",
) -> pd.DataFrame:
    sequence_table = pd.read_csv(sequence_table_path, sep="\t")
    features = codon_feature_table(
        sequence_table, target_column=target_column, cds_column=cds_column
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out, sep="\t", index=False)
    return features


def codon_features(sequence: object) -> dict[str, float]:
    """Compute coding-sequence features from frame-0 RNA codons."""
    seq = normalize_rna(sequence if isinstance(sequence, str) else "")
    codons = [seq[index : index + 3] for index in range(0, len(seq) - 2, 3)]
    valid_codons = [codon for codon in codons if len(codon) == 3 and codon in RNA_GENETIC_CODE]
    codon_count = len(valid_codons)
    denom = max(codon_count, 1)
    codon_counts = Counter(valid_codons)
    aa_counts = Counter(RNA_GENETIC_CODE[codon] for codon in valid_codons)
    output: dict[str, float] = {
        "cds_codon_count": float(codon_count),
        "cds_frame_remainder": float(len(seq) % 3),
        "cds_has_start_aug": float(valid_codons[:1] == ["AUG"]),
        "cds_terminal_stop": float(bool(valid_codons) and valid_codons[-1] in {"UAA", "UAG", "UGA"}),
        "cds_internal_stop_fraction": float(
            sum(codon in {"UAA", "UAG", "UGA"} for codon in valid_codons[:-1]) / max(codon_count - 1, 1)
        ),
    }
    for codon in sorted(RNA_GENETIC_CODE):
        output[f"cds_codon_{codon}"] = codon_counts[codon] / denom
    for aa in AMINO_ACIDS:
        output[f"cds_aa_{aa}"] = aa_counts[aa] / denom
    for position in range(3):
        bases = [codon[position] for codon in valid_codons]
        pos_denom = max(len(bases), 1)
        output[f"cds_codon_pos{position + 1}_gc_fraction"] = (
            sum(base in {"G", "C"} for base in bases) / pos_denom
        )
        output[f"cds_codon_pos{position + 1}_u_fraction"] = sum(base == "U" for base in bases) / pos_denom
    output["cds_synonymous_family_entropy"] = synonymous_family_entropy(valid_codons)
    output["cds_mean_synonymous_gc_rank"] = mean_synonymous_gc_rank(valid_codons)
    return output


def synonymous_family_entropy(codons: list[str]) -> float:
    values = []
    for aa, family in SYNONYMOUS_CODONS.items():
        if aa == "*" or len(family) <= 1:
            continue
        counts = [codons.count(codon) for codon in family]
        total = sum(counts)
        if total == 0:
            continue
        probabilities = [count / total for count in counts if count]
        values.append(-sum(value * _log2(value) for value in probabilities))
    return float(sum(values) / max(len(values), 1))


def mean_synonymous_gc_rank(codons: list[str]) -> float:
    ranks = []
    for codon in codons:
        aa = RNA_GENETIC_CODE.get(codon)
        family = SYNONYMOUS_CODONS.get(aa or "", [])
        if len(family) <= 1:
            continue
        ordered = sorted(family, key=lambda item: (gc_count(item), item))
        ranks.append(ordered.index(codon) / max(len(ordered) - 1, 1))
    return float(sum(ranks) / max(len(ranks), 1))


def synonymous_recoded_sequence(sequence: object, *, mode: str) -> str:
    """Recode CDS with synonymous codons while preserving amino-acid sequence."""
    if mode not in {"min_gc", "max_gc"}:
        raise ValueError("mode must be 'min_gc' or 'max_gc'.")
    seq = normalize_rna(sequence if isinstance(sequence, str) else "")
    output = []
    reverse = mode == "max_gc"
    for index in range(0, len(seq) - 2, 3):
        codon = seq[index : index + 3]
        aa = RNA_GENETIC_CODE.get(codon)
        family = SYNONYMOUS_CODONS.get(aa or "", [codon])
        if len(family) <= 1:
            output.append(codon)
        else:
            output.append(sorted(family, key=lambda item: (gc_count(item), item), reverse=reverse)[0])
    output.append(seq[len(output) * 3 :])
    return "".join(output)


def gc_count(sequence: str) -> int:
    return sum(base in {"G", "C"} for base in sequence)


def _log2(value: float) -> float:
    import math

    return math.log(value, 2)


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
