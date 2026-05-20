from __future__ import annotations

import gzip
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, TextIO

import pandas as pd

from rna_stability_elements.features import normalize_rna, read_fasta


@dataclass
class TranscriptModel:
    gene_id: str
    gene_symbol: str
    transcript_id: str
    transcript_name: str = ""
    chrom: str = ""
    strand: str = "+"
    gene_type: str = ""
    transcript_type: str = ""
    tags: set[str] = field(default_factory=set)
    transcript_support_level: str = ""
    exons: list[tuple[int, int]] = field(default_factory=list)
    cds_segments: list[tuple[int, int]] = field(default_factory=list)


def parse_gencode_gtf(path: str | Path, *, gene_ids: set[str] | None = None) -> dict[str, TranscriptModel]:
    """Parse the GENCODE GTF features needed for canonical transcript sequence extraction."""
    transcripts: dict[str, TranscriptModel] = {}
    gene_ids_versionless = {strip_version(gene_id) for gene_id in gene_ids} if gene_ids else None

    with _open_text(path) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 9:
                continue
            chrom, _source, feature, start, end, _score, strand, _frame, attrs_text = parts
            if feature not in {"transcript", "exon", "CDS"}:
                continue
            attrs = parse_gtf_attributes(attrs_text)
            transcript_id = strip_version(attrs.get("transcript_id", ""))
            gene_id = strip_version(attrs.get("gene_id", ""))
            if not transcript_id or not gene_id:
                continue
            if gene_ids_versionless is not None and gene_id not in gene_ids_versionless:
                continue

            model = transcripts.get(transcript_id)
            if model is None:
                model = TranscriptModel(
                    gene_id=gene_id,
                    gene_symbol=attrs.get("gene_name", gene_id),
                    transcript_id=transcript_id,
                    transcript_name=attrs.get("transcript_name", transcript_id),
                    chrom=chrom,
                    strand=strand,
                    gene_type=attrs.get("gene_type", attrs.get("gene_biotype", "")),
                    transcript_type=attrs.get("transcript_type", attrs.get("transcript_biotype", "")),
                    tags=set(_as_list(attrs.get("tag", ""))),
                    transcript_support_level=attrs.get("transcript_support_level", ""),
                )
                transcripts[transcript_id] = model
            else:
                model.tags.update(_as_list(attrs.get("tag", "")))
                if not model.transcript_support_level:
                    model.transcript_support_level = attrs.get("transcript_support_level", "")

            interval = (int(start), int(end))
            if feature == "exon":
                model.exons.append(interval)
            elif feature == "CDS":
                model.cds_segments.append(interval)

    return transcripts


def build_transcript_sequence_table(
    master_table: pd.DataFrame,
    *,
    gtf_path: str | Path,
    transcript_fasta_path: str | Path,
) -> pd.DataFrame:
    """Select one representative transcript per gene and attach transcript-region sequences."""
    if "gene_id" not in master_table:
        raise ValueError("Master table must contain a gene_id column.")

    gene_ids = {strip_version(gene_id) for gene_id in master_table["gene_id"].astype(str)}
    transcripts = parse_gencode_gtf(gtf_path, gene_ids=gene_ids)
    fasta = read_fasta(transcript_fasta_path)
    transcript_sequences = {strip_version(name.split("|")[0]): seq for name, seq in fasta.items()}

    by_gene: dict[str, list[TranscriptModel]] = defaultdict(list)
    for transcript in transcripts.values():
        if transcript.transcript_id in transcript_sequences:
            by_gene[transcript.gene_id].append(transcript)

    rows = []
    for gene_id in sorted(gene_ids):
        candidates = by_gene.get(gene_id, [])
        if not candidates:
            rows.append(_missing_sequence_row(gene_id))
            continue
        transcript = choose_representative_transcript(candidates, transcript_sequences)
        full_sequence = normalize_rna(transcript_sequences[transcript.transcript_id])
        cds_start, cds_end = transcript_cds_bounds(transcript)
        cds_sequence = full_sequence[cds_start:cds_end] if cds_start is not None and cds_end is not None else ""
        utr5_sequence = full_sequence[:cds_start] if cds_start is not None else ""
        utr3_sequence = full_sequence[cds_end:] if cds_end is not None else ""
        rows.append(
            {
                "gene_id": gene_id,
                "annotation_gene_symbol": transcript.gene_symbol,
                "chromosome": transcript.chrom,
                "strand": transcript.strand,
                "gene_biotype": transcript.gene_type,
                "canonical_transcript_id": transcript.transcript_id,
                "canonical_transcript_name": transcript.transcript_name,
                "transcript_biotype": transcript.transcript_type,
                "transcript_support_level": transcript.transcript_support_level,
                "transcript_tags": ",".join(sorted(transcript.tags)),
                "sequence_full": full_sequence,
                "sequence_5utr": utr5_sequence,
                "sequence_cds": cds_sequence,
                "sequence_3utr": utr3_sequence,
                "length_full": len(full_sequence),
                "length_5utr": len(utr5_sequence),
                "length_cds": len(cds_sequence),
                "length_3utr": len(utr3_sequence),
                "gc_full": gc_fraction(full_sequence),
                "gc_5utr": gc_fraction(utr5_sequence),
                "gc_cds": gc_fraction(cds_sequence),
                "gc_3utr": gc_fraction(utr3_sequence),
                "au_full": au_fraction(full_sequence),
                "au_3utr": au_fraction(utr3_sequence),
                "u_3utr": u_fraction(utr3_sequence),
                "sequence_status": "mapped",
            }
        )
    return pd.DataFrame(rows)


def write_transcript_sequence_table(
    master_path: str | Path,
    *,
    gtf_path: str | Path,
    transcript_fasta_path: str | Path,
    out: str | Path,
) -> pd.DataFrame:
    master = pd.read_csv(master_path, sep="\t")
    table = build_transcript_sequence_table(
        master,
        gtf_path=gtf_path,
        transcript_fasta_path=transcript_fasta_path,
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, sep="\t", index=False)
    return table


def merge_master_with_sequences(
    master: pd.DataFrame,
    sequence_table: pd.DataFrame,
) -> pd.DataFrame:
    if "gene_id" not in master or "gene_id" not in sequence_table:
        raise ValueError("Both inputs must contain gene_id.")
    sequence_columns = [column for column in sequence_table.columns if column != "gene_id"]
    merged = master.drop(columns=[column for column in sequence_columns if column in master], errors="ignore").merge(
        sequence_table,
        on="gene_id",
        how="left",
    )
    if "sequence_status" in merged:
        merged["sequence_status"] = merged["sequence_status"].fillna("missing_gencode_mapping")
    return merged


def write_modeling_table_with_sequences(
    master_path: str | Path,
    sequence_table_path: str | Path,
    *,
    out: str | Path,
) -> pd.DataFrame:
    master = pd.read_csv(master_path, sep="\t")
    sequences = pd.read_csv(sequence_table_path, sep="\t")
    merged = merge_master_with_sequences(master, sequences)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, sep="\t", index=False)
    return merged


def choose_representative_transcript(
    candidates: Iterable[TranscriptModel],
    transcript_sequences: dict[str, str],
) -> TranscriptModel:
    return sorted(
        candidates,
        key=lambda transcript: transcript_rank(transcript, len(transcript_sequences.get(transcript.transcript_id, ""))),
    )[0]


def transcript_rank(transcript: TranscriptModel, transcript_length: int) -> tuple[int, int, int, int, int, int, str]:
    tags = transcript.tags
    appris_score = min(
        [_appris_rank(tag) for tag in tags if tag.startswith("appris_principal") or tag.startswith("appris_candidate")]
        or [9]
    )
    tsl = _support_level_rank(transcript.transcript_support_level)
    return (
        0 if "MANE_Select" in tags else 1,
        0 if transcript.transcript_type == "protein_coding" else 1,
        appris_score,
        0 if "basic" in tags else 1,
        tsl,
        -int(transcript_length),
        transcript.transcript_id,
    )


def transcript_cds_bounds(transcript: TranscriptModel) -> tuple[int | None, int | None]:
    if not transcript.cds_segments or not transcript.exons:
        return None, None
    offsets: list[int] = []
    for cds_start, cds_end in transcript.cds_segments:
        start_offset = genomic_position_to_transcript_offset(transcript.exons, transcript.strand, cds_start)
        end_offset = genomic_position_to_transcript_offset(transcript.exons, transcript.strand, cds_end) + 1
        offsets.extend([start_offset, end_offset])
    return min(offsets), max(offsets)


def genomic_position_to_transcript_offset(exons: list[tuple[int, int]], strand: str, position: int) -> int:
    ordered = sorted(exons, key=lambda item: item[0], reverse=(strand == "-"))
    offset = 0
    for start, end in ordered:
        if start <= position <= end:
            return offset + (position - start if strand == "+" else end - position)
        offset += end - start + 1
    raise ValueError(f"Position {position} is outside transcript exons.")


def parse_gtf_attributes(text: str) -> dict[str, str | list[str]]:
    attrs: dict[str, str | list[str]] = {}
    for chunk in text.strip().strip(";").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if " " not in chunk:
            continue
        key, value = chunk.split(" ", 1)
        value = value.strip().strip('"')
        if key == "tag":
            attrs.setdefault("tag", [])
            assert isinstance(attrs["tag"], list)
            attrs["tag"].append(value)
        else:
            attrs[key] = value
    return attrs


def strip_version(identifier: str) -> str:
    return str(identifier).split(".", 1)[0]


def gc_fraction(sequence: str) -> float:
    seq = normalize_rna(sequence)
    return (seq.count("G") + seq.count("C")) / max(len(seq), 1)


def au_fraction(sequence: str) -> float:
    seq = normalize_rna(sequence)
    return (seq.count("A") + seq.count("U")) / max(len(seq), 1)


def u_fraction(sequence: str) -> float:
    seq = normalize_rna(sequence)
    return seq.count("U") / max(len(seq), 1)


def _missing_sequence_row(gene_id: str) -> dict[str, str]:
    return {
        "gene_id": gene_id,
        "sequence_status": "missing_gencode_mapping",
    }


def _as_list(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        return value
    if not value:
        return []
    return [value]


def _open_text(path: str | Path) -> TextIO:
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _appris_rank(tag: str) -> int:
    if tag.startswith("appris_principal_"):
        suffix = tag.rsplit("_", 1)[-1]
        return int(suffix) if suffix.isdigit() else 1
    if tag == "appris_principal":
        return 1
    if tag.startswith("appris_candidate"):
        return 7
    return 9


def _support_level_rank(value: str) -> int:
    parts = str(value).split()
    if not parts:
        return 9
    first = parts[0]
    return int(first) if first.isdigit() else 9
