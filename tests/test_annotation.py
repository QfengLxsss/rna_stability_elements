from __future__ import annotations

import pandas as pd

from rna_stability_elements.annotation import (
    build_transcript_sequence_table,
    genomic_position_to_transcript_offset,
    merge_master_with_sequences,
)
from rna_stability_elements.features import compact_sequence_model_features
from rna_stability_elements.features import merge_feature_tables


def test_genomic_position_to_transcript_offset_positive_and_negative():
    exons = [(100, 104), (200, 204)]

    assert genomic_position_to_transcript_offset(exons, "+", 100) == 0
    assert genomic_position_to_transcript_offset(exons, "+", 203) == 8
    assert genomic_position_to_transcript_offset(exons, "-", 204) == 0
    assert genomic_position_to_transcript_offset(exons, "-", 101) == 8


def test_build_transcript_sequence_table_slices_regions(tmp_path):
    gtf = tmp_path / "mini.gtf"
    fasta = tmp_path / "mini.fa"
    gtf.write_text(
        "\n".join(
            [
                'chr1\ttest\ttranscript\t100\t204\t.\t+\t.\tgene_id "ENSG1.1"; transcript_id "ENST1.1"; gene_name "G1"; gene_type "protein_coding"; transcript_type "protein_coding"; tag "basic"; transcript_support_level "1";',
                'chr1\ttest\texon\t100\t104\t.\t+\t.\tgene_id "ENSG1.1"; transcript_id "ENST1.1"; gene_name "G1";',
                'chr1\ttest\texon\t200\t204\t.\t+\t.\tgene_id "ENSG1.1"; transcript_id "ENST1.1"; gene_name "G1";',
                'chr1\ttest\tCDS\t102\t104\t.\t+\t0\tgene_id "ENSG1.1"; transcript_id "ENST1.1"; gene_name "G1";',
                'chr1\ttest\tCDS\t200\t202\t.\t+\t0\tgene_id "ENSG1.1"; transcript_id "ENST1.1"; gene_name "G1";',
            ]
        ),
        encoding="utf-8",
    )
    fasta.write_text(">ENST1.1|ENSG1.1|G1\nAACCCGGGUU\n", encoding="utf-8")
    master = pd.DataFrame({"gene_id": ["ENSG1"]})

    table = build_transcript_sequence_table(master, gtf_path=gtf, transcript_fasta_path=fasta)
    row = table.iloc[0]

    assert row["canonical_transcript_id"] == "ENST1"
    assert row["sequence_5utr"] == "AA"
    assert row["sequence_cds"] == "CCCGGG"
    assert row["sequence_3utr"] == "UU"
    assert row["length_cds"] == 6
    assert row["sequence_status"] == "mapped"


def test_merge_master_with_sequences_updates_sequence_status():
    master = pd.DataFrame(
        {
            "gene_id": ["ENSG1", "ENSG2"],
            "sequence_status": ["pending_gencode_mapping", "pending_gencode_mapping"],
            "target_label": [1.0, 2.0],
        }
    )
    sequences = pd.DataFrame(
        {
            "gene_id": ["ENSG1"],
            "sequence_status": ["mapped"],
            "canonical_transcript_id": ["ENST1"],
        }
    )

    merged = merge_master_with_sequences(master, sequences)

    assert merged.loc[merged["gene_id"] == "ENSG1", "sequence_status"].iloc[0] == "mapped"
    assert merged.loc[merged["gene_id"] == "ENSG2", "sequence_status"].iloc[0] == "missing_gencode_mapping"


def test_compact_sequence_model_features_region_prefixes():
    table = pd.DataFrame(
        {
            "gene_id": ["ENSG1"],
            "gene_symbol": ["G1"],
            "target_label": [0.5],
            "sequence_full": ["AACCGGUU"],
            "sequence_3utr": ["AUUUA"],
            "sequence_status": ["mapped"],
        }
    )

    features = compact_sequence_model_features(
        table,
        regions=["full", "3utr"],
        ks=[3],
        motifs={"AU-rich": "AUUUA"},
    )

    assert "sequence_full" not in features.columns
    assert features.iloc[0]["full_length"] == 8
    assert features.iloc[0]["3utr_motif_AU-rich_count"] == 1
    assert "full_kmer_AAC" in features.columns


def test_merge_feature_tables_keeps_one_metadata_copy():
    left = pd.DataFrame(
        {
            "gene_id": ["ENSG1", "ENSG2"],
            "gene_symbol": ["A", "B"],
            "target_label": [0.1, -0.2],
            "lm_emb_0000": [1.0, 2.0],
        }
    )
    right = pd.DataFrame(
        {
            "gene_id": ["ENSG1", "ENSG2"],
            "gene_symbol": ["A2", "B2"],
            "target_label": [0.1, -0.2],
            "kmer_AAA": [3.0, 4.0],
        }
    )

    merged = merge_feature_tables(left, right)

    assert list(merged.columns) == ["gene_id", "gene_symbol", "target_label", "lm_emb_0000", "kmer_AAA"]
    assert merged.loc[0, "gene_symbol"] == "A"
