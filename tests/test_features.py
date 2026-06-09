from __future__ import annotations

import pandas as pd

from rna_stability_elements.features import (
    codon_feature_table,
    codon_features,
    synonymous_recoded_sequence,
)


def test_codon_features_count_frame_and_frequencies():
    features = codon_features("AUGGCCUAA")

    assert features["cds_codon_count"] == 3
    assert features["cds_frame_remainder"] == 0
    assert features["cds_has_start_aug"] == 1
    assert features["cds_terminal_stop"] == 1
    assert features["cds_codon_AUG"] == 1 / 3
    assert features["cds_codon_GCC"] == 1 / 3
    assert features["cds_aa_M"] == 1 / 3
    assert features["cds_aa_A"] == 1 / 3


def test_synonymous_recoding_preserves_singleton_codons_and_changes_gc_choice():
    assert synonymous_recoded_sequence("GCCGCUUGG", mode="min_gc") == "GCAGCAUGG"
    assert synonymous_recoded_sequence("GCCGCUUGG", mode="max_gc") == "GCGGCGUGG"


def test_codon_feature_table_keeps_metadata():
    table = pd.DataFrame(
        {
            "gene_id": ["ENSG1"],
            "gene_symbol": ["GENE1"],
            "target_label": [0.5],
            "sequence_cds": ["AUGGCCUAA"],
        }
    )

    features = codon_feature_table(table)

    assert features.loc[0, "gene_id"] == "ENSG1"
    assert features.loc[0, "target_label"] == 0.5
    assert "cds_codon_GCC" in features.columns
