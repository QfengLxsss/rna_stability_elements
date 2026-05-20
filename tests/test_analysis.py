from __future__ import annotations

import pandas as pd
import pytest

from rna_stability_elements.analysis import (
    build_consensus_targets,
    build_modeling_master_table,
    build_replicate_qc,
    compare_target_sets,
)


def test_build_consensus_targets_filters_and_summarizes():
    targets = pd.DataFrame(
        {
            "gene_id": ["ENSG1", "ENSG1", "ENSG2"],
            "gene_symbol": ["G1", "G1", "G2"],
            "cell_line": ["A", "B", "A"],
            "signal_0h": [2.0, 4.0, 6.0],
            "signal_2h": [1.0, 2.0, 3.0],
            "signal_6h": [0.5, 1.0, 1.5],
            "log2_stability_6h_2h": [-1.0, -3.0, 2.0],
        }
    )

    consensus = build_consensus_targets(targets, min_cell_lines=2)

    assert list(consensus["gene_id"]) == ["ENSG1"]
    row = consensus.iloc[0]
    assert row["n_cell_lines"] == 2
    assert row["stability_consensus_median"] == -2.0
    assert row["stability_consensus_iqr"] == 1.0
    assert row["signal_0h_median"] == 3.0
    assert row["label_quality_flag"] == "pass"


def test_compare_target_sets_shared_rows():
    left = pd.DataFrame(
        {
            "gene_id": ["ENSG1", "ENSG2", "ENSG3"],
            "gene_symbol": ["G1", "G2", "G3"],
            "cell_line": ["A", "A", "B"],
            "log2_stability_6h_2h": [1.0, 2.0, 3.0],
        }
    )
    right = pd.DataFrame(
        {
            "gene_id": ["ENSG1", "ENSG2", "ENSG4"],
            "gene_symbol": ["G1", "G2", "G4"],
            "cell_line": ["A", "A", "B"],
            "log2_stability_6h_2h": [2.0, 4.0, 8.0],
        }
    )

    comparison = compare_target_sets(left, right, left_name="left", right_name="right")
    all_row = comparison[comparison["cell_line"] == "ALL"].iloc[0]

    assert all_row["n_shared_rows"] == 2
    assert all_row["pearson"] == pytest.approx(1.0)
    assert all_row["spearman"] == pytest.approx(1.0)
    assert all_row["delta_median"] == 1.5


def test_build_replicate_qc_flags_discordant_signal():
    signal = pd.DataFrame(
        {
            "gene_id": ["ENSG1", "ENSG1", "ENSG2", "ENSG2"],
            "gene_symbol": ["G1", "G1", "G2", "G2"],
            "cell_line": ["A", "A", "A", "A"],
            "time_h": [0.0, 0.0, 0.0, 0.0],
            "file_accession": ["F1", "F2", "F1", "F2"],
            "rpkm": [8.0, 8.5, 1.0, 8.0],
            "raw_count": [80, 85, 10, 80],
        }
    )

    gene_qc, experiment_qc = build_replicate_qc(signal, max_log2_range=1.0)
    flags = dict(zip(gene_qc["gene_id"], gene_qc["replicate_quality_flag"]))

    assert flags["ENSG1"] == "pass"
    assert flags["ENSG2"] == "high_replicate_discordance"
    assert experiment_qc.iloc[0]["n_replicates"] == 2
    assert experiment_qc.iloc[0]["n_replicate_pairs"] == 1


def test_build_modeling_master_table_merges_replicate_summary():
    consensus = pd.DataFrame(
        {
            "gene_id": ["ENSG1"],
            "gene_symbol": ["G1"],
            "n_cell_lines": [2],
            "stability_consensus_median": [-1.0],
            "stability_consensus_mean": [-1.1],
            "stability_consensus_std": [0.2],
            "stability_consensus_iqr": [0.3],
            "stability_consensus_min": [-1.3],
            "stability_consensus_max": [-0.9],
            "signal_0h_median": [2.0],
            "signal_2h_median": [1.0],
            "signal_6h_median": [0.5],
            "target_source_column": ["log2_stability_6h_2h"],
            "label_quality_flag": ["pass"],
        }
    )
    replicate_qc = pd.DataFrame(
        {
            "gene_id": ["ENSG1", "ENSG1"],
            "replicate_quality_flag": ["pass", "high_replicate_discordance"],
            "replicate_log2_range": [0.2, 1.5],
            "replicate_cv": [0.1, 0.8],
            "mean_signal": [5.0, 2.0],
        }
    )

    table = build_modeling_master_table(consensus, replicate_qc=replicate_qc)

    assert table.iloc[0]["sample_id"] == "ENSG1"
    assert table.iloc[0]["target_label"] == -1.0
    assert table.iloc[0]["sequence_status"] == "pending_gencode_mapping"
    assert table.iloc[0]["replicate_qc_observations"] == 2
