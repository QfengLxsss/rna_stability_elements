import pandas as pd

from rna_stability_elements.target_quality import (
    aggregate_replicate_paired_targets,
    build_replicate_paired_targets,
    build_strict_consensus,
)


def test_replicate_paired_targets_filter_low_2h_denominator():
    signal = pd.DataFrame(
        {
            "gene_id": ["ENSG1"] * 6,
            "gene_symbol": ["GENE1"] * 6,
            "cell_line": ["A"] * 6,
            "biological_replicates": ["1", "1", "1", "2", "2", "2"],
            "time_h": [0.0, 2.0, 6.0, 0.0, 2.0, 6.0],
            "rpkm": [1.0, 0.1, 0.2, 1.0, 1.0, 0.5],
            "raw_count": [10, 1, 2, 10, 10, 5],
        }
    )
    replicate_targets = build_replicate_paired_targets(signal, min_signal_0h=0.5, min_signal_2h=0.5)
    assert replicate_targets["log2_stability_6h_2h_replicate_input_pass"].tolist() == [False, True]
    assert replicate_targets["log2_stability_6h_0h_replicate_input_pass"].tolist() == [True, True]

    targets = aggregate_replicate_paired_targets(replicate_targets)
    row = targets.iloc[0]
    assert row["log2_stability_6h_2h_n_replicates"] == 1
    assert row["log2_stability_6h_2h_quality_flag"] == "single_replicate_after_signal_filter"
    assert row["log2_stability_6h_0h_n_replicates"] == 2


def test_strict_consensus_uses_pass_only_quality_flags():
    targets = pd.DataFrame(
        {
            "gene_id": ["ENSG1", "ENSG1", "ENSG2"],
            "gene_symbol": ["GENE1", "GENE1", "GENE2"],
            "cell_line": ["A", "B", "A"],
            "signal_0h_median": [1.0, 1.0, 1.0],
            "signal_2h_median": [1.0, 1.0, 1.0],
            "signal_6h_median": [1.0, 1.0, 1.0],
            "log2_stability_6h_0h": [0.1, 0.3, 2.0],
            "log2_stability_6h_0h_quality_flag": ["pass", "high_replicate_target_span", "pass"],
        }
    )
    consensus = build_strict_consensus(
        targets,
        target_column="log2_stability_6h_0h",
        min_cell_lines=1,
        pass_only=True,
    )
    gene1 = consensus[consensus["gene_id"] == "ENSG1"].iloc[0]
    assert gene1["n_cell_lines"] == 1
    assert gene1["stability_consensus_median"] == 0.1
