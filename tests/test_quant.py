from __future__ import annotations

import pandas as pd

from rna_stability_elements.quant import (
    aggregate_gene_signal,
    build_stability_targets,
    read_genic_quantification,
)


def test_read_genic_quantification_headerless(tmp_path):
    path = tmp_path / "sample.tsv"
    path.write_text(
        "\n".join(
            [
                "chr1\t0\t100\tENSG000001.1/GENE1\t0\t+\t100\texon_sense\t0\t100\t10\t1\t2",
                "chr1\t0\t100\tENSG000001.1/GENE1\t0\t-\t100\texon_antisense\t0\t100\t5\t0.5\t1",
            ]
        ),
        encoding="utf-8",
    )
    frame = read_genic_quantification(path, feature_type="exon_sense")
    assert len(frame) == 1
    assert frame.iloc[0]["gene_id"] == "ENSG000001"
    assert frame.iloc[0]["gene_symbol"] == "GENE1"


def test_build_stability_targets():
    signal = pd.DataFrame(
        {
            "gene_id": ["ENSG1", "ENSG1", "ENSG1"],
            "gene_symbol": ["G1", "G1", "G1"],
            "cell_line": ["A", "A", "A"],
            "time_h": [0.0, 2.0, 6.0],
            "rpkm": [8.0, 4.0, 2.0],
            "raw_count": [80, 40, 20],
        }
    )
    targets = build_stability_targets(signal, value_column="rpkm", pseudocount=0.0)
    row = targets.iloc[0]
    assert row["log2_stability_2h_0h"] == -1.0
    assert row["log2_stability_6h_2h"] == -1.0
    assert row["half_life_0_2h"] == 2.0


def test_aggregate_gene_signal():
    frame = pd.DataFrame(
        {
            "gene_id": ["ENSG1", "ENSG1"],
            "gene_symbol": ["G1", "G1"],
            "raw_count": [3.0, 4.0],
            "feature_length": [50.0, 50.0],
            "rpkm": [1.0, 2.0],
        }
    )
    grouped = aggregate_gene_signal(frame)
    assert grouped.iloc[0]["raw_count"] == 7.0
    assert grouped.iloc[0]["rpkm"] == 3.0
