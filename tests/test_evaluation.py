from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rna_stability_elements.models.evaluation import (
    chromosome_holdout_splits,
    evaluate_sequence_models,
    feature_groups,
    input_ablation_feature_sets,
    numeric_feature_columns,
    repeated_random_splits,
    resolve_feature_set,
)


def test_input_ablation_feature_sets_separate_full_and_structured_regions():
    columns = [
        "full_length",
        "full_kmer_AAA",
        "5utr_length",
        "5utr_motif_CPE_count",
        "cds_gc_fraction",
        "3utr_kmer_AAAA",
    ]
    sets = input_ablation_feature_sets(columns)

    assert sets["full_only"] == ["full_length", "full_kmer_AAA"]
    assert sets["structured_regions"] == columns[2:]
    assert sets["utr_only"] == ["5utr_length", "5utr_motif_CPE_count", "3utr_kmer_AAAA"]
    assert sets["structured_no_3utr"] == ["5utr_length", "5utr_motif_CPE_count", "cds_gc_fraction"]
from rna_stability_elements.models.rna_bert import encode_kmer_block
from rna_stability_elements.models.rna_lm_embeddings import (
    format_sequence_for_lm,
    normalize_sequence,
    sequence_chunks,
    write_multi_region_rna_lm_embeddings,
)
from rna_stability_elements.models.ramht import RamhtConfig, build_ramht_model, encode_codons
from rna_stability_elements.models.saluki_like import make_region_ids
from rna_stability_elements.models.sequence_cnn import encode_sequence
from rna_stability_elements.models.sequence_cnn import RegionLengths
from rna_stability_elements.models.sequence_transformer import total_length


def _feature_frame(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    length = rng.normal(size=n)
    gc = rng.normal(size=n)
    kmer = rng.normal(size=n)
    motif = rng.normal(size=n)
    target = 0.5 * length - 0.2 * gc + 0.1 * kmer + rng.normal(scale=0.05, size=n)
    return pd.DataFrame(
        {
            "gene_id": [f"ENSG{i:05d}" for i in range(n)],
            "gene_symbol": [f"G{i}" for i in range(n)],
            "chromosome": ["chr1"] * 10 + ["chr2"] * 10 + ["chr3"] * (n - 20),
            "target_label": target,
            "full_length": length,
            "full_gc_fraction": gc,
            "full_kmer_AAA": kmer,
            "3utr_motif_AU-rich_count": motif,
            "lm_5utr_emb_0000": rng.normal(size=n),
            "lm_cds_emb_0000": rng.normal(size=n),
            "lm_3utr_emb_0000": rng.normal(size=n),
        }
    )


def test_feature_set_resolution():
    frame = _feature_frame()
    numeric = numeric_feature_columns(frame, target_column="target_label")
    groups = feature_groups(numeric)

    assert resolve_feature_set("length_only", numeric, groups) == ["full_length"]
    assert resolve_feature_set("composition_only", numeric, groups) == ["full_gc_fraction"]
    assert resolve_feature_set("motif_only", numeric, groups) == ["3utr_motif_AU-rich_count"]
    assert resolve_feature_set("no_kmer", numeric, groups) == [
        "full_length",
        "full_gc_fraction",
        "3utr_motif_AU-rich_count",
        "lm_5utr_emb_0000",
        "lm_cds_emb_0000",
        "lm_3utr_emb_0000",
    ]
    assert resolve_feature_set("lm_5utr_only", numeric, groups) == ["lm_5utr_emb_0000"]
    assert resolve_feature_set("no_lm_cds", numeric, groups) == [
        "full_length",
        "full_gc_fraction",
        "full_kmer_AAA",
        "3utr_motif_AU-rich_count",
        "lm_5utr_emb_0000",
        "lm_3utr_emb_0000",
    ]


def test_splits_do_not_overlap():
    frame = _feature_frame()
    random_split = repeated_random_splits(frame, n_repeats=1, test_size=0.2, random_state=1)[0]
    chromosome_split = chromosome_holdout_splits(frame, min_test_samples=5)[0]

    assert set(random_split.train_index).isdisjoint(set(random_split.test_index))
    assert set(chromosome_split.train_index).isdisjoint(set(chromosome_split.test_index))
    assert chromosome_split.holdout_group == "chr1"


def test_evaluate_sequence_models_outputs_tables():
    frame = _feature_frame()

    metrics, predictions, summary, importances = evaluate_sequence_models(
        frame,
        target_column="target_label",
        models=["ridge"],
        feature_sets=["all", "length_only"],
        evaluations=["repeated_random", "chromosome_holdout"],
        n_repeats=2,
        test_size=0.2,
        min_test_samples=5,
    )

    assert set(metrics["evaluation"]) == {"repeated_random", "chromosome_holdout"}
    assert set(metrics["feature_set"]) == {"all", "length_only"}
    assert {"y_true", "y_pred", "residual"}.issubset(predictions.columns)
    assert {"pearson_mean", "spearman_mean", "r2_mean", "n_splits"}.issubset(summary.columns)
    assert {"feature", "importance", "feature_group", "model"}.issubset(importances.columns)


def test_evaluate_sequence_models_with_region_pca():
    frame = _feature_frame(n=45)

    metrics, predictions, summary, _importances = evaluate_sequence_models(
        frame,
        target_column="target_label",
        models=["ridge"],
        feature_sets=["lm_only"],
        evaluations=["repeated_random"],
        n_repeats=1,
        test_size=0.2,
        preprocessing="region_pca",
        pca_components=1,
    )

    assert metrics.iloc[0]["preprocessing"] == "region_pca"
    assert predictions.iloc[0]["preprocessing"] == "region_pca"
    assert summary.iloc[0]["n_splits"] == 1


def test_encode_sequence_balanced_truncation():
    encoded = encode_sequence("AACCGGUU", max_length=6)

    assert encoded.tolist() == [1, 1, 2, 3, 4, 4]


def test_encode_sequence_end_crop():
    encoded = encode_sequence("AACCGGUU", max_length=4, crop_strategy="end")

    assert encoded.tolist() == [3, 3, 4, 4]


def test_saluki_like_region_ids_follow_concatenated_order():
    region_ids = make_region_ids(RegionLengths(utr5=2, cds=3, utr3=1))

    assert region_ids.tolist() == [1, 1, 2, 2, 2, 3]


def test_transformer_total_length_sums_regions():
    assert total_length(RegionLengths(utr5=2, cds=3, utr3=4)) == 9


def test_rna_bert_kmer_encoding_uses_base4_vocab():
    sequences = np.array([[1, 2, 3, 4, 0]], dtype=np.int64)
    encoded = encode_kmer_block(sequences, kmer_size=2, kmer_stride=1)

    assert encoded.tolist() == [[3, 8, 13, 0]]


def test_rna_lm_sequence_formatting_modes():
    assert normalize_sequence("ACTU", alphabet="rna") == "ACUU"
    assert normalize_sequence("ACUU", alphabet="dna") == "ACTT"
    assert sequence_chunks("AACCGG", chunk_size=4, chunk_stride=2) == ["AACC", "CCGG"]
    assert format_sequence_for_lm("ACGU", sequence_format="spaced_chars") == "A C G U"
    assert format_sequence_for_lm("ACGU", sequence_format="kmer", kmer_size=3) == "ACG CGU"


def test_multi_region_rna_lm_embedding_merge(tmp_path):
    def write_region(path, values):
        pd.DataFrame(
            {
                "gene_id": ["ENSG1", "ENSG2"],
                "gene_symbol": ["A", "B"],
                "chromosome": ["chr1", "chr2"],
                "target_label": [0.1, -0.2],
                "lm_emb_0000": values,
                "lm_emb_0001": [value + 1 for value in values],
                "lm_sequence_column": ["x", "x"],
            }
        ).to_csv(path, sep="\t", index=False)

    utr5 = tmp_path / "utr5.tsv"
    cds = tmp_path / "cds.tsv"
    utr3 = tmp_path / "utr3.tsv"
    out = tmp_path / "merged.tsv"
    write_region(utr5, [1.0, 2.0])
    write_region(cds, [3.0, 4.0])
    write_region(utr3, [5.0, 6.0])

    merged = write_multi_region_rna_lm_embeddings(utr5_path=utr5, cds_path=cds, utr3_path=utr3, out=out)

    assert list(merged["gene_id"]) == ["ENSG1", "ENSG2"]
    assert "lm_5utr_emb_0000" in merged.columns
    assert "lm_cds_emb_0000" in merged.columns
    assert "lm_3utr_emb_0000" in merged.columns
    assert merged.loc[0, "lm_cds_emb_0001"] == 4.0


def test_ramht_codon_encoding():
    frame = pd.DataFrame({"sequence_cds": ["AUGGCUUAA", "NNNAUG"]})

    encoded = encode_codons(frame, codon_length=3, crop_strategy="balanced")

    assert encoded.shape == (2, 3)
    assert encoded[0].tolist() == [15, 40, 49]
    assert encoded[1].tolist() == [0, 15, 0]


def test_ramht_forward_small_model():
    torch = pytest.importorskip("torch")
    config = RamhtConfig(
        region_lengths=RegionLengths(utr5=4, cds=6, utr3=4),
        codon_length=2,
        embedding_dim=4,
        model_dim=16,
        codon_dim=16,
        transformer_layers=1,
        codon_layers=1,
        attention_heads=4,
        feedforward_dim=32,
        feature_hidden_dim=16,
        head_hidden_dim=16,
        dropout=0.0,
        token_dropout=0.0,
    )
    model = build_ramht_model(config, feature_dim=5, n_tasks=4)
    prediction, gate = model(
        {
            "5utr": torch.tensor([[1, 2, 0, 0], [1, 1, 2, 2]]),
            "cds": torch.tensor([[1, 2, 3, 4, 0, 0], [1, 2, 3, 4, 1, 2]]),
            "3utr": torch.tensor([[4, 3, 0, 0], [4, 4, 3, 3]]),
            "codon": torch.tensor([[15, 0], [15, 28]]),
            "_numeric": torch.zeros(2, 5),
        }
    )

    assert prediction.shape == (2, 4)
    assert gate.shape == (2, 3)
    assert torch.allclose(gate.sum(dim=1), torch.ones(2))


def test_ramht_v2_task_specific_gates_and_empty_regions_are_finite():
    torch = pytest.importorskip("torch")
    config = RamhtConfig(
        region_lengths=RegionLengths(utr5=4, cds=6, utr3=4),
        codon_length=2,
        embedding_dim=4,
        model_dim=16,
        codon_dim=16,
        transformer_layers=1,
        codon_layers=1,
        attention_heads=4,
        feedforward_dim=32,
        feature_hidden_dim=16,
        head_hidden_dim=16,
        dropout=0.0,
        token_dropout=0.0,
        fusion_mode="gated_residual",
        separate_codon_stream=True,
        task_specific_gates=True,
        mask_padding_attention=True,
        engineered_output_skip=True,
    )
    model = build_ramht_model(config, feature_dim=5, n_tasks=4)
    prediction, gate = model(
        {
            "5utr": torch.zeros(2, 4, dtype=torch.long),
            "cds": torch.tensor([[1, 2, 3, 0, 0, 0], [1, 2, 3, 4, 1, 2]]),
            "3utr": torch.zeros(2, 4, dtype=torch.long),
            "codon": torch.tensor([[15, 0], [15, 28]]),
            "_numeric": torch.zeros(2, 5),
        }
    )

    assert prediction.shape == (2, 4)
    assert gate.shape == (2, 4, 3)
    assert torch.isfinite(prediction).all()
    assert torch.allclose(gate.sum(dim=2), torch.ones(2, 4))
