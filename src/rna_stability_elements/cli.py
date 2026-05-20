from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rna_stability_elements.annotation import (
    write_modeling_table_with_sequences,
    write_transcript_sequence_table,
)
from rna_stability_elements.config import cell_line_aliases, expected_encode_terms, load_config
from rna_stability_elements.encode import (
    EncodeQuery,
    collect_experiment_files,
    discover_pulse_chase_series,
    download_files,
)
from rna_stability_elements.features import (
    sequence_feature_table,
    write_compact_sequence_model_features,
    write_merged_feature_table,
)
from rna_stability_elements.interpretation import write_leaderboard_and_grammar_report
from rna_stability_elements.models.baselines import save_metrics, train_baseline
from rna_stability_elements.models.evaluation import write_sequence_model_evaluation
from rna_stability_elements.models.rna_lm_embeddings import (
    write_multi_region_rna_lm_embeddings,
    write_rna_lm_embeddings,
)
from rna_stability_elements.models.rna_bert import write_rna_bert_evaluation
from rna_stability_elements.models.saluki_like import write_saluki_like_evaluation
from rna_stability_elements.models.sequence_transformer import write_sequence_transformer_evaluation
from rna_stability_elements.models.sequence_cnn import RegionLengths, write_region_cnn_evaluation
from rna_stability_elements.quant import build_targets_from_manifest
from rna_stability_elements.visualization import make_progress_figures, write_progress_report
from rna_stability_elements.analysis import (
    write_consensus_targets,
    write_modeling_master_table,
    write_replicate_qc,
    write_target_comparison,
    write_target_summaries,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rse")
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser("discover-encode", help="Discover ENCODE series and files.")
    discover.add_argument("--config", default="configs/project.yaml")
    discover.add_argument("--series-out", default="data/interim/encode_series.tsv")
    discover.add_argument("--files-out", default="data/interim/encode_files.tsv")
    discover.add_argument(
        "--file-format",
        action="append",
        help="Optional ENCODE file format filter. Can be passed multiple times.",
    )
    discover.add_argument(
        "--output-type",
        action="append",
        help="Optional ENCODE output type filter. Can be passed multiple times.",
    )

    downloader = subparsers.add_parser("download-files", help="Download files from a manifest.")
    downloader.add_argument("--files", required=True)
    downloader.add_argument("--out-dir", default="data/raw/encode")
    downloader.add_argument("--file-format", default="tsv")
    downloader.add_argument("--output-type", default="genic features quantifications")
    downloader.add_argument("--overwrite", action="store_true")
    downloader.add_argument("--workers", type=int, default=1)

    targets = subparsers.add_parser("build-targets", help="Build stability targets.")
    targets.add_argument("--files", required=True)
    targets.add_argument("--out", default="data/processed/stability_targets.tsv")
    targets.add_argument("--feature-type", default="exon_sense")
    targets.add_argument("--value-column", default="rpkm")
    targets.add_argument("--pseudocount", type=float, default=0.1)
    targets.add_argument("--min-signal-0h", type=float, default=0.5)
    targets.add_argument("--min-cell-lines-per-gene", type=int, default=1)

    seq = subparsers.add_parser("make-sequence-features", help="Build sequence feature table.")
    seq.add_argument("--fasta", required=True)
    seq.add_argument("--config", default="configs/project.yaml")
    seq.add_argument("--out", default="data/processed/sequence_features.tsv")

    train = subparsers.add_parser("train-baseline", help="Train a numeric baseline model.")
    train.add_argument("--features", required=True)
    train.add_argument("--target-column", required=True)
    train.add_argument(
        "--model",
        choices=["ridge", "elasticnet", "random_forest", "xgboost_gpu", "xgboost_cpu"],
        default="ridge",
    )
    train.add_argument("--group-column", default="cell_line")
    train.add_argument("--leave-group")
    train.add_argument("--out", default="data/processed/baseline_metrics.json")

    evaluate = subparsers.add_parser(
        "evaluate-sequence-models",
        help="Run repeated random split, chromosome holdout, and feature ablation evaluation.",
    )
    evaluate.add_argument("--features", default="data/processed/sequence_model_features.tsv")
    evaluate.add_argument("--target-column", default="target_label")
    evaluate.add_argument("--metrics-out", default="data/processed/strict_sequence_metrics.tsv")
    evaluate.add_argument("--predictions-out", default="data/processed/strict_sequence_predictions.tsv")
    evaluate.add_argument("--summary-out", default="data/processed/strict_sequence_summary.tsv")
    evaluate.add_argument("--importance-out")
    evaluate.add_argument(
        "--model",
        action="append",
        choices=["ridge", "elasticnet", "random_forest", "xgboost_gpu", "xgboost_cpu", "mlp"],
        help="Model to evaluate. Can repeat. Default: elasticnet.",
    )
    evaluate.add_argument(
        "--feature-set",
        action="append",
        help=(
            "Feature set to evaluate. Examples: all, length_only, composition_only, motif_only, "
            "kmer_only, kmer3_only, kmer4_only, no_length, no_kmer, 3utr_region."
        ),
    )
    evaluate.add_argument(
        "--evaluation",
        action="append",
        choices=["repeated_random", "chromosome_holdout"],
        help="Evaluation design. Can repeat. Default: both.",
    )
    evaluate.add_argument("--n-repeats", type=int, default=5)
    evaluate.add_argument("--test-size", type=float, default=0.2)
    evaluate.add_argument("--random-state", type=int, default=13)
    evaluate.add_argument("--chromosome-column", default="chromosome")
    evaluate.add_argument("--min-test-samples", type=int, default=50)
    evaluate.add_argument("--preprocessing", choices=["standard", "pca", "region_pca"], default="standard")
    evaluate.add_argument("--pca-components", type=int, default=128)

    region_cnn = subparsers.add_parser(
        "train-region-cnn",
        help="Train and evaluate a region-aware 5'UTR/CDS/3'UTR sequence CNN.",
    )
    region_cnn.add_argument("--table", default="data/processed/modeling_master_with_sequences.tsv")
    region_cnn.add_argument("--feature-table")
    region_cnn.add_argument("--target-column", default="target_label")
    region_cnn.add_argument("--metrics-out", default="data/processed/region_cnn_metrics.tsv")
    region_cnn.add_argument("--predictions-out", default="data/processed/region_cnn_predictions.tsv")
    region_cnn.add_argument("--history-out", default="data/processed/region_cnn_training_history.tsv")
    region_cnn.add_argument(
        "--evaluation",
        action="append",
        choices=["repeated_random", "chromosome_holdout"],
        help="Evaluation design. Can repeat. Default: repeated_random.",
    )
    region_cnn.add_argument("--n-repeats", type=int, default=1)
    region_cnn.add_argument("--test-size", type=float, default=0.2)
    region_cnn.add_argument("--random-state", type=int, default=13)
    region_cnn.add_argument("--chromosome-column", default="chromosome")
    region_cnn.add_argument("--min-test-samples", type=int, default=50)
    region_cnn.add_argument("--max-length-5utr", type=int, default=512)
    region_cnn.add_argument("--max-length-cds", type=int, default=4096)
    region_cnn.add_argument("--max-length-3utr", type=int, default=4096)
    region_cnn.add_argument("--batch-size", type=int, default=64)
    region_cnn.add_argument("--max-epochs", type=int, default=30)
    region_cnn.add_argument("--patience", type=int, default=6)
    region_cnn.add_argument("--learning-rate", type=float, default=1e-3)
    region_cnn.add_argument("--weight-decay", type=float, default=1e-4)
    region_cnn.add_argument("--embedding-dim", type=int, default=8)
    region_cnn.add_argument("--channels", type=int, default=96)
    region_cnn.add_argument("--hidden-dim", type=int, default=192)
    region_cnn.add_argument("--tabular-hidden-dim", type=int, default=128)
    region_cnn.add_argument("--dropout", type=float, default=0.2)
    region_cnn.add_argument("--region-dropout", type=float, default=0.0)
    region_cnn.add_argument("--token-dropout", type=float, default=0.0)
    region_cnn.add_argument(
        "--crop-strategy",
        choices=["balanced", "start", "end", "random"],
        default="balanced",
    )
    region_cnn.add_argument("--device", default="cuda")

    saluki_like = subparsers.add_parser(
        "train-saluki-like",
        help="Train and evaluate a Saluki-like CNN+GRU RNA sequence model.",
    )
    saluki_like.add_argument("--table", default="data/processed/modeling_master_with_sequences.tsv")
    saluki_like.add_argument("--feature-table")
    saluki_like.add_argument("--target-column", default="target_label")
    saluki_like.add_argument("--metrics-out", default="data/processed/saluki_like_metrics.tsv")
    saluki_like.add_argument("--predictions-out", default="data/processed/saluki_like_predictions.tsv")
    saluki_like.add_argument("--history-out", default="data/processed/saluki_like_training_history.tsv")
    saluki_like.add_argument(
        "--evaluation",
        action="append",
        choices=["repeated_random", "chromosome_holdout"],
        help="Evaluation design. Can repeat. Default: repeated_random.",
    )
    saluki_like.add_argument("--n-repeats", type=int, default=1)
    saluki_like.add_argument("--test-size", type=float, default=0.2)
    saluki_like.add_argument("--random-state", type=int, default=13)
    saluki_like.add_argument("--chromosome-column", default="chromosome")
    saluki_like.add_argument("--min-test-samples", type=int, default=50)
    saluki_like.add_argument("--max-length-5utr", type=int, default=512)
    saluki_like.add_argument("--max-length-cds", type=int, default=2048)
    saluki_like.add_argument("--max-length-3utr", type=int, default=2048)
    saluki_like.add_argument("--batch-size", type=int, default=48)
    saluki_like.add_argument("--max-epochs", type=int, default=25)
    saluki_like.add_argument("--patience", type=int, default=5)
    saluki_like.add_argument("--learning-rate", type=float, default=5e-4)
    saluki_like.add_argument("--weight-decay", type=float, default=5e-4)
    saluki_like.add_argument("--embedding-dim", type=int, default=8)
    saluki_like.add_argument("--region-embedding-dim", type=int, default=4)
    saluki_like.add_argument("--channels", type=int, default=96)
    saluki_like.add_argument("--conv-pool-size", type=int, default=4)
    saluki_like.add_argument("--gru-hidden-dim", type=int, default=96)
    saluki_like.add_argument("--gru-layers", type=int, default=1)
    saluki_like.add_argument("--hidden-dim", type=int, default=192)
    saluki_like.add_argument("--tabular-hidden-dim", type=int, default=128)
    saluki_like.add_argument("--dropout", type=float, default=0.35)
    saluki_like.add_argument("--token-dropout", type=float, default=0.02)
    saluki_like.add_argument(
        "--crop-strategy",
        choices=["balanced", "start", "end", "random"],
        default="balanced",
    )
    saluki_like.add_argument("--device", default="cuda")

    sequence_transformer = subparsers.add_parser(
        "train-sequence-transformer",
        help="Train and evaluate a Conv-tokenized Transformer RNA sequence model.",
    )
    sequence_transformer.add_argument("--table", default="data/processed/modeling_master_with_sequences.tsv")
    sequence_transformer.add_argument("--feature-table")
    sequence_transformer.add_argument("--target-column", default="target_label")
    sequence_transformer.add_argument("--metrics-out", default="data/processed/sequence_transformer_metrics.tsv")
    sequence_transformer.add_argument(
        "--predictions-out", default="data/processed/sequence_transformer_predictions.tsv"
    )
    sequence_transformer.add_argument(
        "--history-out", default="data/processed/sequence_transformer_training_history.tsv"
    )
    sequence_transformer.add_argument(
        "--evaluation",
        action="append",
        choices=["repeated_random", "chromosome_holdout"],
        help="Evaluation design. Can repeat. Default: repeated_random.",
    )
    sequence_transformer.add_argument("--n-repeats", type=int, default=1)
    sequence_transformer.add_argument("--test-size", type=float, default=0.2)
    sequence_transformer.add_argument("--random-state", type=int, default=13)
    sequence_transformer.add_argument("--chromosome-column", default="chromosome")
    sequence_transformer.add_argument("--min-test-samples", type=int, default=50)
    sequence_transformer.add_argument("--max-length-5utr", type=int, default=256)
    sequence_transformer.add_argument("--max-length-cds", type=int, default=1024)
    sequence_transformer.add_argument("--max-length-3utr", type=int, default=1024)
    sequence_transformer.add_argument("--batch-size", type=int, default=48)
    sequence_transformer.add_argument("--max-epochs", type=int, default=20)
    sequence_transformer.add_argument("--patience", type=int, default=5)
    sequence_transformer.add_argument("--learning-rate", type=float, default=3e-4)
    sequence_transformer.add_argument("--weight-decay", type=float, default=5e-4)
    sequence_transformer.add_argument("--embedding-dim", type=int, default=8)
    sequence_transformer.add_argument("--region-embedding-dim", type=int, default=4)
    sequence_transformer.add_argument("--model-dim", type=int, default=128)
    sequence_transformer.add_argument("--conv-pool-size", type=int, default=4)
    sequence_transformer.add_argument("--transformer-layers", type=int, default=2)
    sequence_transformer.add_argument("--attention-heads", type=int, default=4)
    sequence_transformer.add_argument("--feedforward-dim", type=int, default=256)
    sequence_transformer.add_argument("--hidden-dim", type=int, default=192)
    sequence_transformer.add_argument("--tabular-hidden-dim", type=int, default=128)
    sequence_transformer.add_argument("--dropout", type=float, default=0.25)
    sequence_transformer.add_argument("--token-dropout", type=float, default=0.02)
    sequence_transformer.add_argument(
        "--crop-strategy",
        choices=["balanced", "start", "end", "random"],
        default="balanced",
    )
    sequence_transformer.add_argument("--device", default="cuda")

    rna_bert = subparsers.add_parser(
        "train-rna-bert",
        help="Train and evaluate a DNABERT-style k-mer RNA BERT encoder.",
    )
    rna_bert.add_argument("--table", default="data/processed/modeling_master_with_sequences.tsv")
    rna_bert.add_argument("--feature-table")
    rna_bert.add_argument("--target-column", default="target_label")
    rna_bert.add_argument("--metrics-out", default="data/processed/rna_bert_metrics.tsv")
    rna_bert.add_argument("--predictions-out", default="data/processed/rna_bert_predictions.tsv")
    rna_bert.add_argument("--history-out", default="data/processed/rna_bert_training_history.tsv")
    rna_bert.add_argument(
        "--evaluation",
        action="append",
        choices=["repeated_random", "chromosome_holdout"],
        help="Evaluation design. Can repeat. Default: repeated_random.",
    )
    rna_bert.add_argument("--n-repeats", type=int, default=1)
    rna_bert.add_argument("--test-size", type=float, default=0.2)
    rna_bert.add_argument("--random-state", type=int, default=13)
    rna_bert.add_argument("--chromosome-column", default="chromosome")
    rna_bert.add_argument("--min-test-samples", type=int, default=50)
    rna_bert.add_argument("--max-length-5utr", type=int, default=256)
    rna_bert.add_argument("--max-length-cds", type=int, default=1024)
    rna_bert.add_argument("--max-length-3utr", type=int, default=1024)
    rna_bert.add_argument("--kmer-size", type=int, default=4)
    rna_bert.add_argument("--kmer-stride", type=int, default=4)
    rna_bert.add_argument("--batch-size", type=int, default=32)
    rna_bert.add_argument("--max-epochs", type=int, default=20)
    rna_bert.add_argument("--patience", type=int, default=5)
    rna_bert.add_argument("--learning-rate", type=float, default=3e-4)
    rna_bert.add_argument("--weight-decay", type=float, default=5e-4)
    rna_bert.add_argument("--model-dim", type=int, default=128)
    rna_bert.add_argument("--region-embedding-dim", type=int, default=8)
    rna_bert.add_argument("--transformer-layers", type=int, default=2)
    rna_bert.add_argument("--attention-heads", type=int, default=4)
    rna_bert.add_argument("--feedforward-dim", type=int, default=256)
    rna_bert.add_argument("--hidden-dim", type=int, default=192)
    rna_bert.add_argument("--tabular-hidden-dim", type=int, default=128)
    rna_bert.add_argument("--dropout", type=float, default=0.25)
    rna_bert.add_argument("--token-dropout", type=float, default=0.02)
    rna_bert.add_argument(
        "--crop-strategy",
        choices=["balanced", "start", "end", "random"],
        default="balanced",
    )
    rna_bert.add_argument("--device", default="cuda")

    rna_lm = subparsers.add_parser(
        "extract-rna-lm-embeddings",
        help="Extract frozen HuggingFace RNA/DNA language-model embeddings.",
    )
    rna_lm.add_argument("--table", default="data/processed/modeling_master_with_sequences.tsv")
    rna_lm.add_argument("--out", default="data/processed/rna_lm_embeddings.tsv")
    rna_lm.add_argument("--model-name-or-path", required=True)
    rna_lm.add_argument("--sequence-column", default="sequence_full")
    rna_lm.add_argument("--target-column", default="target_label")
    rna_lm.add_argument("--sequence-format", choices=["raw", "spaced_chars", "kmer"], default="raw")
    rna_lm.add_argument("--alphabet", choices=["rna", "dna"], default="rna")
    rna_lm.add_argument("--kmer-size", type=int, default=6)
    rna_lm.add_argument("--kmer-stride", type=int, default=1)
    rna_lm.add_argument("--max-length", type=int, default=512)
    rna_lm.add_argument("--chunk-size", type=int, default=1024)
    rna_lm.add_argument("--chunk-stride", type=int, default=1024)
    rna_lm.add_argument("--batch-size", type=int, default=8)
    rna_lm.add_argument("--device", default="cuda")
    rna_lm.add_argument("--trust-remote-code", action="store_true")
    rna_lm.add_argument("--local-files-only", action="store_true", help="Load HuggingFace model/tokenizer from local cache only.")
    rna_lm.add_argument("--disable-safetensors", action="store_true", help="Prefer PyTorch weights and skip safetensors lookup.")
    rna_lm.add_argument("--limit", type=int, help="Optional row limit for smoke tests.")
    rna_lm.add_argument("--resume", action="store_true", help="Skip gene_id rows already present in --out.")
    rna_lm.add_argument("--flush-every", type=int, default=100, help="Append embeddings every N rows.")

    rna_lm_merge = subparsers.add_parser(
        "merge-rna-lm-region-embeddings",
        help="Merge separately extracted 5'UTR/CDS/3'UTR RNA LM embeddings.",
    )
    rna_lm_merge.add_argument("--utr5", required=True, help="5'UTR embedding TSV.")
    rna_lm_merge.add_argument("--cds", required=True, help="CDS embedding TSV.")
    rna_lm_merge.add_argument("--utr3", required=True, help="3'UTR embedding TSV.")
    rna_lm_merge.add_argument("--out", default="data/processed/rna_lm_nucleotide_transformer_multi_region_embeddings.tsv")
    rna_lm_merge.add_argument("--target-column", default="target_label")
    rna_lm_merge.add_argument("--join", choices=["inner", "outer"], default="inner")

    summarize = subparsers.add_parser("summarize-targets", help="Summarize target QC and variability.")
    summarize.add_argument("--targets", default="data/processed/stability_targets.tsv")
    summarize.add_argument("--cell-out", default="data/processed/qc_cell_line_summary.tsv")
    summarize.add_argument("--gene-out", default="data/processed/gene_stability_variability.tsv")
    summarize.add_argument(
        "--target-columns",
        nargs="*",
        default=["log2_stability_2h_0h", "log2_stability_6h_2h", "log2_stability_6h_0h"],
    )

    consensus = subparsers.add_parser(
        "build-consensus-targets",
        help="Collapse gene x cell-line targets to one context-agnostic label per gene.",
    )
    consensus.add_argument("--targets", default="data/processed/stability_targets.tsv")
    consensus.add_argument("--out", default="data/processed/stability_consensus_targets.tsv")
    consensus.add_argument("--target-column", default="log2_stability_6h_2h")
    consensus.add_argument("--min-cell-lines", type=int, default=8)

    compare = subparsers.add_parser("compare-targets", help="Compare two target tables on shared gene x cell-line rows.")
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    compare.add_argument("--out", required=True)
    compare.add_argument("--left-name", default="left")
    compare.add_argument("--right-name", default="right")
    compare.add_argument("--target-column", default="log2_stability_6h_2h")

    replicate = subparsers.add_parser("replicate-qc", help="Summarize biological replicate agreement.")
    replicate.add_argument("--files", required=True)
    replicate.add_argument("--gene-out", default="data/processed/qc_replicate_gene_signal_gene_sense.tsv")
    replicate.add_argument("--experiment-out", default="data/processed/qc_replicate_experiment_gene_sense.tsv")
    replicate.add_argument("--feature-type", default="gene_sense")
    replicate.add_argument("--value-column", default="rpkm")
    replicate.add_argument("--pseudocount", type=float, default=0.1)
    replicate.add_argument("--min-mean-signal", type=float, default=0.5)
    replicate.add_argument("--max-log2-range", type=float, default=1.0)

    master = subparsers.add_parser(
        "build-modeling-master",
        help="Build the first-stage context-agnostic modeling master table.",
    )
    master.add_argument("--consensus", default="data/processed/stability_consensus_targets_gene_sense.tsv")
    master.add_argument("--replicate-qc", default="data/processed/qc_replicate_gene_signal_gene_sense.tsv")
    master.add_argument("--out", default="data/processed/modeling_master_table.tsv")
    master.add_argument("--target-label-column", default="stability_consensus_median")
    master.add_argument("--target-label-name", default="gene_sense_log2_stability_6h_2h_consensus_median")
    master.add_argument("--source-dataset", default="ENCODE_Ljungman_BrU_BruChase_gene_sense")

    transcript = subparsers.add_parser(
        "build-transcript-sequences",
        help="Build gene-level canonical transcript, UTR, and CDS sequence table from GENCODE.",
    )
    transcript.add_argument("--master", default="data/processed/modeling_master_table.tsv")
    transcript.add_argument("--gtf", required=True)
    transcript.add_argument("--transcript-fasta", required=True)
    transcript.add_argument("--out", default="data/processed/transcript_sequences.tsv")

    merge_seq = subparsers.add_parser(
        "merge-master-sequences",
        help="Merge modeling master table with canonical transcript sequences.",
    )
    merge_seq.add_argument("--master", default="data/processed/modeling_master_table.tsv")
    merge_seq.add_argument("--sequences", default="data/processed/transcript_sequences.tsv")
    merge_seq.add_argument("--out", default="data/processed/modeling_master_with_sequences.tsv")

    compact_features = subparsers.add_parser(
        "make-compact-sequence-features",
        help="Build compact region-level sequence features from the modeling table with sequences.",
    )
    compact_features.add_argument("--table", default="data/processed/modeling_master_with_sequences.tsv")
    compact_features.add_argument("--config", default="configs/project.yaml")
    compact_features.add_argument("--out", default="data/processed/sequence_model_features.tsv")
    compact_features.add_argument("--target-column", default="target_label")
    compact_features.add_argument("--region", action="append", help="Region to include: full, 5utr, cds, or 3utr.")
    compact_features.add_argument("--k", action="append", type=int, help="k-mer length to include. Can repeat.")

    merge_features = subparsers.add_parser("merge-feature-tables", help="Merge two gene-level feature TSV tables.")
    merge_features.add_argument("--left", required=True)
    merge_features.add_argument("--right", required=True)
    merge_features.add_argument("--out", required=True)
    merge_features.add_argument("--key", default="gene_id")
    merge_features.add_argument("--target-column", default="target_label")
    merge_features.add_argument("--right-prefix", default="")
    merge_features.add_argument("--how", choices=["inner", "left", "right", "outer"], default="inner")

    figures = subparsers.add_parser("make-progress-figures", help="Create progress and result visualization figures.")
    figures.add_argument("--processed-dir", default="data/processed")
    figures.add_argument("--out-dir", default="docs/figures")

    visual_report = subparsers.add_parser("write-visual-report", help="Create figures and a Markdown visual report.")
    visual_report.add_argument("--processed-dir", default="data/processed")
    visual_report.add_argument("--figure-dir", default="docs/figures")
    visual_report.add_argument("--out", default="docs/progress_visual_report.md")

    grammar_report = subparsers.add_parser(
        "write-grammar-report",
        help="Create unified model leaderboard and RNA stability grammar interpretation report.",
    )
    grammar_report.add_argument("--processed-dir", default="data/processed")
    grammar_report.add_argument("--figure-dir", default="docs/figures")
    grammar_report.add_argument("--out", default="docs/rna_stability_grammar_interpretation_report.md")

    args = parser.parse_args(argv)
    if args.command == "discover-encode":
        run_discover(args)
    elif args.command == "download-files":
        run_download(args)
    elif args.command == "build-targets":
        run_build_targets(args)
    elif args.command == "make-sequence-features":
        run_make_sequence_features(args)
    elif args.command == "train-baseline":
        run_train_baseline(args)
    elif args.command == "evaluate-sequence-models":
        run_evaluate_sequence_models(args)
    elif args.command == "train-region-cnn":
        run_train_region_cnn(args)
    elif args.command == "train-saluki-like":
        run_train_saluki_like(args)
    elif args.command == "train-sequence-transformer":
        run_train_sequence_transformer(args)
    elif args.command == "train-rna-bert":
        run_train_rna_bert(args)
    elif args.command == "extract-rna-lm-embeddings":
        run_extract_rna_lm_embeddings(args)
    elif args.command == "merge-rna-lm-region-embeddings":
        run_merge_rna_lm_region_embeddings(args)
    elif args.command == "summarize-targets":
        run_summarize_targets(args)
    elif args.command == "build-consensus-targets":
        run_build_consensus_targets(args)
    elif args.command == "compare-targets":
        run_compare_targets(args)
    elif args.command == "replicate-qc":
        run_replicate_qc(args)
    elif args.command == "build-modeling-master":
        run_build_modeling_master(args)
    elif args.command == "build-transcript-sequences":
        run_build_transcript_sequences(args)
    elif args.command == "merge-master-sequences":
        run_merge_master_sequences(args)
    elif args.command == "make-compact-sequence-features":
        run_make_compact_sequence_features(args)
    elif args.command == "merge-feature-tables":
        run_merge_feature_tables(args)
    elif args.command == "make-progress-figures":
        run_make_progress_figures(args)
    elif args.command == "write-visual-report":
        run_write_visual_report(args)
    elif args.command == "write-grammar-report":
        run_write_grammar_report(args)


def run_discover(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    encode_config = config.get("encode", {})
    query = EncodeQuery(
        base_url=encode_config.get("base_url", "https://www.encodeproject.org"),
        lab_title=encode_config.get("lab_title", "Mats Ljungman, UMichigan"),
        status=encode_config.get("status", "released"),
        series_type=encode_config.get("series_type", "PulseChaseTimeSeries"),
    )
    series, experiments = discover_pulse_chase_series(
        query,
        expected_terms=expected_encode_terms(config),
        aliases=cell_line_aliases(config),
    )
    Path(args.series_out).parent.mkdir(parents=True, exist_ok=True)
    series.to_csv(args.series_out, sep="\t", index=False)

    file_filters = encode_config.get("file_filters", {})
    file_formats = set(args.file_format or file_filters.get("file_format", [])) or None
    output_types = set(args.output_type or file_filters.get("output_type", [])) or None
    files = collect_experiment_files(
        experiments["experiment_accession"].unique(),
        query=query,
        file_formats=file_formats,
        output_types=output_types,
    )
    if not files.empty:
        files["paper_name"] = files["cell_line"].map(cell_line_aliases(config)).fillna(files["cell_line"])
    Path(args.files_out).parent.mkdir(parents=True, exist_ok=True)
    files.to_csv(args.files_out, sep="\t", index=False)
    print(f"Wrote {len(series)} series to {args.series_out}")
    print(f"Wrote {len(files)} files to {args.files_out}")


def run_download(args: argparse.Namespace) -> None:
    files = pd.read_csv(args.files, sep="\t")
    updated = download_files(
        files,
        args.out_dir,
        file_format=args.file_format,
        output_type=args.output_type,
        overwrite=args.overwrite,
        workers=args.workers,
    )
    updated.to_csv(args.files, sep="\t", index=False)
    print(f"Updated local paths in {args.files}")


def run_build_targets(args: argparse.Namespace) -> None:
    targets = build_targets_from_manifest(
        args.files,
        feature_type=args.feature_type,
        value_column=args.value_column,
        pseudocount=args.pseudocount,
        min_signal_0h=args.min_signal_0h,
        min_cell_lines_per_gene=args.min_cell_lines_per_gene,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    targets.to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {len(targets)} target rows to {args.out}")


def run_make_sequence_features(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    seq_config = config.get("sequence_features", {})
    features = sequence_feature_table(
        args.fasta,
        ks=seq_config.get("kmer_ks", [3, 4, 5, 6]),
        motifs=seq_config.get("motifs", {}),
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {len(features)} sequence rows to {args.out}")


def run_train_baseline(args: argparse.Namespace) -> None:
    features = pd.read_csv(args.features, sep="\t")
    metrics = train_baseline(
        features,
        target_column=args.target_column,
        model_name=args.model,
        group_column=args.group_column,
        leave_group=args.leave_group,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_metrics(metrics, args.out)
    print(f"Wrote metrics to {args.out}")


def run_evaluate_sequence_models(args: argparse.Namespace) -> None:
    metrics, predictions, summary, importances = write_sequence_model_evaluation(
        args.features,
        metrics_out=args.metrics_out,
        predictions_out=args.predictions_out,
        summary_out=args.summary_out,
        importance_out=args.importance_out,
        target_column=args.target_column,
        models=args.model or ["elasticnet"],
        feature_sets=args.feature_set or ["all"],
        evaluations=args.evaluation or ["repeated_random", "chromosome_holdout"],
        n_repeats=args.n_repeats,
        test_size=args.test_size,
        random_state=args.random_state,
        chromosome_column=args.chromosome_column,
        min_test_samples=args.min_test_samples,
        preprocessing=args.preprocessing,
        pca_components=args.pca_components,
    )
    print(f"Wrote {len(metrics)} evaluation metric rows to {args.metrics_out}")
    print(f"Wrote {len(predictions)} prediction rows to {args.predictions_out}")
    print(f"Wrote {len(summary)} summary rows to {args.summary_out}")
    if args.importance_out:
        print(f"Wrote {len(importances)} feature importance rows to {args.importance_out}")


def run_train_region_cnn(args: argparse.Namespace) -> None:
    metrics, predictions, history = write_region_cnn_evaluation(
        args.table,
        metrics_out=args.metrics_out,
        predictions_out=args.predictions_out,
        history_out=args.history_out,
        feature_table_path=args.feature_table,
        target_column=args.target_column,
        evaluations=args.evaluation or ["repeated_random"],
        n_repeats=args.n_repeats,
        test_size=args.test_size,
        random_state=args.random_state,
        chromosome_column=args.chromosome_column,
        min_test_samples=args.min_test_samples,
        region_lengths=RegionLengths(
            utr5=args.max_length_5utr,
            cds=args.max_length_cds,
            utr3=args.max_length_3utr,
        ),
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        embedding_dim=args.embedding_dim,
        channels=args.channels,
        hidden_dim=args.hidden_dim,
        tabular_hidden_dim=args.tabular_hidden_dim,
        dropout=args.dropout,
        region_dropout=args.region_dropout,
        token_dropout=args.token_dropout,
        crop_strategy=args.crop_strategy,
        device=args.device,
    )
    print(f"Wrote {len(metrics)} region CNN metric rows to {args.metrics_out}")
    print(f"Wrote {len(predictions)} region CNN prediction rows to {args.predictions_out}")
    print(f"Wrote {len(history)} training history rows to {args.history_out}")


def run_train_saluki_like(args: argparse.Namespace) -> None:
    metrics, predictions, history = write_saluki_like_evaluation(
        args.table,
        metrics_out=args.metrics_out,
        predictions_out=args.predictions_out,
        history_out=args.history_out,
        feature_table_path=args.feature_table,
        target_column=args.target_column,
        evaluations=args.evaluation or ["repeated_random"],
        n_repeats=args.n_repeats,
        test_size=args.test_size,
        random_state=args.random_state,
        chromosome_column=args.chromosome_column,
        min_test_samples=args.min_test_samples,
        region_lengths=RegionLengths(
            utr5=args.max_length_5utr,
            cds=args.max_length_cds,
            utr3=args.max_length_3utr,
        ),
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        embedding_dim=args.embedding_dim,
        region_embedding_dim=args.region_embedding_dim,
        channels=args.channels,
        conv_pool_size=args.conv_pool_size,
        gru_hidden_dim=args.gru_hidden_dim,
        gru_layers=args.gru_layers,
        hidden_dim=args.hidden_dim,
        tabular_hidden_dim=args.tabular_hidden_dim,
        dropout=args.dropout,
        token_dropout=args.token_dropout,
        crop_strategy=args.crop_strategy,
        device=args.device,
    )
    print(f"Wrote {len(metrics)} Saluki-like metric rows to {args.metrics_out}")
    print(f"Wrote {len(predictions)} Saluki-like prediction rows to {args.predictions_out}")
    print(f"Wrote {len(history)} training history rows to {args.history_out}")


def run_train_sequence_transformer(args: argparse.Namespace) -> None:
    metrics, predictions, history = write_sequence_transformer_evaluation(
        args.table,
        metrics_out=args.metrics_out,
        predictions_out=args.predictions_out,
        history_out=args.history_out,
        feature_table_path=args.feature_table,
        target_column=args.target_column,
        evaluations=args.evaluation or ["repeated_random"],
        n_repeats=args.n_repeats,
        test_size=args.test_size,
        random_state=args.random_state,
        chromosome_column=args.chromosome_column,
        min_test_samples=args.min_test_samples,
        region_lengths=RegionLengths(
            utr5=args.max_length_5utr,
            cds=args.max_length_cds,
            utr3=args.max_length_3utr,
        ),
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        embedding_dim=args.embedding_dim,
        region_embedding_dim=args.region_embedding_dim,
        model_dim=args.model_dim,
        conv_pool_size=args.conv_pool_size,
        transformer_layers=args.transformer_layers,
        attention_heads=args.attention_heads,
        feedforward_dim=args.feedforward_dim,
        hidden_dim=args.hidden_dim,
        tabular_hidden_dim=args.tabular_hidden_dim,
        dropout=args.dropout,
        token_dropout=args.token_dropout,
        crop_strategy=args.crop_strategy,
        device=args.device,
    )
    print(f"Wrote {len(metrics)} Transformer metric rows to {args.metrics_out}")
    print(f"Wrote {len(predictions)} Transformer prediction rows to {args.predictions_out}")
    print(f"Wrote {len(history)} training history rows to {args.history_out}")


def run_train_rna_bert(args: argparse.Namespace) -> None:
    metrics, predictions, history = write_rna_bert_evaluation(
        args.table,
        metrics_out=args.metrics_out,
        predictions_out=args.predictions_out,
        history_out=args.history_out,
        feature_table_path=args.feature_table,
        target_column=args.target_column,
        evaluations=args.evaluation or ["repeated_random"],
        n_repeats=args.n_repeats,
        test_size=args.test_size,
        random_state=args.random_state,
        chromosome_column=args.chromosome_column,
        min_test_samples=args.min_test_samples,
        region_lengths=RegionLengths(
            utr5=args.max_length_5utr,
            cds=args.max_length_cds,
            utr3=args.max_length_3utr,
        ),
        kmer_size=args.kmer_size,
        kmer_stride=args.kmer_stride,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        model_dim=args.model_dim,
        region_embedding_dim=args.region_embedding_dim,
        transformer_layers=args.transformer_layers,
        attention_heads=args.attention_heads,
        feedforward_dim=args.feedforward_dim,
        hidden_dim=args.hidden_dim,
        tabular_hidden_dim=args.tabular_hidden_dim,
        dropout=args.dropout,
        token_dropout=args.token_dropout,
        crop_strategy=args.crop_strategy,
        device=args.device,
    )
    print(f"Wrote {len(metrics)} RNA BERT metric rows to {args.metrics_out}")
    print(f"Wrote {len(predictions)} RNA BERT prediction rows to {args.predictions_out}")
    print(f"Wrote {len(history)} training history rows to {args.history_out}")


def run_extract_rna_lm_embeddings(args: argparse.Namespace) -> None:
    embeddings = write_rna_lm_embeddings(
        args.table,
        out=args.out,
        model_name_or_path=args.model_name_or_path,
        sequence_column=args.sequence_column,
        target_column=args.target_column,
        sequence_format=args.sequence_format,
        alphabet=args.alphabet,
        kmer_size=args.kmer_size,
        kmer_stride=args.kmer_stride,
        max_length=args.max_length,
        chunk_size=args.chunk_size,
        chunk_stride=args.chunk_stride,
        batch_size=args.batch_size,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        disable_safetensors=args.disable_safetensors,
        limit=args.limit,
        resume=args.resume,
        flush_every=args.flush_every,
    )
    print(f"Wrote {len(embeddings)} RNA LM embedding rows to {args.out}")


def run_merge_rna_lm_region_embeddings(args: argparse.Namespace) -> None:
    embeddings = write_multi_region_rna_lm_embeddings(
        utr5_path=args.utr5,
        cds_path=args.cds,
        utr3_path=args.utr3,
        out=args.out,
        target_column=args.target_column,
        join=args.join,
    )
    print(f"Wrote {len(embeddings)} multi-region RNA LM embedding rows to {args.out}")


def run_summarize_targets(args: argparse.Namespace) -> None:
    cell_summary, gene_summary = write_target_summaries(
        args.targets,
        cell_out=args.cell_out,
        gene_out=args.gene_out,
        target_columns=args.target_columns,
    )
    print(f"Wrote {len(cell_summary)} cell-line summary rows to {args.cell_out}")
    print(f"Wrote {len(gene_summary)} gene variability rows to {args.gene_out}")


def run_build_consensus_targets(args: argparse.Namespace) -> None:
    consensus = write_consensus_targets(
        args.targets,
        out=args.out,
        target_column=args.target_column,
        min_cell_lines=args.min_cell_lines,
    )
    print(f"Wrote {len(consensus)} consensus target rows to {args.out}")


def run_compare_targets(args: argparse.Namespace) -> None:
    summary = write_target_comparison(
        args.left,
        args.right,
        out=args.out,
        left_name=args.left_name,
        right_name=args.right_name,
        target_column=args.target_column,
    )
    print(f"Wrote {len(summary)} target comparison rows to {args.out}")


def run_replicate_qc(args: argparse.Namespace) -> None:
    gene_qc, experiment_qc = write_replicate_qc(
        args.files,
        gene_out=args.gene_out,
        experiment_out=args.experiment_out,
        feature_type=args.feature_type,
        value_column=args.value_column,
        pseudocount=args.pseudocount,
        min_mean_signal=args.min_mean_signal,
        max_log2_range=args.max_log2_range,
    )
    print(f"Wrote {len(gene_qc)} gene-level replicate QC rows to {args.gene_out}")
    print(f"Wrote {len(experiment_qc)} experiment-level replicate QC rows to {args.experiment_out}")


def run_build_modeling_master(args: argparse.Namespace) -> None:
    table = write_modeling_master_table(
        args.consensus,
        out=args.out,
        replicate_qc_path=args.replicate_qc,
        target_label_column=args.target_label_column,
        target_label_name=args.target_label_name,
        source_dataset=args.source_dataset,
    )
    print(f"Wrote {len(table)} modeling rows to {args.out}")


def run_build_transcript_sequences(args: argparse.Namespace) -> None:
    table = write_transcript_sequence_table(
        args.master,
        gtf_path=args.gtf,
        transcript_fasta_path=args.transcript_fasta,
        out=args.out,
    )
    mapped = int((table["sequence_status"] == "mapped").sum()) if "sequence_status" in table else 0
    print(f"Wrote {len(table)} transcript sequence rows to {args.out}")
    print(f"Mapped {mapped} rows to canonical transcript sequences")


def run_merge_master_sequences(args: argparse.Namespace) -> None:
    table = write_modeling_table_with_sequences(
        args.master,
        args.sequences,
        out=args.out,
    )
    mapped = int((table["sequence_status"] == "mapped").sum()) if "sequence_status" in table else 0
    print(f"Wrote {len(table)} modeling rows with sequences to {args.out}")
    print(f"Mapped {mapped} rows to canonical transcript sequences")


def run_make_compact_sequence_features(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    seq_config = config.get("sequence_features", {})
    table = write_compact_sequence_model_features(
        args.table,
        out=args.out,
        target_column=args.target_column,
        regions=args.region or ["full", "5utr", "cds", "3utr"],
        ks=args.k or [3, 4],
        motifs=seq_config.get("motifs", {}),
    )
    print(f"Wrote {len(table)} sequence feature rows to {args.out}")
    print(f"Wrote {len(table.columns)} columns")


def run_merge_feature_tables(args: argparse.Namespace) -> None:
    table = write_merged_feature_table(
        args.left,
        args.right,
        out=args.out,
        key=args.key,
        target_column=args.target_column,
        right_prefix=args.right_prefix,
        how=args.how,
    )
    print(f"Wrote {len(table)} merged feature rows to {args.out}")
    print(f"Wrote {len(table.columns)} columns")


def run_make_progress_figures(args: argparse.Namespace) -> None:
    paths = make_progress_figures(processed_dir=args.processed_dir, out_dir=args.out_dir)
    for name, path in paths.items():
        print(f"Wrote {name} figure to {path}")


def run_write_visual_report(args: argparse.Namespace) -> None:
    paths = make_progress_figures(processed_dir=args.processed_dir, out_dir=args.figure_dir)
    report = write_progress_report(
        out=args.out,
        figure_dir=args.figure_dir,
        processed_dir=args.processed_dir,
    )
    for name, path in paths.items():
        print(f"Wrote {name} figure to {path}")
    print(f"Wrote visual report to {report}")


def run_write_grammar_report(args: argparse.Namespace) -> None:
    paths = write_leaderboard_and_grammar_report(
        processed_dir=args.processed_dir,
        figure_dir=args.figure_dir,
        report_out=args.out,
    )
    for name, path in paths.items():
        print(f"Wrote {name} to {path}")


if __name__ == "__main__":
    main()
