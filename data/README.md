# Data Layout

Large files are not tracked by git.

- `raw/encode/`: downloaded ENCODE TSV/BAM/bigWig/FASTQ files.
- `raw/expression/`: RNA-seq, RBP expression, miRNA expression matrices.
- `raw/binding/`: eCLIP/CLIP peaks and binding annotations.
- `external/`: reference FASTA, GTF, motif databases.
- `interim/`: manifests and normalized intermediate tables.
- `processed/`: final modeling matrices, targets, metrics, predictions, and interpretation tables.

Current high-level outputs:

- `parallel_label_quality_summary.tsv`: strict four-label QC and coverage.
- `parallel_label_feature_tables.tsv`: paths and metadata for four engineered-feature tables.
- `parallel_deep_sequence_tables.tsv`: paths and metadata for four raw-sequence tables.
- `parallel_deep_gpu_full_summary.tsv`: GPU-full raw-sequence model results.
- `parallel_model_suite_summary.tsv`: combined model-suite results.
- `figure_source_data/`: source tables for current project figures.

The complete raw and processed data are intentionally not tracked by git.
