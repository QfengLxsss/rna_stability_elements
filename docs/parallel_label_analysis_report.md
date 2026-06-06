# Parallel RNA Stability Label Analysis

当前阶段总结与下一步请参阅 [current_results.md](current_results.md)。

This report summarizes the stricter upfront target processing and four-way label comparison:

- `gene_sense_late_chase_6h_2h`
- `gene_sense_total_chase_6h_0h`
- `exon_sense_late_chase_6h_2h`
- `exon_sense_total_chase_6h_0h`

## Upfront Processing Changes

The previous first-pass target builder averaged biological replicates before computing time-point ratios. The new analysis adds a stricter target layer:

1. Pair biological replicates across 0h, 2h and 6h before calculating ratios.
2. Require `signal_0h >= 0.5` for all target definitions.
3. Require `signal_2h >= 0.5` for the `6h/2h` label to avoid unstable low-denominator ratios.
4. Flag gene-cell labels with a replicate target span greater than 1.0 log2 unit.
5. Build strict consensus labels from pass-only gene-cell targets.
6. Add sample-level PCA QC for `gene_sense` and `exon_sense` signal matrices.

Core implementation:

- `src/rna_stability_elements/target_quality.py`
- `scripts/run_four_way_label_analysis.py`
- `scripts/summarize_parallel_label_models.py`

## Output Tables

Main QC and model summaries:

- `data/processed/parallel_label_quality_summary.tsv`
- `data/processed/parallel_label_model_comparison.tsv`
- `data/processed/parallel_label_importance_overlap.tsv`
- `data/processed/parallel_label_cross_cell_consistency.tsv`
- `data/processed/parallel_label_signal_correlations.tsv`

Four strict feature tables:

- `data/processed/parallel_sequence_model_features_gene_sense_late_chase_6h_2h.tsv`
- `data/processed/parallel_sequence_model_features_gene_sense_total_chase_6h_0h.tsv`
- `data/processed/parallel_sequence_model_features_exon_sense_late_chase_6h_2h.tsv`
- `data/processed/parallel_sequence_model_features_exon_sense_total_chase_6h_0h.tsv`

Figures:

- `docs/figures/parallel_label_qc_summary.png`
- `docs/figures/parallel_label_model_comparison.png`
- `docs/figures/sample_signal_pca_gene_sense.png`
- `docs/figures/sample_signal_pca_exon_sense.png`

## Strict Label Coverage

| label | strict consensus genes | feature rows |
| --- | ---: | ---: |
| gene_sense_late_chase_6h_2h | 8,428 | 8,428 |
| gene_sense_total_chase_6h_0h | 9,848 | 9,848 |
| exon_sense_late_chase_6h_2h | 9,765 | 9,018 |
| exon_sense_total_chase_6h_0h | 10,678 | 9,881 |

The exon-sense consensus has some genes outside the current base sequence-feature universe, so model feature rows are lower than consensus rows.

## ElasticNet Parallel Evaluation

All four labels were evaluated with the same compact sequence features and the same split design:

- 3 repeated random gene splits.
- chromosome holdout splits.
- ElasticNet, feature set `all`.

| label | repeated Pearson | chromosome Pearson | repeated R2 | chromosome R2 |
| --- | ---: | ---: | ---: | ---: |
| gene_sense_late_chase_6h_2h | 0.474 | 0.437 | 0.225 | 0.188 |
| gene_sense_total_chase_6h_0h | 0.466 | 0.479 | 0.194 | 0.225 |
| exon_sense_late_chase_6h_2h | 0.494 | 0.491 | 0.243 | 0.240 |
| exon_sense_total_chase_6h_0h | 0.757 | 0.758 | 0.573 | 0.573 |

## Interpretation

`exon_sense_total_chase_6h_0h` is by far the easiest label to predict from compact sequence features. This does not automatically make it the best biological RNA-stability label: it may capture mature exonic RNA retention, processing and abundance-linked sequence signals in addition to decay.

The `late_chase_6h_2h` labels are less predictable but more conservative. Their feature importance is also more consistent between `gene_sense` and `exon_sense`: top-100 overlap is 60 features, with signed coefficient Pearson around 0.81.

Practical recommendation:

- Keep all four labels as parallel endpoints.
- Treat shared signals across `gene_sense_late_chase_6h_2h` and `exon_sense_late_chase_6h_2h` as high-confidence late-chase stability grammar.
- Treat strong `exon_sense_total_chase_6h_0h` signals as a broader mature-exonic retention grammar until further validated.
- Use the four-way overlap table to separate robust grammar from label-specific biology.
