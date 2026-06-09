# RAMHT Improvement Experiments

This note records the RAMHT architecture and performance experiments run after the first all-split
GPU benchmark.

## Architecture Changes Implemented

The RAMHT implementation now supports:

- optional padding-aware within-region Transformer attention;
- a separate codon stream that is not pooled into the nucleotide stream;
- task-specific sequence/codon/engineered fusion gates;
- gated residual concatenation fusion;
- a direct engineered-feature output skip;
- configurable learning rate and architecture flags in both GPU launchers.

The original architecture remains the default so the earlier benchmark is reproducible.

## Neural Architecture Screening

All screening runs used the three fixed repeated-random splits with the reduced-window RTX 2080 Ti
configuration. The original RAMHT Pearson means were:

| Label | Original RAMHT |
| --- | ---: |
| `gene_sense_late_chase_6h_2h` | 0.455 |
| `gene_sense_total_chase_6h_0h` | 0.551 |
| `exon_sense_late_chase_6h_2h` | 0.479 |
| `exon_sense_total_chase_6h_0h` | 0.752 |

Main findings:

- padding-aware attention reduced performance, especially for gene late;
- task-specific gates and the full v2 architecture did not improve the four-label mean;
- gated residual fusion improved gene total slightly (`0.551` to `0.556`) but was not uniformly better;
- a direct engineered-feature output skip strongly hurt performance, consistent with overfitting the
  1,336-dimensional engineered feature vector.

The tested neural variants therefore do not justify replacing the original RAMHT configuration.

## RAMHT-XGBoost Fixed Blend

Because RAMHT and XGBoost have complementary errors, a fixed blend was evaluated using identical
per-gene test predictions:

```text
y_blend = 0.2 * y_RAMHT + 0.8 * y_XGBoost
```

The weight is fixed globally across all labels and evaluation schemes. It improves Pearson mean over
XGBoost alone in every comparison:

| Label | Evaluation | XGBoost | Selected blend | Delta |
| --- | --- | ---: | ---: | ---: |
| `gene_sense_late_chase_6h_2h` | chromosome holdout | 0.497 | 0.502 | +0.005 |
| `gene_sense_total_chase_6h_0h` | chromosome holdout | 0.536 | 0.546 | +0.010 |
| `exon_sense_late_chase_6h_2h` | chromosome holdout | 0.557 | 0.561 | +0.004 |
| `exon_sense_total_chase_6h_0h` | chromosome holdout | 0.772 | 0.774 | +0.002 |
| `gene_sense_late_chase_6h_2h` | repeated random | 0.512 | 0.517 | +0.004 |
| `gene_sense_total_chase_6h_0h` | repeated random | 0.579 | 0.586 | +0.007 |
| `exon_sense_late_chase_6h_2h` | repeated random | 0.547 | 0.548 | +0.002 |
| `exon_sense_total_chase_6h_0h` | repeated random | 0.778 | 0.780 | +0.002 |

Outputs:

- `data/processed/ramht_xgboost_blend_metrics.tsv`
- `data/processed/ramht_xgboost_blend_predictions.tsv`
- `data/processed/ramht_xgboost_blend_summary.tsv`
- `data/processed/ramht_xgboost_blend_selected_summary.tsv`
- `data/processed/ramht_xgboost_blend_selected_predictions.tsv`

## Recommended Framing

The current strongest result is a fixed RAMHT-XGBoost ensemble rather than standalone RAMHT. The
consistent gain across all labels and evaluation schemes supports the claim that RAMHT captures
sequence-derived information complementary to engineered-feature tree models.
