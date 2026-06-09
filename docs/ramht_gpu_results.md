# RAMHT GPU Results

This note records the GPU work completed after `docs/ramht_gpu_handoff.md`.

## Runs

### RAMHT 2080Ti reduced-window baseline

Command family:

```bash
GPUS=0,1,2,3,4,5,6,7 \
RUN_ID=ramht_2080ti_parallel \
SPLIT_SET=all \
BATCH_SIZE=4 \
MODEL_DIM=128 \
TRANSFORMER_LAYERS=2 \
CODON_LAYERS=1 \
FEEDFORWARD_DIM=256 \
FEATURE_HIDDEN_DIM=256 \
HEAD_HIDDEN_DIM=128 \
MAX_LENGTH_5UTR=128 \
MAX_LENGTH_CDS=768 \
MAX_LENGTH_3UTR=768 \
CODON_LENGTH=256 \
scripts/launch_ramht_gpu_splits.sh
```

Outputs:

- `data/processed/ramht_2080ti_parallel_metrics.tsv`
- `data/processed/ramht_2080ti_parallel_predictions.tsv`
- `data/processed/ramht_2080ti_parallel_history.tsv`
- `data/processed/ramht_2080ti_parallel_summary.tsv`

### RAMHT wide-window batch-2 tuning

Command family:

```bash
GPUS=0,1,2,3,4,5,6,7 \
RUN_ID=ramht_2080ti_wide_b2_parallel \
SPLIT_SET=all \
BATCH_SIZE=2 \
MODEL_DIM=128 \
TRANSFORMER_LAYERS=2 \
CODON_LAYERS=1 \
FEEDFORWARD_DIM=256 \
FEATURE_HIDDEN_DIM=256 \
HEAD_HIDDEN_DIM=128 \
MAX_LENGTH_5UTR=256 \
MAX_LENGTH_CDS=1024 \
MAX_LENGTH_3UTR=1024 \
CODON_LENGTH=342 \
scripts/launch_ramht_gpu_splits.sh
```

Outputs:

- `data/processed/ramht_2080ti_wide_b2_parallel_metrics.tsv`
- `data/processed/ramht_2080ti_wide_b2_parallel_predictions.tsv`
- `data/processed/ramht_2080ti_wide_b2_parallel_history.tsv`
- `data/processed/ramht_2080ti_wide_b2_parallel_summary.tsv`

## Main Comparison

Pearson means are shown below. The stronger RAMHT setting for each row is marked in the last column.

| Label | Evaluation | Reduced | Wide B2 | XGBoost | Best Raw Deep | Best Hybrid | RAMHT Pick |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `gene_sense_late_chase_6h_2h` | chromosome | 0.457 | 0.444 | 0.497 | 0.428 | 0.475 | reduced |
| `gene_sense_total_chase_6h_0h` | chromosome | 0.530 | 0.517 | 0.536 | 0.480 | 0.510 | reduced |
| `exon_sense_late_chase_6h_2h` | chromosome | 0.514 | 0.514 | 0.557 | 0.512 | 0.547 | wide B2 |
| `exon_sense_total_chase_6h_0h` | chromosome | 0.754 | 0.755 | 0.772 | 0.747 | 0.766 | wide B2 |
| `gene_sense_late_chase_6h_2h` | repeated random | 0.455 | 0.461 | 0.512 | 0.462 | 0.491 | wide B2 |
| `gene_sense_total_chase_6h_0h` | repeated random | 0.551 | 0.542 | 0.579 | 0.522 | 0.548 | reduced |
| `exon_sense_late_chase_6h_2h` | repeated random | 0.479 | 0.487 | 0.547 | 0.511 | 0.534 | wide B2 |
| `exon_sense_total_chase_6h_0h` | repeated random | 0.752 | 0.753 | 0.778 | 0.752 | 0.768 | wide B2 |

## Interpretation

The reduced-window GPU baseline is the better current RAMHT configuration for gene-sense labels. The
wide-window batch-2 tuning is stable on RTX 2080 Ti and gives tiny gains for exon labels, but it does
not materially change the paper-level conclusion.

RAMHT beats the best raw deep baseline on some labels, especially gene total and exon total, but it
does not consistently beat the existing raw-plus-engineered hybrid models and does not exceed XGBoost.
The current result therefore supports RAMHT as a viable multi-task neural baseline, not yet as the
strongest model.

Recommended next tuning steps:

1. Try a lower learning rate (`1e-4`) with the reduced-window configuration first, because the wide
   window did not improve gene labels.
2. If learning-rate tuning helps, then test `MODEL_DIM=192` with `BATCH_SIZE=2`.
3. Only run ablations after a RAMHT configuration approaches the existing hybrid or XGBoost baselines.
