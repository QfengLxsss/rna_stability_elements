# RAMHT: Region-aware Multi-task Hybrid Transformer

This document records the current proposed deep-model path for turning the project from a
benchmark story into a strongest-neural-model story.

## Architecture

`RAMHT-small` combines four evidence streams:

1. region-specific nucleotide Transformer encoders for 5'UTR, CDS, and 3'UTR;
2. a CDS codon-aware Transformer branch;
3. an engineered grammar-feature MLP over the existing 1,336 compact sequence features;
4. gated fusion followed by four task-specific regression heads.

The four output heads are:

- `gene_sense_late_chase_6h_2h`
- `gene_sense_total_chase_6h_0h`
- `exon_sense_late_chase_6h_2h`
- `exon_sense_total_chase_6h_0h`

The current default window is the previously selected `medium_balanced` design:

```text
5'UTR: 256 nt
CDS:   1024 nt
3'UTR: 1024 nt
CDS codon branch: 342 codons
```

## Leakage Control

For each fixed fair-benchmark `split_name`, RAMHT builds one multi-task split. To avoid shared-encoder
leakage, genes that are test genes for any label in that split are excluded from all training and
validation tasks. Genes that are validation genes for any label are held out of training as validation
unless they are already in the union test set.

This is conservative and may reduce the available training set, but it keeps the multi-task comparison
clean.

## Current Implementation

- Model module: `src/rna_stability_elements/models/ramht.py`
- Training script: `scripts/run_ramht_multitask.py`
- Smoke outputs: `data/processed/ramht_smoke_*.tsv`

The CPU smoke run uses tiny windows and one epoch only. It verifies table building, target masking,
codon encoding, gated fusion, metric writing, and four-task prediction.

## CPU Pilot Status

The current workspace has `torch 1.12.1` with `cuda_available=False`, so the formal GPU baseline was
not launched in this environment. A small CPU pilot was completed with tiny windows
(`16/32/32 nt`, 10 codons) and 3 epochs on `random_repeat_0`:

| Label | Pearson | Spearman | R2 |
| --- | ---: | ---: | ---: |
| `gene_sense_late_chase_6h_2h` | 0.428 | 0.439 | 0.170 |
| `gene_sense_total_chase_6h_0h` | 0.548 | 0.566 | 0.294 |
| `exon_sense_late_chase_6h_2h` | 0.468 | 0.517 | 0.204 |
| `exon_sense_total_chase_6h_0h` | 0.764 | 0.684 | 0.579 |

The pilot outputs are:

- `data/processed/ramht_cpu_pilot_metrics.tsv`
- `data/processed/ramht_cpu_pilot_predictions.tsv`
- `data/processed/ramht_cpu_pilot_history.tsv`

These numbers are not paper results because the windows are intentionally tiny; they only confirm that
the training path is healthy before moving to GPU.

## Smoke Command

```bash
PYTHONPATH=src python scripts/run_ramht_multitask.py \
  --device cpu \
  --split-name random_repeat_0 \
  --out-prefix data/processed/ramht_smoke \
  --max-epochs 1 \
  --patience 1 \
  --batch-size 16 \
  --max-length-5utr 16 \
  --max-length-cds 32 \
  --max-length-3utr 32 \
  --codon-length 10 \
  --model-dim 32 \
  --codon-dim 32 \
  --transformer-layers 1 \
  --codon-layers 1 \
  --attention-heads 4 \
  --feedforward-dim 64 \
  --feature-hidden-dim 64 \
  --head-hidden-dim 32 \
  --dropout 0.05 \
  --token-dropout 0.0
```

## First GPU Baseline

Run one real split first. For RTX 2080 Ti 11GB cards, start with the memory-safe setting:

```bash
GPU_ID=0 \
SPLIT_SET=random0 \
OUT_PREFIX=data/processed/ramht_2080ti_random_repeat_0 \
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
scripts/launch_ramht_gpu.sh
```

Then compare `data/processed/ramht_2080ti_random_repeat_0_metrics.tsv` against the matching
`random_repeat_0` rows in `fair_benchmark_summary.tsv` / `fair_benchmark_all_metrics.tsv`.

If the 2080Ti setting fits comfortably, increase the windows to `256/1024/1024` and keep
`BATCH_SIZE=4`.

## Full Fixed-Split Run

For this project, the preferred 8-GPU strategy is split-level parallelism rather than DDP. Each GPU
runs an independent split, which avoids cross-GPU synchronization overhead and maximizes throughput for
the fixed benchmark.

After the first split is stable, run all fixed splits across 8 GPUs:

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

Merged outputs:

- `data/processed/ramht_2080ti_parallel_metrics.tsv`
- `data/processed/ramht_2080ti_parallel_predictions.tsv`
- `data/processed/ramht_2080ti_parallel_history.tsv`
- `data/processed/ramht_2080ti_parallel_summary.tsv`

## Next Ablations

The next model variants should be added only after `RAMHT-small` has a real GPU baseline:

1. `no_codon`: remove the CDS codon branch.
2. `concat_fusion`: replace gated fusion with plain concatenation.
3. `single_task`: train one head at a time.
4. `no_engineered`: remove the engineered-feature branch.
5. `raw_only_multitask`: keep multi-task learning but remove engineered features and codon features.

These ablations are necessary before claiming architecture-level novelty.
