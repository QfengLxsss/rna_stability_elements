# RAMHT GPU Session Handoff

This handoff summarizes the model-design and training work completed in the current session, and the
recommended next steps for a new session with direct GPU access.

## Project Context

The project studies whether RNA sequence contains generalizable stability-regulatory grammar using
ENCODE BrU-seq / BruChase-seq pulse-chase data. The current mature benchmark already includes:

- four strict labels: `gene_sense/exon_sense` x `6h/2h` and `6h/0h`;
- fixed fair-benchmark cohorts and split manifests;
- repeated-random and chromosome-holdout evaluation;
- engineered-feature models, raw sequence deep models, hybrid deep models, input ablations, and
  biological interpretation reports.

The earlier benchmark story is strong, but the user wants to pivot toward a strongest-neural-model
paper story. The proposed model is:

> **RAMHT: Region-aware Multi-task Hybrid Transformer**

## RAMHT Design

RAMHT combines four streams:

1. 5'UTR nucleotide Transformer encoder.
2. CDS nucleotide Transformer encoder.
3. 3'UTR nucleotide Transformer encoder.
4. CDS codon-aware Transformer branch.
5. Engineered sequence-grammar MLP over the existing compact features.

The encoded streams are fused through a learned gate:

```text
h_fused = gate_seq * h_sequence + gate_codon * h_codon + gate_engineered * h_engineered
```

Then four task-specific regression heads predict:

- `gene_sense_late_chase_6h_2h`
- `gene_sense_total_chase_6h_0h`
- `exon_sense_late_chase_6h_2h`
- `exon_sense_total_chase_6h_0h`

The multi-task split is conservative: for a given `split_name`, any gene that is test for any label is
excluded from all training and validation tasks. This prevents shared-encoder leakage across labels.

## Files Added Or Changed

Core model:

- `src/rna_stability_elements/models/ramht.py`

Training and launch scripts:

- `scripts/run_ramht_multitask.py`
- `scripts/launch_ramht_gpu.sh`
- `scripts/launch_ramht_gpu_splits.sh`
- `scripts/summarize_ramht_runs.py`

Documentation:

- `docs/ramht_model_plan.md`
- `docs/ramht_gpu_handoff.md`
- `docs/README.md`
- `scripts/README.md`

Tests:

- `tests/test_evaluation.py` now includes RAMHT codon encoding and small forward-pass tests.

Current test status:

```text
pytest -q
33 passed, 1 warning
```

## Completed Runs

### CPU Smoke

Purpose: verify table construction, codon encoding, target masking, gated fusion, and output writing.

Outputs:

- `data/processed/ramht_smoke_metrics.tsv`
- `data/processed/ramht_smoke_predictions.tsv`
- `data/processed/ramht_smoke_history.tsv`

### CPU Pilot

Tiny windows, 3 epochs, `random_repeat_0`.

Outputs:

- `data/processed/ramht_cpu_pilot_metrics.tsv`
- `data/processed/ramht_cpu_pilot_predictions.tsv`
- `data/processed/ramht_cpu_pilot_history.tsv`

Metrics:

| Label | Pearson | Spearman | R2 |
| --- | ---: | ---: | ---: |
| `gene_sense_late_chase_6h_2h` | 0.428 | 0.439 | 0.170 |
| `gene_sense_total_chase_6h_0h` | 0.548 | 0.566 | 0.294 |
| `exon_sense_late_chase_6h_2h` | 0.468 | 0.517 | 0.204 |
| `exon_sense_total_chase_6h_0h` | 0.764 | 0.684 | 0.579 |

### Failed Full RAMHT-Small Attempt

Initial command used full window + large batch:

```text
batch_size=32, 5'UTR/CDS/3'UTR=256/1024/1024, model_dim=192, transformer_layers=3
```

It failed on RTX 2080 Ti 11GB:

```text
RuntimeError: CUDA out of memory
```

Log:

- `logs/ramht/ramht_small_random_repeat_0.log`

### Successful RAMHT-2080Ti Single Split

This run succeeded on GPU with a memory-safe configuration:

```text
split: random_repeat_0
batch_size: 4
5'UTR/CDS/3'UTR: 128/768/768
codon_length: 256
model_dim: 128
codon_dim: 128
transformer_layers: 2
codon_layers: 1
feedforward_dim: 256
feature_hidden_dim: 256
head_hidden_dim: 128
```

Outputs:

- `data/processed/ramht_2080ti_random_repeat_0_metrics.tsv`
- `data/processed/ramht_2080ti_random_repeat_0_predictions.tsv`
- `data/processed/ramht_2080ti_random_repeat_0_history.tsv`
- `logs/ramht/ramht_2080ti_random_repeat_0.log`

Metrics:

| Label | Pearson | Spearman | R2 |
| --- | ---: | ---: | ---: |
| `gene_sense_late_chase_6h_2h` | 0.412 | 0.431 | 0.170 |
| `gene_sense_total_chase_6h_0h` | 0.547 | 0.573 | 0.285 |
| `exon_sense_late_chase_6h_2h` | 0.453 | 0.504 | 0.195 |
| `exon_sense_total_chase_6h_0h` | 0.764 | 0.687 | 0.584 |

Interpretation: this is a successful GPU baseline, but it is a memory-safe reduced-window model. Do
not yet claim strongest performance. Use it as the first GPU sanity point before all-split training and
ablation.

## GPU Access Note

The current Codex session cannot access `/dev/nvidia*`, even though the server has 8 RTX 2080 Ti GPUs.
The user terminal can access CUDA. GPU training therefore needs to be launched from the user's GPU-visible
terminal or from a new Codex session that has GPU device passthrough.

## Recommended 8-GPU Strategy

Do not start with DDP. For this project, the best use of 8 GPUs is split-level parallelism:

- one independent split per GPU;
- no cross-GPU synchronization;
- robust resume/skip behavior;
- directly matches the fixed benchmark design.

Use `scripts/launch_ramht_gpu_splits.sh`.

## Next Command: All Splits On 8 GPUs

Run from a GPU-visible terminal:

```bash
cd /data15/data15_5/junguang/wangshuo/rna_stability

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

Expected merged outputs:

- `data/processed/ramht_2080ti_parallel_metrics.tsv`
- `data/processed/ramht_2080ti_parallel_predictions.tsv`
- `data/processed/ramht_2080ti_parallel_history.tsv`
- `data/processed/ramht_2080ti_parallel_summary.tsv`

Per-split outputs:

- `data/processed/ramht_runs/ramht_2080ti_parallel/`
- `logs/ramht/ramht_2080ti_parallel/`

## Monitoring

```bash
tail -f logs/ramht/ramht_2080ti_parallel/*.log
```

GPU usage:

```bash
watch -n 5 gpustat
```

If a few splits fail, inspect:

```bash
rg -n "RuntimeError|CUDA out of memory|Traceback|error" logs/ramht/ramht_2080ti_parallel
```

The scheduler skips completed split metrics, so it can be re-run after failures.

## After All-Split Training

1. Compare `ramht_2080ti_parallel_summary.tsv` to:
   - `data/processed/fair_benchmark_summary.tsv`
   - `data/processed/fair_benchmark_all_metrics.tsv`
   - existing deep hybrid results in `data/processed/deep_input_ablation_summary.tsv`
2. Check whether RAMHT beats previous deep models on all four labels.
3. Check whether RAMHT approaches or exceeds XGBoost on any label.
4. If performance is good, run ablations:
   - `no_codon`
   - `no_engineered`
   - `concat_fusion`
   - `single_task`
   - `raw_only_multitask`
5. If performance is weak, tune:
   - increase windows to `256/1024/1024` with `BATCH_SIZE=2`;
   - increase `MODEL_DIM` back to 192 only after memory is stable;
   - try lower learning rate `1e-4`;
   - add mixed precision / gradient accumulation later if needed.

## Current Best Paper Framing

Do not yet write "RAMHT is the strongest model" until all fixed splits are complete. The safe current
framing is:

> We designed a region-aware multi-task hybrid Transformer that jointly models four RNA stability
> proxy labels through nucleotide, codon, and engineered grammar streams. A reduced-window GPU baseline
> runs successfully on RTX 2080 Ti and is ready for full fixed-split evaluation.

The strongest-model claim depends on the 26-split result and ablations.
