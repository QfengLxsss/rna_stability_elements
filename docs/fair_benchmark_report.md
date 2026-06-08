# Fair Model Benchmark

All six models use the same four label-specific cohorts and the same fixed train, validation, and test gene assignments.

## Design

- Four shared cohorts: engineered-feature and raw-sequence tables have identical gene order.
- Fixed splits per label: 3 repeated-random and 23 chromosome-holdout splits.
- Existing GPU-full deep results were reused only after exact per-split test-gene audit.
- Classical models train only on the manifest `train` role; validation and test genes are excluded.

![Fair benchmark overview](figures/fair_benchmark_overview.png)

## Best Models by Label

| Label | Repeated-random best | Pearson | Chromosome-holdout best | Pearson |
| --- | --- | ---: | --- | ---: |
| `gene_sense_late_chase_6h_2h` | XGBoost | 0.507 | XGBoost | 0.504 |
| `gene_sense_total_chase_6h_0h` | XGBoost | 0.575 | XGBoost | 0.544 |
| `exon_sense_late_chase_6h_2h` | XGBoost | 0.544 | XGBoost | 0.546 |
| `exon_sense_total_chase_6h_0h` | RandomForest | 0.774 | XGBoost | 0.778 |

## Main Conclusions

1. Full XGBoost leads all four chromosome-holdout tasks and three of four repeated-random tasks; RandomForest narrowly leads the repeated-random median for exon_sense 6h/0h.
2. The label definition remains a larger source of performance variation than model architecture.
3. Deep raw-sequence models are competitive, but do not exceed engineered-feature XGBoost in the current configuration.
4. Chromosome-holdout distributions show that model rankings are generally stable, while absolute difficulty varies by chromosome.
5. The controlled cost benchmark shows that XGBoost is also substantially faster than the current full deep models.

## Cost Benchmark

Costs were measured on the fixed `random_repeat_0` split. Deep-model cost runs reproduce the same test genes as the manifest.

| Model | Median train wall time (s) | Median peak GPU memory (MB) |
| --- | ---: | ---: |
| ElasticNet | 13.1 | NA |
| RandomForest | 24.4 | NA |
| XGBoost | 5.6 | NA |
| Region-CNN | 92.0 | 1847 |
| Transformer | 121.3 | 1470 |
| Saluki-like | 270.0 | 1570 |

## Statistical Outputs

- `data/processed/fair_benchmark_summary.tsv`: mean, median, standard deviation, IQR, and bootstrap mean CI.
- `data/processed/fair_benchmark_paired_differences.tsv`: paired Pearson differences and win fractions versus XGBoost.
- `data/processed/fair_benchmark_cost_summary.tsv`: controlled computation-cost measurements.
- `data/processed/fair_benchmark_deep_reuse_audit.tsv`: exact split-level deep-result reuse audit.

## Figures

- `docs/figures/fair_benchmark_overview.{png,svg,pdf}`
- `docs/figures/fair_benchmark_split_distributions.{png,svg,pdf}`
- `docs/figures/fair_benchmark_chromosome_heatmap.{png,svg,pdf}`
