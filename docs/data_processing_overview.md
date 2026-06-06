# Data Processing Overview

## 目标

当前数据流程同时构建四套 RNA stability proxy：

```text
gene_sense × log2(6h / 2h)
gene_sense × log2(6h / 0h)
exon_sense × log2(6h / 2h)
exon_sense × log2(6h / 0h)
```

这样可以区分定量口径与标签时间窗口带来的影响，而不是预先假定某一标签最优。

## 数据来源

```text
16 cell lines × 3 time points × 2 biological replicates
```

- `gene_sense`: ENCODE `gene quantifications`，96 个 TSV。
- `exon_sense`: ENCODE `genic features quantifications`，96 个 TSV。
- 序列与注释：GENCODE v29 GTF 和 transcript FASTA。

Manifest：

```text
data/interim/encode_gene_quant_files.tsv
data/interim/encode_genic_feature_files.tsv
```

## 严格标签构建

核心实现：

```text
src/rna_stability_elements/target_quality.py
scripts/run_four_way_label_analysis.py
```

流程：

```text
raw quantification TSV
  -> sample-level gene signal
  -> pair replicate 1/2 across 0h, 2h, 6h
  -> calculate replicate-level ratios
  -> denominator and replicate-span QC
  -> aggregate passing replicate targets
  -> pass-only cross-cell consensus
  -> merge sequence features
```

标签定义：

```text
log2_stability_6h_2h = log2((signal_6h + 0.1) / (signal_2h + 0.1))
log2_stability_6h_0h = log2((signal_6h + 0.1) / (signal_0h + 0.1))
```

QC 规则：

- 所有标签要求 `signal_0h >= 0.5`。
- `6h/2h` 额外要求 `signal_2h >= 0.5`。
- 重复间标签跨度超过 1.0 log2 unit 时标记为 `high_replicate_target_span`。
- 仅使用 `quality_flag == pass` 的 gene-cell 标签。
- consensus 至少要求 8 个细胞系。

主要输出：

```text
replicate_paired_targets_{gene_sense,exon_sense}.tsv
robust_stability_targets_{gene_sense,exon_sense}.tsv
robust_consensus_<label_id>.tsv
parallel_label_quality_summary.tsv
parallel_label_cross_cell_consistency.tsv
parallel_label_signal_correlations.tsv
```

## 样本 PCA

PCA 使用每种定量口径中方差最高的 5,000 个基因，对 `log2(signal + 0.1)` 标准化后分析。
它用于发现时间点、细胞系或重复异常，不参与标签预测。

输出：

```text
data/processed/sample_signal_pca_gene_sense.tsv
data/processed/sample_signal_pca_exon_sense.tsv
docs/figures/sample_signal_pca_gene_sense.png
docs/figures/sample_signal_pca_exon_sense.png
```

## 序列与特征

GENCODE v29 用于选择代表性转录本并切分：

```text
5'UTR / CDS / 3'UTR / full transcript
```

人工特征包括：

- 区域长度；
- GC/AU 与碱基组成；
- 区域特异 3-mer、4-mer；
- 已知调控 motif count。

四套人工特征表：

```text
parallel_sequence_model_features_<label_id>.tsv
```

四套原始序列深度学习表：

```text
parallel_modeling_master_with_sequences_<label_id>.tsv
```

## 当前覆盖

| 标签 | gene-cell 可用记录 | QC 通过记录 | 严格 consensus | 模型可用基因 |
| --- | ---: | ---: | ---: | ---: |
| `gene_sense + 6h/2h` | 167,205 | 139,831 | 8,428 | 8,428 |
| `gene_sense + 6h/0h` | 191,148 | 160,368 | 9,848 | 9,848 |
| `exon_sense + 6h/2h` | 189,537 | 160,859 | 9,765 | 9,018 |
| `exon_sense + 6h/0h` | 206,270 | 172,539 | 10,678 | 9,881 |

exon-sense consensus 中部分基因不在当前代表性转录本/基础特征集合中，因此模型可用基因数稍低。

## 复现

```bash
PYTHONPATH=src python scripts/run_four_way_label_analysis.py
PYTHONPATH=src python scripts/summarize_parallel_label_models.py
PYTHONPATH=src python scripts/make_parallel_deep_sequence_tables.py
```

数据产物较大，位于 `data/` 且默认不提交到 git。
