# Current Results and Next Steps

## 项目当前完成度

项目已完成从数据下载、严格标签构建、序列映射、人工特征提取，到四标签传统模型和
GPU-full 原始序列模型评估的完整闭环。

### 已完成的数据工作

- 下载 96 个 ENCODE gene quantification 文件和 96 个 genic-feature quantification 文件。
- 下载并解析 GENCODE v29 GTF 与转录本序列。
- 建立重复配对、低分母过滤、重复跨度过滤和 pass-only consensus。
- 对 `gene_sense`、`exon_sense` 信号矩阵完成样本 PCA。
- 构建四套严格标签及对应人工特征表、原始序列表。

### 已完成的模型工作

- 人工特征模型：Ridge、ElasticNet、RandomForest、XGBoost-light。
- 原始序列模型：Region-CNN、Conv-tokenized Transformer、Saluki-like CNN-GRU。
- GPU-full 原始序列评估：每个模型和标签包含 3 次 repeated-random 与 23 次
  chromosome-holdout，共 312 次 CUDA 训练。

## 核心结果

![Current results overview](figures/current_results_overview.png)

### 标签质量与覆盖

| 标签 | 严格 consensus 基因 | 模型可用基因 | QC pass fraction |
| --- | ---: | ---: | ---: |
| `gene_sense + 6h/2h` | 8,428 | 8,428 | 0.836 |
| `gene_sense + 6h/0h` | 9,848 | 9,848 | 0.839 |
| `exon_sense + 6h/2h` | 9,765 | 9,018 | 0.849 |
| `exon_sense + 6h/0h` | 10,678 | 9,881 | 0.836 |

### GPU-full 原始序列模型

| 标签 | 最佳模型 | Repeated-random Pearson | Chromosome-holdout Pearson |
| --- | --- | ---: | ---: |
| `gene_sense + 6h/2h` | Transformer | 0.469 | 0.427 |
| `gene_sense + 6h/0h` | Transformer | 0.522 | 0.472 |
| `exon_sense + 6h/2h` | Saluki-like | 0.511 | 0.485 |
| `exon_sense + 6h/0h` | Saluki-like | 0.748 | 0.763 |

### 可支持的结论

1. RNA 稳定性代理标签中存在可泛化的序列信号。
2. 标签定义对性能影响非常明显，超过当前三个深度架构之间的差异。
3. `exon_sense + 6h/0h` 在随机拆分和染色体留出中均表现最好。
4. `6h/0h` 更容易预测，但更可能混入 processing、成熟 RNA retention 和 abundance-linked
   信息；`6h/2h` 更适合保守验证。
5. 原始序列深度模型与人工特征模型都有效，但尚未进行完全公平的同 split、同预算比较。

## 当前证据边界

- `log2(6h/0h)` 和 `log2(6h/2h)` 是稳定性代理，不是直接测得的 half-life。
- `gene_sense` 与 `exon_sense` 可能对应不同生物过程，不能只根据预测性能选一个并丢弃另一个。
- XGBoost-light 使用人工特征，深度模型使用原始序列；两者比较反映 pipeline，而非纯架构。
- CPU quick 深度模型只用于早期流程验证，不应与 GPU-full 结果混为最终排名。
- 当前深度模型使用固定默认序列窗口，尚未证明更长窗口一定更好。

## 下一步工作

### P0：完成公平模型比较

- 为四套标签保存并复用完全一致的 split manifest。
- 使用相同 repeated-random 与 chromosome-holdout 运行 Full XGBoost、RandomForest、
  ElasticNet 和三个 GPU-full 深度模型。
- 报告 split-level 分布、置信区间、训练耗时和参数规模。

### P1：输入与区域消融

- 比较人工特征 only、原始序列 only、人工特征 + 原始序列 hybrid。
- 分别测试 5'UTR、CDS、3'UTR、UTR-only 和全转录本。
- 系统比较短/中/长窗口和 `balanced/start/end/random` 裁剪。
- 用 GC、长度和 k-mer-only 基线判断性能是否主要来自简单统计量。

### P2：生物学解释

- 对 XGBoost 做 SHAP、特征组重要性和 motif family 聚类。
- 对 Transformer/Saluki-like 做 attribution 与 in-silico mutagenesis。
- 优先保留跨 `gene_sense/exon_sense`、跨 `6h/2h` 与 `6h/0h` 可重复的候选元件。
- 对候选基因和 motif 做 RBP motif、GO 与 Reactome 富集。

### P3：扩展到 context-aware 模型

- 加入 RBP、miRNA expression 和 eCLIP binding。
- 从 consensus sequence-only 预测扩展到 gene × cell-line stability 预测。

## 关键产物

- `data/processed/parallel_label_quality_summary.tsv`
- `data/processed/parallel_deep_gpu_full_summary.tsv`
- `data/processed/parallel_model_suite_summary.tsv`
- `data/processed/figure_source_data/`
- `docs/figures/current_results_overview.{png,svg,pdf}`
- `docs/figures/gpu_full_model_comparison.{png,svg,pdf}`
