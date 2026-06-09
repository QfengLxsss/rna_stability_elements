# RNA Stability Elements

本项目使用 ENCODE BrU-seq / BruChase-seq pulse-chase 数据，研究 RNA 序列中是否存在可跨
细胞系泛化的稳定性调控语法。当前阶段重点不是选择单一标签，而是系统比较：

- `gene_sense` 与 `exon_sense` 两种定量口径；
- `log2(6h / 2h)` 与 `log2(6h / 0h)` 两种稳定性代理标签；
- 人工序列特征模型与端到端原始序列深度模型。

## 当前结论

严格质控后形成四套平行标签，并已在完全一致的 cohort 与固定 split 上完成六种模型的公平
比较：

| 标签 | 模型可用基因数 | Repeated-random 最佳模型 / Pearson | Chromosome-holdout 最佳模型 / Pearson |
| --- | ---: | --- | --- |
| `gene_sense + 6h/2h` | 8,428 | XGBoost / 0.507 | XGBoost / 0.504 |
| `gene_sense + 6h/0h` | 9,848 | XGBoost / 0.575 | XGBoost / 0.544 |
| `exon_sense + 6h/2h` | 9,018 | XGBoost / 0.544 | XGBoost / 0.546 |
| `exon_sense + 6h/0h` | 9,881 | RandomForest / **0.774** | XGBoost / **0.778** |

目前最稳健的观察是：

1. 标签定义对模型性能的影响大于深度模型架构差异。
2. `exon_sense + 6h/0h` 最容易从序列预测，但可能同时包含成熟 RNA 保留、processing
   和 abundance-linked 信号，不能直接等同于纯降解速率。
3. `6h/2h` 标签更保守、极端值更少；`6h/0h` 重复一致性和序列可预测性更强。因此两者
   应继续平行分析。
4. XGBoost 在全部 chromosome-holdout 和 3/4 repeated-random 任务中领先；唯一例外是
   `exon_sense + 6h/0h` 的 repeated-random 中位数由 RandomForest 以 0.003 的差距领先。
5. 公平 benchmark 比较的是完整 pipeline；人工特征模型与原始序列模型输入不同，排名不能
   单独归因于模型架构。
6. XGBoost 输入消融显示 CDS 是最关键区域；仅使用 k-mer 已保留大部分完整性能，而当前
   motif panel 单独预测能力有限。
7. GPU-full 深度输入消融进一步确认 CDS 是主要区域；去除 CDS 的跨模型平均
   chromosome-holdout Pearson 损失为 0.059，而 CDS-only 平均仅损失 0.017。
8. raw sequence + engineered features hybrid 在 12/12 个深度模型-标签组合中提升性能，
   跨组合平均 chromosome-holdout Pearson 增益为 0.037。
9. Transformer hybrid 输入设计筛选显示 `medium_balanced`（5'UTR/CDS/3'UTR =
   256/1024/1024）仍是最稳健默认设置；更长窗口没有稳定增益，固定总长度下增加 3'UTR
   配额反而降低性能。
10. 生物学解释性汇总显示，单变量 association 中 GC-rich / C/G-rich k-mer 信号很强，
    但预测性消融更支持 CDS 是主要可用信息来源；当前 motif panel 只能作为候选机制线索。
11. 机制解释补充实验显示 codon-only 特征本身具有较强预测力，但加入完整 engineered
    features 后增益较小；group permutation 仍以 k-mer3/k-mer4 为主，同义 CDS recoding
    可显著改变预测值，支持 codon-usage-sensitive CDS 信号。

![Fair benchmark overview](docs/figures/fair_benchmark_overview.png)

## 数据与评估设计

数据来自 ENCODE Ljungman lab 的 16 个细胞系、3 个时间点和 2 个生物学重复：

```text
16 cell lines × 3 time points × 2 replicates = 96 quantification files
```

稳定性代理标签：

```text
late chase:  log2_stability_6h_2h = log2((signal_6h + 0.1) / (signal_2h + 0.1))
total chase: log2_stability_6h_0h = log2((signal_6h + 0.1) / (signal_0h + 0.1))
```

当前严格标签流程先配对生物学重复，再计算比值，并执行：

- 所有标签要求 `signal_0h >= 0.5`；
- `6h/2h` 额外要求 `signal_2h >= 0.5`；
- 重复间标签跨度大于 1.0 log2 unit 时标记为不通过；
- 仅使用通过 QC 且至少覆盖 8 个细胞系的基因构建 consensus。

模型使用两种评估：

- `repeated_random`: 3 次随机基因拆分；
- `chromosome_holdout`: 23 个染色体分别作为测试集。

GPU-full 深度实验共完成 4 标签 × 3 模型 × 26 splits = **312 次训练**。深度结果经逐 split
测试基因审计后全部可复用；Full ElasticNet、RandomForest 与 XGBoost 另完成 **312 次拟合**。
深度输入消融另完成 7 个新条件 × 4 标签 × 3 模型 × 26 splits = **2,184 次 CUDA 训练**。
Transformer hybrid 输入设计筛选另完成 13 个新配置 × 2 标签 × 26 splits =
**676 次 CUDA 训练**，并复用已审计的 `medium_balanced` hybrid 结果扩展到四标签三模型。

## 项目入口

- [当前成果、限制与下一步](docs/current_results.md)
- [公平模型比较报告](docs/fair_benchmark_report.md)
- [输入信息消融报告](docs/input_ablation_report.md)
- [深度原始序列区域消融与 hybrid 报告](docs/deep_input_ablation_report.md)
- [深度 hybrid 输入设计报告](docs/deep_input_design_report.md)
- [生物学解释报告](docs/biological_interpretation_report.md)
- [机制解释补充报告](docs/mechanistic_interpretation_report.md)
- [文档索引](docs/README.md)
- [数据处理流程](docs/data_processing_overview.md)
- [实现与复现指南](docs/implementation_guide.md)
- [模型实现原理](docs/models_submodule.md)
- [四标签分析报告](docs/parallel_label_analysis_report.md)
- [模型套件报告](docs/parallel_model_suite_report.md)
- [实验脚本说明](scripts/README.md)

## 仓库结构

```text
configs/                       项目配置与 ENCODE 清单
data/                          raw / external / interim / processed 数据
docs/                          当前报告、技术文档和 figures
scripts/                       项目级实验编排、汇总与绘图
src/rna_stability_elements/    可复用源码与 CLI
tests/                         单元测试
workflow/                      早期阶段 Snakemake 工作流
```

核心源码边界：

- `target_quality.py`: 重复配对、信号过滤、严格 consensus；
- `quant.py`: ENCODE 定量表解析；
- `annotation.py`: GENCODE 转录本与区域序列；
- `features.py`: 长度、组成、k-mer、motif 特征；
- `models/`: 传统模型、Region-CNN、Transformer、Saluki-like；
- `cli.py`: `rse` 命令行入口。

## 快速开始

推荐环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,analysis,boosting,deep,workflow]"
```

运行测试：

```bash
PYTHONPATH=src pytest tests
```

刷新当前汇总表与结果图：

```bash
PYTHONPATH=src python scripts/build_current_results.py
```

重新构建四套严格标签：

```bash
PYTHONPATH=src python scripts/run_four_way_label_analysis.py
PYTHONPATH=src python scripts/summarize_parallel_label_models.py
PYTHONPATH=src python scripts/make_parallel_deep_sequence_tables.py
```

四张 GPU 并行运行 full raw-sequence 深度模型：

```bash
PYTHONPATH=src python scripts/run_parallel_deep_gpu_full.py \
  --gpus 0,1,2,3 \
  --n-repeats 3
```

该运行器按标签分配 GPU，依次运行 Region-CNN、Transformer 和 Saluki-like；已完成的指标文件
会被自动跳过。

## 下一步优先级

1. 使用 SHAP、attribution 和 in-silico mutagenesis 识别稳健候选调控元件。
2. 将四标签共享信号与标签特异信号分开，避免把 processing 或 abundance 信号误解释为 decay。
3. 以 `medium_balanced` hybrid 作为默认深度输入，探索 cell-line context-aware 模型。

## 数据来源与方法参考

- ENCODE Portal: Ljungman lab BrU-seq / BruChase-seq pulse-chase series。
- Agarwal and Kelley, *The genetic and biochemical determinants of mRNA degradation rates in
  mammals*, Genome Biology, 2022。
- Dalla-Torre et al., *Nucleotide Transformer*, Nature Methods, 2024。
