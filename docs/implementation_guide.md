# Implementation and Reproducibility Guide

## 代码组织原则

- 可复用、可测试的逻辑放在 `src/rna_stability_elements/`。
- 稳定功能通过 `rse` CLI 暴露。
- 跨模块实验编排、长时间训练和项目级汇总放在 `scripts/`。
- 大型数据和结果表放在 `data/`，不提交到 git。
- 当前结论写入 `docs/current_results.md`；历史报告保留但不作为当前排名。

## 目录结构

```text
configs/                       配置与 ENCODE 清单
data/
  raw/                         原始 ENCODE 文件
  external/                    GENCODE 等外部资源
  interim/                     manifest 和中间表
  processed/                   标签、模型输入、指标、预测和 source data
docs/                          技术文档、当前报告与图
scripts/                       实验编排、汇总与绘图
src/rna_stability_elements/    Python 包
tests/                         单元测试
workflow/                      早期 Snakemake 流程
```

## 核心源码模块

| 模块 | 职责 |
| --- | --- |
| `encode.py` | ENCODE metadata discovery 与文件下载 |
| `quant.py` | gene/genic-feature quantification 解析 |
| `target_quality.py` | 重复配对、标签计算、QC 与严格 consensus |
| `analysis.py` | 共识标签、重复 QC 和分析主表 |
| `annotation.py` | GENCODE 注释、代表性转录本和区域切分 |
| `features.py` | 长度、组成、k-mer 和 motif 特征 |
| `models/evaluation.py` | repeated-random 与 chromosome-holdout split |
| `models/sequence_cnn.py` | Region-CNN |
| `models/sequence_transformer.py` | Conv-tokenized Transformer |
| `models/saluki_like.py` | CNN + BiGRU + attention |
| `interpretation.py` | 历史阶段 leaderboard 与序列语法解释 |
| `cli.py` | `rse` 命令入口 |

## 当前主流程

### 1. 构建四套严格标签

```bash
PYTHONPATH=src python scripts/run_four_way_label_analysis.py
PYTHONPATH=src python scripts/summarize_parallel_label_models.py
```

### 2. 构建四套深度模型序列表

```bash
PYTHONPATH=src python scripts/make_parallel_deep_sequence_tables.py
```

### 3. 运行 GPU-full 原始序列模型

```bash
PYTHONPATH=src python scripts/run_parallel_deep_gpu_full.py \
  --gpus 0,1,2,3 \
  --n-repeats 3
```

运行器特性：

- 每张 GPU 负责一套标签；
- 每套标签依次运行 Region-CNN、Transformer、Saluki-like；
- 每个模型运行 repeated-random 和 chromosome-holdout；
- 指标文件存在时自动跳过，支持中断后恢复；
- 日志写入 `logs/parallel_deep_gpu_full/`。

### 4. 构建并运行固定 split 公平 benchmark

```bash
PYTHONPATH=src python scripts/build_fair_benchmark_manifests.py
PYTHONPATH=src python scripts/run_fair_classical_benchmark.py --gpus 0,1,2,3
PYTHONPATH=src python scripts/benchmark_fair_deep_cost.py --gpus 0,1,2,3
PYTHONPATH=src python scripts/summarize_fair_benchmark.py
```

现有 GPU-full 深度性能结果只有在 `fair_benchmark_deep_reuse_audit.tsv` 逐 split 精确通过后
才复用。深度成本脚本仅在固定 `random_repeat_0` 上重训，用于统一测量计算成本。

### 5. 运行人工特征输入信息消融

```bash
PYTHONPATH=src python scripts/run_input_ablation_benchmark.py --gpus 0,1,2,3
PYTHONPATH=src python scripts/summarize_input_ablation.py
```

该实验使用固定 fair-benchmark manifests，对 XGBoost 运行区域 only、structured
leave-one-region-out，以及 length/composition/motif/k-mer 特征类型消融。

### 6. 运行深度原始序列区域消融与 hybrid

```bash
PYTHONPATH=src python scripts/run_deep_input_ablation_gpu_full.py \
  --gpus 0,1,2,3 \
  --n-repeats 3
PYTHONPATH=src python scripts/summarize_deep_input_ablation.py
```

该实验复用 raw-all 结果，并正式运行单区域 only、leave-one-region-out 和
raw sequence + engineered features hybrid；所有新增结果逐 split 对固定 manifest 审计。

### 7. 运行深度 hybrid 输入设计筛选

```bash
PYTHONPATH=src python scripts/run_deep_input_design_gpu_full.py \
  --stage screen \
  --gpus 0,1,2,3 \
  --n-repeats 3
PYTHONPATH=src python scripts/summarize_deep_input_design.py
PYTHONPATH=src python scripts/run_deep_input_design_gpu_full.py \
  --stage expand \
  --best-config medium_balanced \
  --gpus 0,1,2,3 \
  --n-repeats 3
```

筛选阶段优先使用 Transformer hybrid，在 `gene_sense + 6h/2h` 与 `exon_sense + 6h/2h`
上比较 short/medium/long 窗口、四种裁剪策略，以及固定总长度下 CDS-heavy 和 3'UTR-heavy
配额。若最佳配置为已完成的 `medium_balanced`，扩展阶段会直接复用已审计 hybrid 结果。

### 8. 刷新结果表、报告图和 source data

```bash
PYTHONPATH=src python scripts/build_current_results.py
```

它依次运行：

```text
summarize_parallel_deep_gpu_full.py
summarize_parallel_model_suite.py
generate_current_results_figures.py
summarize_fair_benchmark.py
summarize_input_ablation.py
summarize_deep_input_ablation.py
summarize_deep_input_design.py
```

## 当前关键结果文件

```text
data/processed/parallel_label_quality_summary.tsv
data/processed/parallel_label_feature_tables.tsv
data/processed/parallel_label_model_comparison.tsv
data/processed/parallel_deep_gpu_full_summary.tsv
data/processed/parallel_model_suite_summary.tsv
data/processed/fair_benchmark_cohort_summary.tsv
data/processed/fair_benchmark_summary.tsv
data/processed/fair_benchmark_paired_differences.tsv
data/processed/fair_benchmark_cost_summary.tsv
data/processed/input_ablation_summary.tsv
data/processed/input_ablation_paired_differences.tsv
data/processed/deep_input_ablation_summary.tsv
data/processed/deep_input_ablation_paired_differences.tsv
data/processed/deep_input_design_summary.tsv
data/processed/deep_input_design_paired_differences.tsv
data/processed/deep_input_design_screen_ranking.tsv
data/processed/figure_source_data/
```

每个 GPU-full 模型和标签还会产生：

```text
parallel_deep_gpu_full_<model>_<label_id>_metrics.tsv
parallel_deep_gpu_full_<model>_<label_id>_predictions.tsv
parallel_deep_gpu_full_<model>_<label_id>_history.tsv
```

## 结果解释约定

- `full_deep_gpu`: 默认完整模型规模、early stopping、3 次随机拆分和 23 个染色体留出。
- `quick_deep_cpu`: 早期流程验证，不用于最终模型排名。
- `quick_compact`: 轻量传统模型 benchmark，适合筛选但不是最终公平比较。
- `full`: 完整模型设置，使用全部 3 个 repeated-random 和 23 个 chromosome-holdout splits。

比较模型时必须同时说明：

- 输入表示：人工特征、原始序列、embedding 或 hybrid；
- evaluation 类型与 split 数；
- 是否 quick/full；
- 指标是 split 中位数还是单次结果。

## 图表复现

当前核心图由 Python/matplotlib 生成：

```bash
PYTHONPATH=src python scripts/generate_current_results_figures.py
```

输出：

```text
docs/figures/current_results_overview.{png,svg,pdf}
docs/figures/gpu_full_model_comparison.{png,svg,pdf}
docs/figures/fair_benchmark_overview.{png,svg,pdf}
docs/figures/fair_benchmark_split_distributions.{png,svg,pdf}
docs/figures/fair_benchmark_chromosome_heatmap.{png,svg,pdf}
docs/figures/input_ablation_overview.{png,svg,pdf}
docs/figures/input_ablation_chromosome_holdout.{png,svg,pdf}
docs/figures/deep_input_ablation_chromosome_holdout.{png,svg,pdf}
docs/figures/deep_input_ablation_paired_differences.{png,svg,pdf}
docs/figures/deep_input_design_screen_ranking.{png,svg,pdf}
docs/figures/deep_input_design_screen_paired_differences.{png,svg,pdf}
data/processed/figure_source_data/*.tsv
```

SVG/PDF 保留可编辑文本；PNG 用于 README 和快速预览。

## 测试与检查

```bash
PYTHONPATH=src pytest tests
python -m py_compile scripts/*.py
```

新增长期功能时应同时：

1. 在 `src/` 中实现；
2. 添加单元测试；
3. 暴露 CLI 或明确的脚本入口；
4. 更新相应技术文档。

## 已知技术债务

- 四标签与公平 benchmark 实验仍由项目级脚本编排，尚未全部迁移到 CLI/Snakemake。
- 当前计算成本基于每套标签的固定 `random_repeat_0`，不是全部 26 splits 的总成本。
- 旧报告仍包含早期单标签和 hybrid 模型结果，阅读时应以
  `docs/current_results.md` 为当前入口。
