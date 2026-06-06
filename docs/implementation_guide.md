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

### 4. 刷新结果表、报告图和 source data

```bash
PYTHONPATH=src python scripts/build_current_results.py
```

它依次运行：

```text
summarize_parallel_deep_gpu_full.py
summarize_parallel_model_suite.py
generate_current_results_figures.py
```

## 当前关键结果文件

```text
data/processed/parallel_label_quality_summary.tsv
data/processed/parallel_label_feature_tables.tsv
data/processed/parallel_label_model_comparison.tsv
data/processed/parallel_deep_gpu_full_summary.tsv
data/processed/parallel_model_suite_summary.tsv
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
- `full`: 当前主要指 ElasticNet 完整 split 评估。

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

- 四标签实验仍由项目级脚本编排，尚未全部迁移到 CLI/Snakemake。
- Full XGBoost/RandomForest 尚未与深度模型共享完全一致的 split manifest。
- 当前 split 由各运行函数确定，应进一步保存为显式 manifest。
- 旧报告仍包含早期单标签和 hybrid 模型结果，阅读时应以
  `docs/current_results.md` 为当前入口。
