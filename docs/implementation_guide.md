# Implementation Guide

本文档面向新合作者和未来的自己：说明这个仓库的实现方式、目录职责、代码边界、数据流和扩展方式。项目思路、技术原理和关键结果已经集中到 `README.md`；这里保留实现层面的细节。

## 顶层结构

```text
rna_stability_elements/
├── README.md
├── pyproject.toml
├── configs/
├── data/
├── docs/
├── notebooks/
├── scripts/
├── src/rna_stability_elements/
├── tests/
└── workflow/
```

### `README.md`

项目首页：

- 这个项目研究什么问题。
- 当前数据和模型进展到哪里。
- 如何从命令行复现核心流程。


### `pyproject.toml`

Python 包和命令行入口配置。当前命令行入口是：

```text
rse = rna_stability_elements.cli:main
```

推荐用可编辑安装运行项目：

```bash
pip install -e ".[dev,analysis,boosting,deep,rna-lm,workflow]"
```

如果只想阅读文档、运行轻量测试和基础数据处理，可以使用较小环境：

```bash
pip install -e ".[dev,analysis]"
```

### `configs/`

项目配置和小型手工清单。

```text
configs/project.yaml
configs/encode_bru_series.tsv
```

使用原则：

- 细胞系白名单、ENCODE 查询条件、默认 target 参数、motif 列表放在 `project.yaml`。
- 新数据源优先新增 config 字段，再让 CLI 读取 config。

### `data/`

数据目录。大文件不应提交到 git，当前 `.gitignore` 已经忽略 `raw/`、`interim/`、`processed/` 和 `external/` 下的大部分文件，只保留 `.gitkeep` 和 `data/README.md`。

```text
data/
├── raw/
├── external/
├── interim/
└── processed/
```

职责：

- `data/raw/`: 原始下载文件，例如 ENCODE quantification TSV。
- `data/external/`: 外部参考文件，例如 GENCODE GTF/FASTA、motif database。
- `data/interim/`: manifest、中间表、标准化后的半成品。
- `data/processed/`: 可直接用于分析、建模、可视化的结果表。

移植项目时，代码仓库可以单独复制；大数据用下载命令或外部存储恢复。

### `docs/`

实现指南、结果报告和可视化图片。

```text
docs/progress_visual_report.md
docs/rna_stability_grammar_interpretation_report.md
docs/figures/
docs/implementation_guide.md
```

职责：

- `implementation_guide.md`: 仓库结构、模块边界、数据流和扩展说明。
- `progress_visual_report.md`: 当前进度的图文报告，由 `rse write-visual-report` 生成。
- `rna_stability_grammar_interpretation_report.md`: 统一模型 leaderboard 和第一版 RNA stability sequence grammar 解释报告，由 `rse write-grammar-report` 生成。
- `figures/`: 可汇报图片。

### `notebooks/`

探索性分析位置。建议只放轻量 notebook，不把核心逻辑写死在 notebook 里。成熟逻辑应迁移到 `src/rna_stability_elements/`，再由 CLI 调用。

### `scripts/`

一次性或兼容旧流程的小脚本。若某段逻辑会被长期复用，应进入 `src/rna_stability_elements/` 并配测试。

### `workflow/`

工作流骨架，目前是 Snakemake：

```text
workflow/Snakefile
```

适合把 CLI 命令串成可复现 pipeline。当前 Snakefile 覆盖阶段一核心路径：gene_sense target、replicate QC、GENCODE sequence、compact sequence features、严格 XGBoost / ElasticNet 评估、progress report 和 grammar report。深度模型、pretrained LM embedding 和长时间 GPU 消融保留为本文档中的显式命令，不默认放进 `rule all`。

### `tests/`

单元测试。当前覆盖：

- ENCODE metadata/file 逻辑。
- quantification 解析和 stability target 构建。
- consensus target、replicate QC、modeling master table。
- GENCODE transcript 解析、UTR/CDS 切分、sequence feature 表。

新增长期功能时，应该同步新增测试。

## 源码模块边界

源码在：

```text
src/rna_stability_elements/
```

当前模块：

```text
config.py
encode.py
quant.py
analysis.py
annotation.py
features.py
visualization.py
cli.py
interpretation.py
models/
```

### `config.py`

读取 `configs/project.yaml`，提供 cell line alias、expected ENCODE terms 等配置帮助函数。

扩展规则：

- 新增全局参数时，先加入 `project.yaml`。
- 只在 `config.py` 写轻量解析。

### `encode.py`

ENCODE Portal 查询、series/experiment/file discovery、文件下载。

输入：

- ENCODE 查询条件。
- 细胞系白名单。

输出：

- `data/interim/encode_series.tsv`
- `data/interim/encode_files.tsv`
- `data/interim/encode_gene_quant_files.tsv`
- downloaded TSV files

扩展方向：

- 新 ENCODE output type。
- 新 assay。
- metadata QC。

### `quant.py`

解析 ENCODE quantification TSV，构建 gene x cell_line stability targets。

核心输出：

- `stability_targets.tsv`
- `stability_targets_gene_sense.tsv`

扩展方向：

- 新 feature type，例如 `exon_sense`、`gene_sense`、其他 strand/feature definition。
- 新 target definition。
- 更严格 replicate-aware target construction。

### `analysis.py`

标签 QC、consensus target、target comparison、replicate QC、modeling master table。

核心输出：

- `stability_consensus_targets_*`
- `target_comparison_exon_vs_gene_sense.tsv`
- `qc_replicate_*`
- `modeling_master_table.tsv`

扩展方向：

- 高低 signal bin QC。
- candidate ranking。
- bootstrap confidence interval。
- ablation summary。

### `annotation.py`

GENCODE GTF 和 transcript FASTA 解析，选择 representative transcript，切分 full / 5'UTR / CDS / 3'UTR。

核心输出：

- `transcript_sequences_gencode_v29.tsv`
- `modeling_master_with_sequences.tsv`

扩展方向：

- MANE Select 专门表。
- APPRIS/Ensembl canonical 外部映射。
- isoform-level 建模。
- genomic sequence extraction。

### `features.py`

序列基础特征，包括 FASTA 读取、RNA normalization、k-mer、motif count、compact sequence model feature table。

核心输出：

- `sequence_model_features.tsv`

扩展方向：

- codon usage。
- RNA structure proxy。
- motif database 批量扫描。
- region-specific feature ablation。

### `models/`

模型相关代码。

```text
models/baselines.py
models/evaluation.py
models/multimodal.py
models/rna_bert.py
models/rna_lm_embeddings.py
models/saluki_like.py
models/sequence_cnn.py
models/sequence_transformer.py
```

当前：

- Ridge
- ElasticNet
- RandomForest
- XGBoost-GPU / XGBoost-CPU
- repeated random split / chromosome holdout / feature ablation
- split-level feature importance export
- region-aware 5'UTR / CDS / 3'UTR CNN
- Saluki-like CNN+GRU with region embedding, configurable convolutional downsampling, and attention pooling
- Conv-tokenized Transformer with region embedding, CLS pooling, and attention pooling
- DNABERT-style k-mer RNA encoder with CLS pooling and attention pooling
- HuggingFace pretrained RNA/DNA LM frozen embedding extraction
- 5'UTR / CDS / 3'UTR pretrained LM embedding merge for strict evaluation
- sequence + expression 多模态模型骨架

扩展方向：

- interpretation module。

### `visualization.py`

当前进度、QC、target distribution、target agreement、sequence landscape、baseline performance 的可视化。

核心输出：

- `docs/figures/*.png`
- `docs/progress_visual_report.md`

扩展方向：

- paper figure panels。
- model interpretation figures。
- feature importance dashboards。
- candidate gene visual summaries。

### `interpretation.py`

阶段一收束报告生成模块。汇总模型 leaderboard、XGBoost feature importance、ElasticNet coefficient direction，并输出第一版 RNA stability grammar interpretation。

核心输出：

```text
data/processed/model_leaderboard.tsv
data/processed/rna_stability_sequence_grammar.tsv
data/processed/rna_stability_feature_group_importance.tsv
docs/rna_stability_grammar_interpretation_report.md
docs/figures/model_leaderboard_*.png
docs/figures/grammar_*.png
```

扩展方向：

- k-mer clustering to motif families。
- in silico mutagenesis summary figures。
- compact vs LM residual interpretation。

### `cli.py`

所有稳定入口都在 CLI 中。优先通过 `rse ...` 使用项目，而不是直接运行内部函数。

当前主要命令：

```text
rse discover-encode
rse download-files
rse build-targets
rse summarize-targets
rse build-consensus-targets
rse compare-targets
rse replicate-qc
rse build-modeling-master
rse build-transcript-sequences
rse merge-master-sequences
rse make-compact-sequence-features
rse train-baseline
rse evaluate-sequence-models
rse train-region-cnn
rse train-saluki-like
rse train-sequence-transformer
rse train-rna-bert
rse extract-rna-lm-embeddings
rse merge-rna-lm-region-embeddings
rse merge-feature-tables
rse write-visual-report
rse write-grammar-report
```

## 常用复现命令

阶段一核心流程：

```bash
snakemake -s workflow/Snakefile --cores 4
```

严格 compact feature 评估：

```bash
CUDA_VISIBLE_DEVICES=0 rse evaluate-sequence-models \
  --features data/processed/sequence_model_features.tsv \
  --target-column target_label \
  --metrics-out data/processed/strict_sequence_model_evaluation_metrics.tsv \
  --predictions-out data/processed/strict_sequence_model_evaluation_predictions.tsv \
  --summary-out data/processed/strict_sequence_model_evaluation_summary.tsv \
  --importance-out data/processed/strict_sequence_xgboost_gpu_importance.tsv \
  --model xgboost_gpu \
  --feature-set all \
  --evaluation repeated_random \
  --evaluation chromosome_holdout \
  --n-repeats 5
```

训练当前最佳 deep sequence model：

```bash
CUDA_VISIBLE_DEVICES=0 rse train-sequence-transformer \
  --table data/processed/modeling_master_with_sequences.tsv \
  --feature-table data/processed/sequence_model_features.tsv \
  --target-column target_label \
  --metrics-out data/processed/sequence_transformer_short_hybrid_random5_metrics.tsv \
  --predictions-out data/processed/sequence_transformer_short_hybrid_random5_predictions.tsv \
  --history-out data/processed/sequence_transformer_short_hybrid_random5_history.tsv \
  --evaluation repeated_random \
  --n-repeats 5 \
  --max-length-5utr 256 \
  --max-length-cds 1024 \
  --max-length-3utr 1024 \
  --model-dim 128 \
  --transformer-layers 2 \
  --attention-heads 4 \
  --conv-pool-size 4 \
  --device cuda
```

提取 pretrained Nucleotide Transformer embedding：

```bash
CUDA_VISIBLE_DEVICES=4 rse extract-rna-lm-embeddings \
  --table data/processed/modeling_master_with_sequences.tsv \
  --out data/processed/rna_lm_nucleotide_transformer_3utr_embeddings.tsv \
  --model-name-or-path InstaDeepAI/nucleotide-transformer-500m-human-ref \
  --sequence-column sequence_3utr \
  --target-column target_label \
  --sequence-format raw \
  --alphabet dna \
  --max-length 512 \
  --chunk-size 512 \
  --chunk-stride 512 \
  --batch-size 1 \
  --device cuda \
  --resume \
  --flush-every 50
```

拼接 5'UTR / CDS / 3'UTR embedding：

```bash
rse merge-rna-lm-region-embeddings \
  --utr5 data/processed/rna_lm_nucleotide_transformer_5utr_embeddings.tsv \
  --cds data/processed/rna_lm_nucleotide_transformer_cds_embeddings.tsv \
  --utr3 data/processed/rna_lm_nucleotide_transformer_3utr_embeddings.tsv \
  --out data/processed/rna_lm_nucleotide_transformer_multi_region_embeddings.tsv
```

生成报告：

```bash
rse write-visual-report \
  --processed-dir data/processed \
  --figure-dir docs/figures \
  --out docs/progress_visual_report.md

rse write-grammar-report \
  --processed-dir data/processed \
  --figure-dir docs/figures \
  --out docs/rna_stability_grammar_interpretation_report.md
```

新增长期功能时，建议同时做三件事：

1. 在 `src/rna_stability_elements/` 中写可测试函数。
2. 在 `cli.py` 中暴露命令。
3. 在 `tests/` 和 `README.md`/`docs/` 中补用法。

## 数据流

当前第一阶段数据流：

```text
ENCODE metadata
  -> encode_series.tsv / encode_gene_quant_files.tsv
  -> raw ENCODE gene quantification TSV
  -> gene_sense gene x cell_line target
  -> gene-level consensus target
  -> replicate QC
  -> modeling_master_table.tsv
  -> GENCODE v29 representative transcript sequences
  -> modeling_master_with_sequences.tsv
  -> sequence_model_features.tsv
  -> strict sequence model evaluation
  -> deep sequence models / pretrained LM embedding / hybrid models
  -> model_leaderboard.tsv
  -> rna_stability_sequence_grammar.tsv
  -> progress and grammar reports
```

核心主表：

```text
data/processed/modeling_master_with_sequences.tsv
```

核心特征表：

```text
data/processed/sequence_model_features.tsv
```

核心结果图文报告：

```text
docs/progress_visual_report.md
docs/rna_stability_grammar_interpretation_report.md
```

## 可扩展性约定

### 新增数据源

推荐放置：

```text
src/rna_stability_elements/<source>.py
configs/project.yaml
data/raw/<source>/
data/interim/<source>_manifest.tsv
```

最小要求：

- manifest 中保留 accession、source URL、本地路径、样本元数据。
- 下载和解析逻辑分开。
- 原始文件不提交到 git。

### 新增 target definition

推荐改动：

- 在 `quant.py` 中新增 target 计算函数。
- 在 `analysis.py` 中新增 QC/summary。
- 输出文件名带清楚 target 来源，例如 `stability_targets_gene_sense.tsv`。
- 不覆盖旧 target，方便横向比较。

### 新增 sequence feature

推荐改动：

- 在 `features.py` 中实现纯函数。
- 在 `make-compact-sequence-features` 中接入。
- 输出列名带 region 前缀，例如 `3utr_motif_AU-rich_count`。

### 新增模型

推荐放置：

```text
src/rna_stability_elements/models/
```

建议接口：

```text
train_xxx(features: pd.DataFrame, target_column: str, split_config: dict) -> dict
```

输出：

```text
data/processed/<model>_metrics.json
data/processed/<model>_predictions.tsv
data/processed/<model>_feature_importance.tsv
```

### 新增可视化

推荐放置：

```text
src/rna_stability_elements/visualization.py
docs/figures/
docs/*.md
```

原则：

- 图直接读取 `data/processed/` 的稳定产物。
- 生成图的命令必须可重复运行。
- 图题和轴标签能独立说明含义，方便放进组会和论文草图。

## 可移植性清单


- 使用 `pyproject.toml` 定义 Python 包。
- 使用 `rse` CLI 作为统一入口。
- 大数据目录被 `.gitignore` 忽略。
- 测试位于 `tests/`。
- 配置集中在 `configs/project.yaml`。

后续补强：

1. 增加 `environment.yml` 或 `requirements-lock.txt`，固定 HPC / 服务器环境。
2. 为大型产物增加 checksum 或 manifest，方便跨机器搬运。
3. 将长时间 GPU 训练整理为 profile 或独立 Snakemake rule group，避免默认工作流过重。
4. 给外部大文件增加下载状态检查和更清晰的错误提示。

## 阅读顺序

建议顺序：

1. `README.md`
2. `docs/rna_stability_grammar_interpretation_report.md`
3. `docs/implementation_guide.md`
4. `configs/project.yaml`
5. `src/rna_stability_elements/cli.py`
6. 对应模块源码和 `tests/`

如果只想快速看成果，打开：

```text
docs/progress_visual_report.md
```

如果想复现流程，从 README 的命令开始。
