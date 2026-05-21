# `models/` 子模块说明

本文档介绍 `src/rna_stability_elements/models/` 下各脚本的模型构建思路、数据输入、输出结果和代码实现方式。该子模块服务于 RNA stability prediction：用人工构建的序列特征、预训练 RNA/DNA 语言模型嵌入，以及 5'UTR/CDS/3'UTR 原始序列来预测 `target_label`，通常对应 consensus RNA stability label。

## 总览

`models/` 子模块可以分成四类：

| 脚本 | 主要职责 | 典型 CLI |
| --- | --- | --- |
| `baselines.py` | 训练单次数值特征基线模型 | `rse train-baseline` |
| `evaluation.py` | 对数值特征模型做严格评估、特征组消融和重要性输出 | `rse evaluate-sequence-models` |
| `rna_lm_embeddings.py` | 用 HuggingFace 预训练 RNA/DNA LM 提取 frozen embedding，并合并多区域 embedding | `rse extract-rna-lm-embeddings`, `rse merge-rna-lm-region-embeddings` |
| `sequence_cnn.py` | 直接从 5'UTR/CDS/3'UTR token 训练区域感知 CNN | `rse train-region-cnn` |
| `saluki_like.py` | Saluki 风格 CNN + BiGRU + attention 序列模型 | `rse train-saluki-like` |
| `sequence_transformer.py` | 卷积分词后接 Transformer encoder 的序列模型 | `rse train-sequence-transformer` |
| `rna_bert.py` | DNABERT 风格 k-mer Transformer 序列模型 | `rse train-rna-bert` |
| `multimodal.py` | PyTorch 依赖检查和一个序列 + 表达/context 融合模型构造器 | 当前主要作为工具模块 |
| `__init__.py` | 包初始化说明 | 无 |

## 共同数据约定

### 目标列

多数训练函数默认使用：

- `target_label`：回归目标，通常来自 `modeling_master_table` 中的稳定性 consensus label。

训练前都会先删除目标缺失行：

```python
data = table.dropna(subset=[target_column]).reset_index(drop=True).copy()
```

### 常用元数据列

模型评估输出会尽量保留以下列，便于追踪预测结果：

- `gene_id`
- `gene_symbol`
- `canonical_transcript_id`
- `chromosome`
- `strand`
- `gene_biotype`
- `transcript_biotype`
- `sequence_status`
- `replicate_qc_flag`

其中 `evaluation.py` 会把这些列排除出数值特征集合，避免把基因标识或 QC 标签误当作模型输入。

### 序列列

深度序列模型默认从以下三列读取区域序列：

| 区域 | 输入列 |
| --- | --- |
| 5'UTR | `sequence_5utr` |
| CDS | `sequence_cds` |
| 3'UTR | `sequence_3utr` |

序列会被标准化为 RNA alphabet：`A/C/G/U`，其中 `T` 会替换成 `U`。碱基 token 编码为：

| 符号 | token |
| --- | --- |
| padding/未知 | 0 |
| A | 1 |
| C | 2 |
| G | 3 |
| U | 4 |

### 评估切分

`evaluation.py` 提供统一的 split 逻辑，传统机器学习模型和深度模型都会复用：

- `repeated_random`：重复随机切分，默认测试集比例 `test_size=0.2`。
- `chromosome_holdout`：按染色体留出，只有测试样本数不少于 `min_test_samples` 的染色体会形成 split。

输出指标统一使用 `regression_metrics()`：

- `rmse`
- `mae`
- `r2`
- `pearson`
- `spearman`

## `baselines.py`

### 模型构建思路

`baselines.py` 是最轻量的数值特征回归基线。它从输入表中选取数值列，排除目标列、group 列和用户指定的 `drop_columns`，再用 scikit-learn `Pipeline` 训练回归模型。

支持模型：

- `ridge`：线性 Ridge 回归，适合作为稳定、可解释的基础线。
- `elasticnet`：L1/L2 混合正则线性模型，可做稀疏特征选择。
- `random_forest`：非线性树模型，捕捉 k-mer、motif、长度等特征之间的非线性关系。
- `mlp`：多层感知机回归器，供 `evaluation.py` 使用。
- `xgboost_gpu` / `xgboost_cpu` / `xgboost`：XGBoost 回归器，GPU 版本用于大特征表和 LM embedding。

### 数据输入

入口函数：

```python
train_baseline(features, target_column, model_name, group_column, leave_group)
```

输入表一般是 TSV 读入后的 `pandas.DataFrame`，要求：

- 必须包含 `target_column`。
- 至少有一个可用数值特征列。
- 如果使用 group holdout，表中需要有 `group_column`，默认是 `cell_line`。

CLI：

```bash
rse train-baseline \
  --features data/processed/sequence_model_features.tsv \
  --target-column target_label \
  --model ridge \
  --out data/processed/baseline_metrics.json
```

### 代码实现

实现流程：

1. 删除目标缺失行。
2. 如果未指定 `leave_group`，默认取排序后最后一个 group 做 holdout。
3. 如果没有 group，则退化为确定性 80/20 切分。
4. 选择数值特征列。
5. 构建 Pipeline：`SimpleImputer(strategy="median")` -> `StandardScaler(with_mean=False)` -> 回归模型。
6. 训练、预测并返回指标。

`save_metrics()` 会把指标写成 JSON；`regression_metrics()` 是全模块共享的指标函数。

## `evaluation.py`

### 模型构建思路

`evaluation.py` 是严格模型评估主线。它不定义新的回归器，而是复用 `baselines.py` 的 `_make_model()`，把重点放在：

- 多模型比较。
- 多 feature set 消融。
- 重复随机切分和 chromosome holdout。
- PCA / region-wise PCA 预处理。
- 预测表、summary 表和 feature importance 表输出。

这部分用于回答“哪些特征组真正泛化”“模型在随机切分和染色体留出下是否稳定”等问题。

### 数据输入

入口函数：

```python
evaluate_sequence_models(
    features,
    target_column="target_label",
    models=("elasticnet", "random_forest"),
    feature_sets=("all",),
    evaluations=("repeated_random", "chromosome_holdout"),
)
```

输入表通常是：

- `data/processed/sequence_model_features.tsv`
- 或拼接了预训练 LM embedding 的特征表。

必需列：

- `target_label` 或用户指定目标列。
- 若使用 `chromosome_holdout`，需要 `chromosome`。
- 数值特征列，例如 `full_length`、`full_gc_fraction`、`*_kmer_*`、`*_motif_*`、`lm_*_emb_*`。

CLI：

```bash
rse evaluate-sequence-models \
  --features data/processed/sequence_model_features.tsv \
  --model elasticnet \
  --model xgboost_gpu \
  --feature-set all \
  --feature-set no_kmer \
  --evaluation repeated_random \
  --evaluation chromosome_holdout \
  --metrics-out data/processed/strict_sequence_metrics.tsv \
  --predictions-out data/processed/strict_sequence_predictions.tsv \
  --summary-out data/processed/strict_sequence_summary.tsv \
  --importance-out data/processed/strict_sequence_importance.tsv
```

### 特征组逻辑

`feature_groups()` 按列名规则分组：

- `length`：以 `_length` 结尾。
- `composition`：以 `_gc_fraction`、`_au_fraction`、`_u_fraction` 结尾。
- `motif`：包含 `_motif_`。
- `kmer` / `kmer3` / `kmer4`：包含 `_kmer_` 并按 k-mer 长度拆分。
- `full_region`、`5utr_region`、`cds_region`、`3utr_region`：按区域前缀。
- `lm`、`lm_5utr`、`lm_cds`、`lm_3utr`：预训练 LM embedding。

`resolve_feature_set()` 支持三类选择：

- 直接组名：`all`、`lm_5utr`。
- `*_only`：如 `length_only`、`lm_only`。
- `no_*`：如 `no_kmer`、`no_lm_cds`。

### 代码实现

核心流程：

1. `numeric_feature_columns()` 排除元数据列和目标列。
2. `build_splits()` 生成 repeated random 和 chromosome holdout splits。
3. 对每个 feature set、model、split 调用 `evaluate_one_split()`。
4. `make_evaluation_pipeline()` 构建预处理和模型：
   - `standard`：median impute + standard scale。
   - `pca`：全体特征统一 PCA。
   - `region_pca`：对 `lm_5utr`、`lm_cds`、`lm_3utr` 分别 PCA，其余特征单独标准化。
5. `model_feature_importance()` 读取 `coef_` 或 `feature_importances_`，并标注 feature group。
6. `summarize_evaluation_metrics()` 生成每个 evaluation/model/feature_set 的均值、标准差、中位数、最小值和最大值。

输出四张表：

- metrics：每个 split 的指标。
- predictions：测试样本级预测。
- summary：跨 split 汇总。
- importances：线性系数或树模型重要性。

## `rna_lm_embeddings.py`

### 模型构建思路

该脚本不训练预测模型，而是把预训练 HuggingFace RNA/DNA language model 当作 frozen encoder，为每条 transcript 或区域序列提取 embedding。后续可把 embedding 作为数值特征交给 `evaluation.py` 中的 ElasticNet、XGBoost 等模型。

核心思想：

1. 对长序列按 `chunk_size` 和 `chunk_stride` 切块。
2. 每个 chunk 按模型需要格式化为 raw、spaced characters 或 k-mer string。
3. 送入 `AutoTokenizer` 和 `AutoModel`。
4. 对 `last_hidden_state` 做 attention-mask mean pooling。
5. 对多个 chunk 的 pooled embedding 再取平均，得到每条序列一个固定长度向量。

### 数据输入

单区域 embedding 入口：

```python
write_rna_lm_embeddings(
    table_path,
    out,
    model_name_or_path,
    sequence_column="sequence_full",
)
```

输入 TSV 需要：

- `sequence_column`，默认 `sequence_full`，也可用 `sequence_5utr`、`sequence_cds`、`sequence_3utr`。
- 推荐包含 `gene_id`，用于 resume 和后续 merge。
- 可选元数据列和 `target_label` 会被保留。

CLI 示例：

```bash
rse extract-rna-lm-embeddings \
  --table data/processed/modeling_master_with_sequences.tsv \
  --out data/processed/rna_lm_nucleotide_transformer_3utr_embeddings.tsv \
  --model-name-or-path InstaDeepAI/nucleotide-transformer-500m-human-ref \
  --sequence-column sequence_3utr \
  --sequence-format raw \
  --alphabet dna \
  --max-length 512 \
  --chunk-size 1024 \
  --chunk-stride 1024 \
  --batch-size 8 \
  --device cuda \
  --resume
```

多区域合并入口：

```python
write_multi_region_rna_lm_embeddings(utr5_path, cds_path, utr3_path, out)
```

CLI 示例：

```bash
rse merge-rna-lm-region-embeddings \
  --utr5 data/processed/rna_lm_nucleotide_transformer_5utr_embeddings.tsv \
  --cds data/processed/rna_lm_nucleotide_transformer_cds_embeddings.tsv \
  --utr3 data/processed/rna_lm_nucleotide_transformer_3utr_embeddings.tsv \
  --out data/processed/rna_lm_nucleotide_transformer_multi_region_embeddings.tsv
```

### 代码实现

关键函数：

- `normalize_sequence()`：按 `alphabet` 在 RNA/DNA 间转换，并移除 `N`。
- `sequence_chunks()`：长序列滑窗切块。
- `format_sequence_for_lm()`：支持 `raw`、`spaced_chars`、`kmer`。
- `embed_sequence()`：模型前向、hidden state pooling、chunk 平均。
- `embedding_output_frame()`：生成 `lm_emb_0000` 等 embedding 列并追加元数据。
- `read_region_embedding_table()`：读取单区域 embedding，重命名为 `lm_5utr_emb_*`、`lm_cds_emb_*`、`lm_3utr_emb_*`。

输出表中 embedding 列命名为：

- 单区域：`lm_emb_0000`, `lm_emb_0001`, ...
- 多区域：`lm_5utr_emb_0000`, `lm_cds_emb_0000`, `lm_3utr_emb_0000`, ...

## `sequence_cnn.py`

### 模型构建思路

`sequence_cnn.py` 直接从 5'UTR/CDS/3'UTR 原始序列训练一个区域感知 CNN。它为每个区域使用独立的卷积编码器，然后拼接三个区域 latent；如果提供了数值特征表，还会编码 tabular features 并与序列 latent 融合。

架构：

1. `nn.Embedding(5, embedding_dim)` 把碱基 token 转成向量。
2. 每个区域独立经过 `RegionEncoder`：
   - `Conv1d(kernel_size=9)`
   - `GELU`
   - dilation convolution
   - `AdaptiveMaxPool1d(1)`
3. 拼接 5'UTR/CDS/3'UTR latent。
4. 可选 tabular branch：`Linear` -> `LayerNorm` -> `GELU` -> `Dropout`。
5. MLP head 输出一个回归值。

该模型适合做“原始序列能否捕获稳定性信号”的直接检验。

### 数据输入

入口函数：

```python
evaluate_region_cnn(table, feature_table=None, target_column="target_label")
```

主表默认来自：

- `data/processed/modeling_master_with_sequences.tsv`

要求包含：

- `target_label`
- `sequence_5utr`
- `sequence_cds`
- `sequence_3utr`
- 若使用 chromosome holdout，需要 `chromosome`

可选 `feature_table`：

- 需要与主表都包含 `gene_id`。
- 数值特征会按 `gene_id` 对齐，再做 median impute 和 standard scale。

CLI：

```bash
rse train-region-cnn \
  --table data/processed/modeling_master_with_sequences.tsv \
  --feature-table data/processed/sequence_model_features.tsv \
  --metrics-out data/processed/region_cnn_metrics.tsv \
  --predictions-out data/processed/region_cnn_predictions.tsv \
  --history-out data/processed/region_cnn_training_history.tsv
```

### 代码实现

关键实现：

- `RegionLengths`：定义每个区域最大长度，默认 5'UTR 512、CDS 4096、3'UTR 4096。
- `encode_regions()`：把三列序列编码成定长 token matrix。
- `crop_sequence()`：支持 `balanced`、`start`、`end`、`random`，默认保留序列头尾各一部分。
- `make_train_val_indices()`：从训练 split 中再切 10% validation。
- `RegionSequenceDataset`：返回区域 token、可选数值特征和标准化后的 y。
- `train_region_cnn_split()`：
  - 只用 train 子集估计 y mean/std。
  - 用 SmoothL1Loss。
  - 用 AdamW。
  - 根据 validation loss early stopping。
  - 测试预测后把 y 反标准化，再算指标。

输出：

- metrics TSV：每个 split 的指标、长度参数、dropout、训练 epoch 等。
- predictions TSV：每个测试基因的 `y_true`、`y_pred`、`residual`。
- history TSV：每个 epoch 的 train/validation loss。

## `saluki_like.py`

### 模型构建思路

`saluki_like.py` 实现一个轻量 Saluki 风格序列模型：卷积提取局部 motif/结构信号，GRU 建模下采样后的长程上下文，attention pooling 汇总全序列。相比 `sequence_cnn.py`，它把三个区域拼成一条带 region embedding 的长序列，并显式建模区域顺序。

架构：

1. 拼接 `5utr + cds + 3utr` token。
2. 碱基 embedding 与 region embedding 拼接。
3. 多层 `Conv1d + BatchNorm + GELU + Dropout + MaxPool1d` 下采样。
4. 双向 GRU 建模序列上下文。
5. attention pooling 汇总 token states。
6. 可选 tabular encoder 融合数值特征。
7. MLP head 输出 RNA stability 回归值。

### 数据输入

输入约定与 `sequence_cnn.py` 基本相同：

- 主表：`modeling_master_with_sequences.tsv`
- 必需列：`target_label`、`sequence_5utr`、`sequence_cds`、`sequence_3utr`
- 可选：`feature_table`，按 `gene_id` 对齐数值特征

默认区域长度较短：

- 5'UTR：512
- CDS：2048
- 3'UTR：2048

CLI：

```bash
rse train-saluki-like \
  --table data/processed/modeling_master_with_sequences.tsv \
  --feature-table data/processed/sequence_model_features.tsv \
  --metrics-out data/processed/saluki_like_metrics.tsv \
  --predictions-out data/processed/saluki_like_predictions.tsv \
  --history-out data/processed/saluki_like_training_history.tsv
```

### 代码实现

该脚本复用 `sequence_cnn.py` 中的数据准备函数：

- `encode_regions()`
- `align_numeric_features()`
- `RegionSequenceDataset`
- `make_train_val_indices()`
- `preprocess_numeric_features()`
- `run_epoch()`

自身核心函数：

- `make_region_ids()`：生成与拼接序列等长的 region id，5'UTR 为 1、CDS 为 2、3'UTR 为 3。
- `build_saluki_like_model()`：构造 CNN + BiGRU + attention 模型。
- `train_saluki_like_split()`：执行 split 内训练、early stopping、预测和指标计算。
- `predict_saluki_like()`：测试集前向推理。

模型输出的 `feature_set` 为：

- `raw_5utr_cds_3utr`
- 或提供 tabular 特征时的 `raw_5utr_cds_3utr_plus_tabular`

## `sequence_transformer.py`

### 模型构建思路

`sequence_transformer.py` 是卷积分词 Transformer。直接对全长 token 做 self-attention 成本较高，因此先用卷积和 max pooling 把碱基级序列下采样成较短 token 序列，再用 Transformer encoder 建模区域间和长程依赖。

架构：

1. 拼接 `5utr + cds + 3utr` token。
2. 碱基 embedding 与 region embedding 拼接。
3. 两层卷积分词器：
   - `Conv1d`
   - `BatchNorm1d`
   - `GELU`
   - `Dropout`
   - `MaxPool1d`
4. 加入 learnable `CLS` token 和 position embedding。
5. Transformer encoder。
6. 同时使用：
   - `CLS` pooled representation
   - attention pooled token representation
7. 可选 tabular encoder。
8. MLP head 输出回归值。

### 数据输入

输入约定与 `sequence_cnn.py` 相同。默认区域长度更短，用来控制 Transformer token 数：

- 5'UTR：256
- CDS：1024
- 3'UTR：1024

CLI：

```bash
rse train-sequence-transformer \
  --table data/processed/modeling_master_with_sequences.tsv \
  --feature-table data/processed/sequence_model_features.tsv \
  --metrics-out data/processed/sequence_transformer_metrics.tsv \
  --predictions-out data/processed/sequence_transformer_predictions.tsv \
  --history-out data/processed/sequence_transformer_training_history.tsv
```

### 代码实现

关键函数：

- `total_length()`：计算三段区域拼接后的总长度。
- `build_sequence_transformer_model()`：构造卷积分词器、CLS/position embedding、Transformer encoder、attention pooling 和回归 head。
- `train_sequence_transformer_split()`：训练流程，与 Saluki-like 和 CNN 保持一致。
- `predict_sequence_transformer()`：推理。

实现细节：

- `model_dim` 必须能被 `attention_heads` 整除。
- `conv_downsample_factor = conv_pool_size * conv_pool_size` 会写入 metrics，便于比较 token 压缩比例。
- attention pooling 和 CLS pooling 同时使用，兼顾全局摘要和加权局部证据。

## `rna_bert.py`

### 模型构建思路

`rna_bert.py` 实现 DNABERT 风格 k-mer Transformer。它先把碱基序列转换成 k-mer token，再用 Transformer encoder 学习 k-mer 片段之间的上下文关系。与 `sequence_transformer.py` 不同，它不使用卷积分词，而是直接对 k-mer token 建模。

核心思想：

1. 三个区域分别裁剪并编码成碱基 token。
2. 每个区域按 `kmer_size` 和 `kmer_stride` 转成 k-mer token。
3. 拼接三个区域的 k-mer token 和对应 region id。
4. token embedding + region embedding projection。
5. 加 CLS token 和 position embedding。
6. Transformer encoder，带 padding mask。
7. CLS pooling + masked attention pooling。
8. 可选 tabular encoder。
9. MLP head 回归。

### 数据输入

输入表与其他深度序列模型一致。默认参数：

- 区域长度：5'UTR 256、CDS 1024、3'UTR 1024
- `kmer_size=4`
- `kmer_stride=4`
- `vocab_size = 2 + 4**kmer_size`

CLI：

```bash
rse train-rna-bert \
  --table data/processed/modeling_master_with_sequences.tsv \
  --feature-table data/processed/sequence_model_features.tsv \
  --metrics-out data/processed/rna_bert_metrics.tsv \
  --predictions-out data/processed/rna_bert_predictions.tsv \
  --history-out data/processed/rna_bert_training_history.tsv
```

### 代码实现

关键函数：

- `encode_kmer_regions()`：分别对 5'UTR/CDS/3'UTR 生成 k-mer token，并拼接区域 id。
- `encode_kmer_block()`：将长度为 k 的 A/C/G/U token block 映射到 base-4 vocab。含 padding 或未知 token 的 k-mer 会变成 0。
- `KmerSequenceDataset`：返回 `tokens`、`region_ids`、可选数值特征和标准化 y。
- `build_rna_bert_model()`：构造 k-mer Transformer 回归模型。
- `train_rna_bert_split()`：训练、early stopping、预测和指标。

token 编码中保留两个特殊空间：

- 0：padding/invalid
- 1：未显式使用的保留 token
- 2 起：合法 k-mer token

## `multimodal.py`

### 模型构建思路

`multimodal.py` 当前主要有两个用途：

1. `require_torch()`：集中检查 PyTorch 是否安装。深度模型都通过它导入 `torch` 和 `torch.nn`，如果缺依赖会提示安装 deep extra。
2. `build_sequence_expression_regressor()`：构造一个 compact sequence + context 融合模型。

`build_sequence_expression_regressor()` 的模型结构：

1. `sequence_onehot` 输入形状为 `[batch, 4, length]`。
2. sequence branch 使用两层 `Conv1d` 和 `AdaptiveMaxPool1d(1)` 得到序列 latent。
3. context branch 使用 `Linear` + `LayerNorm` + `GELU` 编码表达或其他上下文向量。
4. 用 context latent 生成 sigmoid gate，调制 sequence latent。
5. 拼接 gated sequence latent 和 context latent 后回归。

### 数据输入

该函数目前不是完整训练管线，只返回 PyTorch module。调用方需要自行提供：

- `sequence_onehot`：float tensor，形状 `[batch, 4, length]`。
- `context_vector`：float tensor，形状 `[batch, context_dim]`。

### 代码实现

`require_torch()` 被 `sequence_cnn.py`、`saluki_like.py`、`sequence_transformer.py` 和 `rna_bert.py` 复用，确保所有深度模型的依赖错误信息一致。

## `__init__.py`

该文件只包含包级 docstring：

```python
"""Model definitions and training helpers."""
```

它的作用是把 `models/` 标记为 Python 子包，方便其他模块用：

```python
from rna_stability_elements.models.sequence_cnn import ...
```

## 深度模型训练流程对比

四个直接序列模型共享同一套训练范式：

| 步骤 | CNN | Saluki-like | Conv Transformer | RNA BERT |
| --- | --- | --- | --- | --- |
| 输入单位 | 碱基 token | 碱基 token | 碱基 token | k-mer token |
| 区域处理 | 三个区域独立 encoder | 三区域拼接 + region embedding | 三区域拼接 + region embedding | 三区域 k-mer 拼接 + region id |
| 长程依赖 | 无显式序列递归/attention | BiGRU + attention | Transformer + attention | Transformer + masked attention |
| 下采样 | 每区域 adaptive pooling | CNN max pooling | CNN max pooling | k-mer stride |
| 可选 tabular 融合 | 支持 | 支持 | 支持 | 支持 |
| loss | SmoothL1Loss | SmoothL1Loss | SmoothL1Loss | SmoothL1Loss |
| optimizer | AdamW | AdamW | AdamW | AdamW |
| early stopping | validation loss | validation loss | validation loss | validation loss |

所有深度模型都会：

1. 复用 `build_splits()` 得到评估 split。
2. 从训练 split 中再取 10% validation。
3. 用训练子集的 y mean/std 标准化目标。
4. 只用训练子集拟合数值特征 imputer/scaler。
5. 根据 validation loss 保存 best state。
6. 在测试集上反标准化预测值，并输出统一回归指标。


