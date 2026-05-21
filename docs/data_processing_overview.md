## 项目数据处理全流程分析

### 第一阶段：从 ENCODE 获取原始数据

```text
ENCODE Portal
    ↓ encode.py
    ↓ rse discover-encode
    ↓ rse download-files
```

**做了什么：**

- 根据 `configs/project.yaml` 中的查询条件（细胞系白名单、实验类型），在 ENCODE 数据库中检索符合条件的 BrU-seq / BruChase-seq 实验
- 生成两张清单表：

```text
data/interim/encode_series.tsv        → 实验系列元数据
data/interim/encode_gene_quant_files.tsv → 每个文件的路径和来源
```

- 下载 96 个基因定量 TSV 文件（16 个细胞系 × 3 个时间点 × 2 个生物学重复）到 `data/raw/`

------

### 第二阶段：构建稳定性标签

```text
96个原始TSV
    ↓ quant.py
    ↓ rse build-targets
```

**做了什么：**

- 解析每个 TSV，提取每个基因在每个时间点的信号值
- 计算三个 log2 稳定性比值：

```text
log2_stability_2h_0h
log2_stability_6h_2h   ← 主标签
log2_stability_6h_0h
```

- 按 gene_sense 和 exon_sense 两种口径分别计算
- 输出：

```text
data/processed/stability_targets_gene_sense.tsv
→ 150,233 条（基因 × 细胞系）
```

------

### 第三阶段：标签质量控制与consensus压缩

```text
stability_targets_gene_sense.tsv
    ↓ analysis.py
    ↓ rse replicate-qc
    ↓ rse build-consensus-targets
    ↓ rse compare-targets
```

**做了什么：**

① **重复实验质量控制**

- 计算每个实验两个生物学重复之间的 Pearson 相关
- 48 个实验平均 Pearson = 0.959，标记 HepG2 为风险点
- 输出：`data/processed/qc_replicate_*.tsv`

② **跨细胞系取consensus**

- 对每个基因，取 16 个细胞系的 `log2_stability_6h_2h` 中位数
- 从 150,233 条压缩到 10,907 个基因
- 输出：`data/processed/stability_consensus_targets_*.tsv`

③ **两种口径一致性验证**

- 比较 gene_sense 和 exon_sense 的 Pearson / Spearman
- 确认信号不是某种计数方式的人工产物
- 输出：`data/processed/target_comparison_exon_vs_gene_sense.tsv`

------

### 第四阶段：序列获取与特征提取

```text
GENCODE v29 GTF + FASTA
    ↓ annotation.py
    ↓ rse build-transcript-sequences

modeling_master_table + sequences
    ↓ features.py
    ↓ rse make-compact-sequence-features
```

**做了什么：**

① **序列获取**

- 从 GENCODE v29 中为每个基因选一条代表性转录本
- 切分出 5'UTR / CDS / 3'UTR 三段序列
- 与稳定性标签合并：

```text
data/processed/modeling_master_with_sequences.tsv
→ 10,907 个基因，每行含序列 + 稳定性标签
```

② **紧凑特征提取**

- 对每个区域计算：区域长度、GC/AU 组成、3/4-mer 计数、已知调控元件计数
- 输出：

```text
data/processed/sequence_model_features.tsv
→ 10,907 × 1,346 特征矩阵
```

------

### 第五阶段：模型训练与评估

```text
sequence_model_features.tsv  →  传统模型（XGBoost / ElasticNet / RF）

modeling_master_with_sequences.tsv  
→  深度模型（CNN / GRU / Transformer）
→  预训练模型（Nucleotide Transformer）
    ↓ models/
    ↓ rse evaluate-sequence-models
    ↓ rse train-region-cnn / train-saluki-like / train-sequence-transformer
    ↓ rse extract-rna-lm-embeddings + rse merge-rna-lm-region-embeddings
```

**做了什么：**

- 统一用两种严格评估方式：重复随机划分 + 染色体留出
- 传统模型直接在紧凑特征上训练
- 深度模型从原始序列学习表示，可附加紧凑特征（hybrid）
- 预训练模型分别提取 5'UTR / CDS / 3'UTR 的嵌入，拼接后再训练 XGBoost
- 每个模型输出：

```text
data/processed/<model>_metrics.tsv       → 性能指标
data/processed/<model>_predictions.tsv   → 预测值
data/processed/<model>_importance.tsv    → 特征重要性
```

------

### 第六阶段：解释分析与报告生成

```text
model_leaderboard + feature importance + ElasticNet coefficients
    ↓ interpretation.py
    ↓ rse write-grammar-report
    ↓ rse write-visual-report
```

**做了什么：**

- 汇总所有模型性能，生成排行榜
- 分析特征重要性，按区域分组（CDS / 3'UTR / 5'UTR）
- 用 ElasticNet 系数方向判断每个 k-mer 是稳定化还是不稳定化
- 输出最终报告：

```text
data/processed/model_leaderboard.tsv
data/processed/rna_stability_sequence_grammar.tsv   → 候选 k-mer 列表
docs/progress_visual_report.md                      → 进度图文报告
docs/rna_stability_grammar_interpretation_report.md → 语法解释报告
```

------

### 整体数据流示意

```text
ENCODE 数据库
    ↓ 下载 96 个 TSV
原始信号
    ↓ 计算 log2 比值
稳定性标签（150,233 条）
    ↓ 质量控制 + 取中位数
共识标签（10,907 个基因）
    ↓                        ↓
紧凑特征矩阵            原始序列
（1,346 维）            （5'UTR/CDS/3'UTR）
    ↓                        ↓
传统模型              深度模型 / 预训练模型
    ↓                        ↓
            模型排行榜
                ↓
        序列语法解释报告
```

------

