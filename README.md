# RNA Stability Elements

本项目研究 RNA 序列能否解释 RNA 稳定性，并进一步为组织或细胞系特异 RNA 调控元件设计打基础。

当前阶段：用 ENCODE BrU-seq / BruChase-seq 数据构建 RNA 稳定性标签，训练不同序列模型预测 gene-level consensus RNA stability，并通过解释分析寻找决定 RNA 稳定性的通用序列语法。

## 文档入口

项目关键信息集中在本 README。更详细的实现和结果放在 [docs/](docs/)：

- 项目实现: [docs/implementation_guide.md](docs/implementation_guide.md)
- 可视化报告: [docs/progress_visual_report.md](docs/progress_visual_report.md)
- 模型 leaderboard 与 grammar 解释: [docs/rna_stability_grammar_interpretation_report.md](docs/rna_stability_grammar_interpretation_report.md)
- 模型实践细节与原理：[docs/models_submodule.md](docs/models_submodule.md)

## 研究思路

长期目标是识别 RNA 稳定性调控的 cis ＋ trans ＋ context 语法：

- `cis`: RNA 自身序列，包括 5'UTR、CDS、3'UTR、motif、k-mer、AU-rich element、miRNA seed、RBP motif 和结构倾向。
- `trans`: 细胞环境中的调控因子，包括 RBP 表达、miRNA 表达、RBP binding / eCLIP 证据。
- `context`: 组织或细胞系表达谱。最终希望在给定细胞环境后，预测同一 RNA 序列在不同细胞中的稳定性变化，并设计能改变 RNA 稳定性的序列模块。

第一阶段先回答一个基础问题：RNA 序列本身是否已经包含可泛化的稳定性语法。希望 sequence-only 模型在严格划分下能稳定预测 RNA stability，并且解释分析能得到合理的 motif / k-mer / region signal。

第一阶段只用序列本身（sequence-only），验证信号是否真实存在，再逐步加入 trans 和 context。即先证明 baseline 信号真实，再做复杂建模

## 技术原理

数据来自 ENCODE4 Ljungman lab 的 16 个细胞系 BrU-seq / BruChase-seq pulse-chase time series。

BrU-seq 用短时间 BrU pulse 标记新生 RNA；BruChase-seq 在 pulse 后加入 uridine chase，并在 2h 和 6h 测量仍保留 BrU 标记的 RNA。直观上，chase 后仍保留较强信号的 RNA 更可能稳定，或经历了更复杂的 processing / maturation 动态。

当前使用三个时间点：

- `0h`: BrU-seq control，近似新生 RNA baseline。
- `2h`: chase 2 小时后的 BrU 标记 RNA。
- `6h`: chase 6 小时后的 BrU 标记 RNA。

相对稳定性标签定义为：

```text
log2_stability_2h_0h = log2((signal_2h + pseudo) / (signal_0h + pseudo))
log2_stability_6h_2h = log2((signal_6h + pseudo) / (signal_2h + pseudo))
log2_stability_6h_0h = log2((signal_6h + pseudo) / (signal_0h + pseudo))
```

第一阶段在 16 个细胞系里，对每个基因取 `gene_sense` 口径下的 `log2_stability_6h_2h` 中位数，作为这个基因的稳定性标签。它是相对稳定性的代理指标，不是绝对 half-life。

## 当前数据

阶段一已经构建出完整的 gene-level sequence modeling 数据集：

| 数据层 | 当前产物 |
| --- | --- |
| ENCODE metadata | 16 个细胞系，48 个 cell-line/time-point experiments |
| 原始量化文件 | 96 个 gene quantification TSV，每个实验 2 个生物学重复 |
| stability target | [data/processed/stability_targets_gene_sense.tsv](data/processed/stability_targets_gene_sense.tsv)，150,233 条 基因 × 细胞系 记录 |
| consensus target | 10,907 个基因，主标签为跨细胞系 `log2_stability_6h_2h` median |
| replicate QC | 2,789,664 条 gene-cell_line-time 记录；48 个 experiment-level QC |
| transcript sequence | GENCODE v29 representative transcript，full / 5'UTR / CDS / 3'UTR 全部映射 |
| compact features | 10,907 x 1,346，包含 region length、GC/AU composition、3/4-mer 和 motif count |
| pretrained LM embedding | Nucleotide Transformer 5'UTR / CDS / 3'UTR frozen embeddings，拼接后 3,840 维 |

`exon_sense` 与 `gene_sense` 标签在共享 gene x cell-line 行上的 Pearson = 0.855，Spearman = 0.836；在 gene-level consensus 上 Pearson = 0.868，Spearman = 0.862。说明这个信号不依赖于某一种特定的特征定义方式 。

Replicate-level QC 整体较好：48 个 cell-line-time experiment 的重复 Pearson 平均值为 0.959，中位数为 0.972。HepG2 的 2h / 6h replicate concordance 偏低，是后续解释时需要关注的风险点。

## 模型结果

> 模型实现原理及比较参阅：[docs/models_submodule.md](docs/models_submodule.md)

当前统一比较使用两类严格评估：

- `repeated random split`: 多次随机 gene split，估计一般泛化能力。
- `chromosome holdout`: 按染色体留出基因，降低相邻基因或局部序列相似性带来的泄漏风险。

核心结果如下：

| 模型 | 输入 | Repeated random Pearson | Chromosome holdout Pearson |
| --- | --- | ---: | ---: |
| XGBoost-GPU | compact k-mer/motif/composition features | 0.512 | 0.496 |
| Hybrid XGBoost-GPU | Nucleotide Transformer embeddings + compact features | 0.498 | 0.494 |
| Conv-tokenized Transformer | raw regional sequence + compact features | 0.489 | 0.470 |
| Region-aware CNN | raw regional sequence + compact features | 0.482 | 0.461 |
| Saluki-like CNN+GRU | raw regional sequence + compact features | 0.482 | 0.457 |
| RandomForest | compact features | 0.475 | 0.461 |
| ElasticNet | compact features | 0.460 | 0.450 |
| Nucleotide Transformer LM-only XGBoost | frozen LM embeddings | 0.442 | 0.442 |

目前结论：

1. RNA 稳定性中存在可被检测到的、不依赖细胞环境的纯序列信号。
2. 当前最强模型仍是 compact k-mer / motif / composition XGBoost-GPU。
3. Conv-tokenized Transformer 是目前最好的深度序列模型，但还没有超过 compact XGBoost。
4. Pretrained Nucleotide Transformer embedding 有稳定性信号；与 compact features 融合后接近 compact XGBoost，但没有形成超越。

## RNA Stability Grammar

当前解释分析结合 XGBoost feature importance、ElasticNet coefficient direction 和 feature group summary。发现：

```text
CDS k-mer > 3'UTR k-mer > full-region k-mer > 5'UTR k-mer
```

这说明目前最主要的可预测信号来自局部序列语法，尤其是 CDS 和 3'UTR 的 k-mer / composition。候选 stabilizing 和 destabilizing k-mers 已整理在：

[data/processed/rna_stability_sequence_grammar.tsv](data/processed/rna_stability_sequence_grammar.tsv)

详细 leaderboard、feature group importance、候选 k-mer 列表和图见：

[docs/rna_stability_grammar_interpretation_report.md](docs/rna_stability_grammar_interpretation_report.md)

这些结果仍是计算候选，下一步可做 in silico perturbation（计算机虚拟扰动）、motif clustering （调控元件聚类）和 residual analysis（残差分析）等

## 仓库结构

```text
configs/                  项目配置、细胞系白名单、motif 设置
data/                     raw / external / interim / processed 数据目录
docs/                     实现指南、结果报告和 figures
scripts/                  绘图或兼容脚本
src/rna_stability_elements/
  encode.py               ENCODE metadata discovery 和下载
  quant.py                quantification TSV 解析与 stability target 构建
  analysis.py             consensus target、replicate QC、target comparison
  annotation.py           GENCODE transcript 解析和 UTR/CDS 切分
  features.py             k-mer、motif、composition 特征
  visualization.py        可视化报告生成
  interpretation.py       leaderboard 和 grammar report 生成
  models/                 baseline、XGBoost、CNN、GRU、Transformer、RNA LM
workflow/Snakefile        阶段核心可复现流程
tests/                    单元测试
```

更完整的模块边界、数据流和扩展约定见 [docs/implementation_guide.md](docs/implementation_guide.md)。

## 快速开始

推荐使用可编辑安装：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,analysis,boosting,deep,rna-lm,workflow]"
```

如果只阅读文档、运行轻量测试和基础数据处理，可以先安装：

```bash
pip install -e ".[dev,analysis]"
```

运行测试：

```bash
pytest tests
```

复现阶段一核心 workflow：

```bash
snakemake -s workflow/Snakefile --cores 4
```

生成当前可视化报告：

```bash
rse write-visual-report \
  --processed-dir data/processed \
  --figure-dir docs/figures \
  --out docs/progress_visual_report.md
```

生成统一模型 leaderboard 和 grammar report：

```bash
rse write-grammar-report \
  --processed-dir data/processed \
  --figure-dir docs/figures \
  --out docs/rna_stability_grammar_interpretation_report.md
```

常用 CLI 命令、完整数据流和长时间 GPU 训练命令见 [docs/implementation_guide.md](docs/implementation_guide.md)。

## 下一步

1. 对 top stabilizing / destabilizing k-mers 做 in silico motif perturbation。
2. 将 top k-mers 聚类成 motif families，形成更可解释的 grammar。
3. 分析 compact XGBoost、Transformer 和 hybrid LM 模型的 residual，寻找当前 grammar 无法解释的序列类别。
4. 可拓展第二阶段：接入 RBP / miRNA expression、RBP binding 和细胞系 context，训练 sequence + context 模型。

## 数据来源与模型参考

- ENCODE Portal: Mats Ljungman lab ENCODE4 BrU-seq / BruChase-seq pulse-chase time series。
- Paper: Narayanan / Bedi / Magnuson 等，`Isoform- and pathway-specific regulation of post-transcriptional RNA processing in human cells`。 bioRxiv 版本为 2024-06-12；截至 2026-05-06，该工作已有 Genome Research 2026-03-26 的期刊版本记录。

本项目的模型路线还参考了以下工作：

- BERT: Devlin et al., `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`, NAACL 2019。BERT 的 masked language modeling 和 bidirectional Transformer 思路是 DNABERT、Nucleotide Transformer、mRNA-LM 等序列语言模型的重要基础。<https://aclanthology.org/N19-1423/>
- DNABERT: Ji et al., `DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome`, Bioinformatics 2021。该工作将 BERT 思路迁移到 DNA k-mer token，是本项目 RNA BERT-style k-mer encoder 的直接方法参照之一。<https://doi.org/10.1093/bioinformatics/btab083>
- Saluki: Agarwal and Kelley, `The genetic and biochemical determinants of mRNA degradation rates in mammals`, Genome Biology 2022。该工作提出 Saluki，用 spliced mRNA sequence 和 gene structure annotations 预测 mRNA half-life，是本项目 Saluki-like CNN+GRU 架构的重要参考。<https://doi.org/10.1186/s13059-022-02811-x>
- Nucleotide Transformer: Dalla-Torre et al., `Nucleotide Transformer: building and evaluating robust foundation models for human genomics`, Nature Methods 2024。当前 pretrained LM embedding 路线使用了 `InstaDeepAI/nucleotide-transformer-500m-human-ref`。<https://pubmed.ncbi.nlm.nih.gov/39609566/>
- mRNA-LM: `mRNA-LM: full-length integrated SLM for mRNA analysis`, Nucleic Acids Research 2025。该工作把不同 mRNA 区域的语言模型整合起来预测 transcript stability、expression、translation rate 和 protein expression，与本项目“5'UTR / CDS / 3'UTR 多区域 embedding 拼接”的思路相关。<https://academic.oup.com/nar/article/53/3/gkaf044/7997216>
- RNA-FM / mRNA-FM: Chen et al., `Interpretable RNA Foundation Model from Unannotated Data for Highly Accurate RNA Structure and Function Predictions`。本项目曾尝试通过 `multimolecule/mrnafm` 接入 mRNA-FM，但受当前环境中 `torch` 与 `multimolecule` 依赖兼容性限制，暂未作为主路线。<https://arxiv.org/abs/2204.00300>
