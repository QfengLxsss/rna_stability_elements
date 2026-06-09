# Documentation Index

## 当前阶段

- [current_results.md](current_results.md): 当前成果、证据边界、限制和下一步优先级。
- [fair_benchmark_report.md](fair_benchmark_report.md): 固定 cohort、固定 split 下六种模型的公平比较。
- [input_ablation_report.md](input_ablation_report.md): XGBoost 区域与特征类型输入信息消融。
- [deep_input_ablation_report.md](deep_input_ablation_report.md): GPU-full 深度原始序列区域消融与
  raw sequence + engineered features hybrid 对照。
- [deep_input_design_report.md](deep_input_design_report.md): Transformer hybrid 窗口预算、
  裁剪策略和固定总长度区域配额筛选。
- [biological_interpretation_report.md](biological_interpretation_report.md): 区域、k-mer、
  motif、长度组成信号的生物学解释性汇总。
- [mechanistic_interpretation_report.md](mechanistic_interpretation_report.md): codon-aware
  features、group permutation importance 和同义 CDS recoding 机制解释补充。
- [ramht_model_plan.md](ramht_model_plan.md): Region-aware Multi-task Hybrid Transformer
  架构、冒烟训练和正式 GPU 训练计划。
- [ramht_gpu_handoff.md](ramht_gpu_handoff.md): 当前 RAMHT 设计、GPU 单 split 结果、
  8 卡 split 并行训练方案和新会话交接说明。
- [parallel_label_analysis_report.md](parallel_label_analysis_report.md): 四套严格标签的 QC 与
  ElasticNet 平行分析。
- [parallel_model_suite_report.md](parallel_model_suite_report.md): 传统模型、CPU quick 与
  GPU-full 深度模型结果。

## 技术文档

- [data_processing_overview.md](data_processing_overview.md): 从 ENCODE 原始文件到四套建模标签。
- [implementation_guide.md](implementation_guide.md): 代码结构、稳定入口和复现命令。
- [models_submodule.md](models_submodule.md): 各模型实现原理与参数。
- [../scripts/README.md](../scripts/README.md): 项目级实验脚本与运行顺序。

## 历史阶段报告

以下报告记录早期单标签和 sequence grammar 探索，保留用于追踪项目演进，但不代表当前最终
模型排名：

- [progress_visual_report.md](progress_visual_report.md)
- [rna_stability_grammar_interpretation_report.md](rna_stability_grammar_interpretation_report.md)

## 当前核心图

- `figures/current_results_overview.{png,svg,pdf}`: 当前成果总览。
- `figures/gpu_full_model_comparison.{png,svg,pdf}`: GPU-full 深度模型比较。
- `figures/fair_benchmark_overview.{png,svg,pdf}`: 六模型性能、配对差异与计算成本总览。
- `figures/fair_benchmark_split_distributions.{png,svg,pdf}`: chromosome-holdout 性能分布。
- `figures/fair_benchmark_chromosome_heatmap.{png,svg,pdf}`: 各染色体留出性能热图。
- `figures/input_ablation_overview.{png,svg,pdf}`: 输入消融性能与配对差异总览。
- `figures/input_ablation_chromosome_holdout.{png,svg,pdf}`: 输入集合 chromosome-holdout 排名。
- `figures/deep_input_ablation_chromosome_holdout.{png,svg,pdf}`: 深度模型区域输入排名。
- `figures/deep_input_ablation_paired_differences.{png,svg,pdf}`: 深度输入相对 raw-all 的配对差异。
- `figures/deep_input_design_screen_ranking.{png,svg,pdf}`: Transformer hybrid 输入设计排名。
- `figures/deep_input_design_screen_paired_differences.{png,svg,pdf}`: 输入设计相对
  `medium_balanced` 的配对差异。
- `figures/biological_region_feature_signal.{png,svg,pdf}`: 区域 × 特征类型解释信号。
- `figures/biological_top_feature_heatmap.{png,svg,pdf}`: 跨标签候选序列特征热图。
- `figures/biological_length_composition_signal.{png,svg,pdf}`: 长度与组成信号热图。
- `figures/biological_motif_signal.{png,svg,pdf}`: 当前 motif panel 信号热图。
- `figures/mechanistic_codon_feature_performance.{png,svg,pdf}`: codon-aware 特征性能。
- `figures/mechanistic_permutation_importance.{png,svg,pdf}`: XGBoost group permutation importance。
- `figures/mechanistic_synonymous_mutagenesis.{png,svg,pdf}`: 同义 CDS recoding 预测扰动。
- `figures/sample_signal_pca_gene_sense.png`: gene-sense 样本 PCA。
- `figures/sample_signal_pca_exon_sense.png`: exon-sense 样本 PCA。

图表 source data 位于 `data/processed/figure_source_data/`，由
`scripts/generate_current_results_figures.py` 生成。
