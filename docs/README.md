# Documentation Index

## 当前阶段

- [current_results.md](current_results.md): 当前成果、证据边界、限制和下一步优先级。
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
- `figures/sample_signal_pca_gene_sense.png`: gene-sense 样本 PCA。
- `figures/sample_signal_pca_exon_sense.png`: exon-sense 样本 PCA。

图表 source data 位于 `data/processed/figure_source_data/`，由
`scripts/generate_current_results_figures.py` 生成。
