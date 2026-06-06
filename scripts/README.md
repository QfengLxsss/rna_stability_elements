# Experiment Scripts

长期复用的数据解析、特征工程和模型实现位于 `src/rna_stability_elements/`，并通过
`rse` CLI 暴露。这个目录保存跨模块实验编排、结果汇总和项目级绘图脚本。

## 当前四标签主流程

按以下顺序运行：

```bash
PYTHONPATH=src python scripts/run_four_way_label_analysis.py
PYTHONPATH=src python scripts/summarize_parallel_label_models.py
PYTHONPATH=src python scripts/make_parallel_deep_sequence_tables.py
PYTHONPATH=src python scripts/run_parallel_deep_gpu_full.py --gpus 0,1,2,3 --n-repeats 3
PYTHONPATH=src python scripts/build_current_results.py
```

| 脚本 | 作用 |
| --- | --- |
| `run_four_way_label_analysis.py` | 严格 QC，并构建 gene/exon × 6h/2h、6h/0h 四套标签 |
| `summarize_parallel_label_models.py` | 汇总四套标签的 ElasticNet 严格评估 |
| `make_parallel_deep_sequence_tables.py` | 生成四套原始序列深度学习输入表 |
| `run_parallel_deep_gpu_full.py` | 四 GPU 并行训练三个深度模型，支持跳过已完成任务 |
| `summarize_parallel_deep_gpu_full.py` | 汇总 GPU-full 指标 |
| `summarize_parallel_model_suite.py` | 汇总传统模型、CPU quick 和 GPU-full 结果 |
| `generate_current_results_figures.py` | 生成当前阶段统一结果图与 source data |
| `build_current_results.py` | 一键刷新当前结果汇总与图表 |

## 探索与历史脚本

- `analyze_target_label_choice.py`: 比较 `6h/2h` 与 `6h/0h`。
- `benchmark_parallel_compact_models.py`: 快速传统模型基准，不是最终公平模型排名。
- `draw_project_workflow.py`: 绘制项目流程图。
- `merge_targets_and_features.py`: 兼容早期目标与特征表合并流程。

新增稳定功能时，优先进入 `src/` 并添加测试；只有项目级编排和一次性分析保留在
`scripts/`。
