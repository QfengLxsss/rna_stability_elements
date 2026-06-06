from __future__ import annotations

from pathlib import Path

import pandas as pd


MODELS = ("region_cnn", "sequence_transformer", "saluki_like")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    processed = root / "data/processed"
    metadata = pd.read_csv(processed / "parallel_label_feature_tables.tsv", sep="\t").set_index("label_id")
    rows = []
    for label_id, meta in metadata.iterrows():
        for model in MODELS:
            path = processed / f"parallel_deep_gpu_full_{model}_{label_id}_metrics.tsv"
            if not path.exists():
                continue
            metrics = pd.read_csv(path, sep="\t")
            for evaluation, group in metrics.groupby("evaluation"):
                rows.append(
                    {
                        "label_id": label_id,
                        "feature_type": meta["feature_type"],
                        "label_key": meta["label_key"],
                        "target_column": meta["target_column"],
                        "model": model,
                        "input_representation": "raw_sequence_only",
                        "evaluation": evaluation,
                        "n_splits": len(group),
                        "pearson_median": group["pearson"].median(),
                        "spearman_median": group["spearman"].median(),
                        "r2_median": group["r2"].median(),
                        "rmse_median": group["rmse"].median(),
                    }
                )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["evaluation", "label_id", "pearson_median"], ascending=[True, True, False]
        )
    out = processed / "parallel_deep_gpu_full_summary.tsv"
    summary.to_csv(out, sep="\t", index=False)
    print(summary.to_string(index=False) if not summary.empty else "No completed GPU-full metrics yet.")


if __name__ == "__main__":
    main()
