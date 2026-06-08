from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from rna_stability_elements.models.evaluation import (
    input_ablation_feature_sets,
    numeric_feature_columns,
)
from run_fair_classical_benchmark import evaluate_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed-split XGBoost input-information ablation across four labels."
    )
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--label-id")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--feature-set", action="append")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.label_id:
        run_label(
            root,
            label_id=args.label_id,
            gpu=args.gpu,
            requested_sets=args.feature_set,
            force=args.force,
        )
        return

    summary = pd.read_csv(root / "data/processed/fair_benchmark_cohort_summary.tsv", sep="\t")
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if len(gpus) < len(summary):
        raise ValueError(f"Need at least {len(summary)} GPUs.")
    processes = []
    for item, gpu in zip(summary.itertuples(index=False), gpus):
        command = [sys.executable, __file__, "--label-id", item.label_id, "--gpu", gpu]
        for feature_set in args.feature_set or []:
            command.extend(["--feature-set", feature_set])
        if args.force:
            command.append("--force")
        processes.append(subprocess.Popen(command, cwd=root))
    codes = [process.wait() for process in processes]
    if any(codes):
        raise SystemExit(f"One or more input-ablation workers failed: {codes}")


def run_label(
    root: Path,
    *,
    label_id: str,
    gpu: str,
    requested_sets: list[str] | None,
    force: bool,
) -> None:
    processed = root / "data/processed"
    metrics_out = processed / f"input_ablation_xgboost_{label_id}_metrics.tsv"
    if metrics_out.exists() and not force and not requested_sets:
        print(f"[skip] {label_id}", flush=True)
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    features = pd.read_csv(processed / f"parallel_sequence_model_features_{label_id}.tsv", sep="\t")
    manifest = pd.read_csv(processed / f"fair_benchmark_splits_{label_id}.tsv", sep="\t")
    numeric_columns = numeric_feature_columns(features, target_column="target_label")
    feature_sets = input_ablation_feature_sets(numeric_columns)
    selected_names = requested_sets or list(feature_sets)
    unknown = sorted(set(selected_names) - set(feature_sets))
    if unknown:
        raise ValueError(f"Unknown feature sets: {unknown}. Available: {sorted(feature_sets)}")

    rows = []
    for feature_set in selected_names:
        columns = feature_sets[feature_set]
        print(
            f"[input-ablation] GPU {gpu} / {label_id} / {feature_set} / {len(columns)} features",
            flush=True,
        )
        for _, split_manifest in manifest.groupby("split_name", sort=False):
            metrics, _ = evaluate_split(
                features,
                split_manifest,
                feature_columns=columns,
                model_name="xgboost_full",
                label_id=label_id,
                gpu=gpu,
            )
            metrics["feature_set"] = feature_set
            metrics["input_representation"] = f"engineered_features:{feature_set}"
            rows.append(metrics)
    result = pd.DataFrame(rows)
    if requested_sets and metrics_out.exists() and not force:
        previous = pd.read_csv(metrics_out, sep="\t")
        previous = previous[~previous["feature_set"].isin(selected_names)]
        result = pd.concat([previous, result], ignore_index=True)
    result.to_csv(metrics_out, sep="\t", index=False)


if __name__ == "__main__":
    main()
