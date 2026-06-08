from __future__ import annotations

import argparse
import os
import pickle
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rna_stability_elements.models.baselines import regression_metrics
from rna_stability_elements.models.evaluation import numeric_feature_columns


MODELS = ("elasticnet_full", "random_forest_full", "xgboost_full")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full classical models on fixed fair-benchmark splits.")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--label-id")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.label_id:
        run_label(root, label_id=args.label_id, gpu=args.gpu, force=args.force)
        return
    summary = pd.read_csv(root / "data/processed/fair_benchmark_cohort_summary.tsv", sep="\t")
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if len(gpus) < len(summary):
        raise ValueError(f"Need at least {len(summary)} GPUs.")
    processes = []
    for item, gpu in zip(summary.itertuples(index=False), gpus):
        command = [sys.executable, __file__, "--label-id", item.label_id, "--gpu", gpu]
        if args.force:
            command.append("--force")
        processes.append(subprocess.Popen(command, cwd=root))
    codes = [process.wait() for process in processes]
    if any(codes):
        raise SystemExit(f"One or more workers failed: {codes}")


def run_label(root: Path, *, label_id: str, gpu: str, force: bool) -> None:
    processed = root / "data/processed"
    metrics_out = processed / f"fair_benchmark_classical_{label_id}_metrics.tsv"
    predictions_out = processed / f"fair_benchmark_classical_{label_id}_predictions.tsv"
    if metrics_out.exists() and predictions_out.exists() and not force:
        print(f"[skip] {label_id}", flush=True)
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    features = pd.read_csv(processed / f"parallel_sequence_model_features_{label_id}.tsv", sep="\t")
    manifest = pd.read_csv(processed / f"fair_benchmark_splits_{label_id}.tsv", sep="\t")
    feature_columns = numeric_feature_columns(features, target_column="target_label")
    metrics_rows = []
    prediction_frames = []
    for model_name in MODELS:
        print(f"[fair-classical] GPU {gpu} / {label_id} / {model_name}", flush=True)
        for split_name, split_manifest in manifest.groupby("split_name", sort=False):
            metrics, predictions = evaluate_split(
                features,
                split_manifest,
                feature_columns=feature_columns,
                model_name=model_name,
                label_id=label_id,
                gpu=gpu,
            )
            metrics_rows.append(metrics)
            prediction_frames.append(predictions)
    pd.DataFrame(metrics_rows).to_csv(metrics_out, sep="\t", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(predictions_out, sep="\t", index=False)


def evaluate_split(
    data: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    feature_columns: list[str],
    model_name: str,
    label_id: str,
    gpu: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    train_index = manifest.loc[manifest["role"] == "train", "row_index"].to_numpy()
    validation_index = manifest.loc[manifest["role"] == "validation", "row_index"].to_numpy()
    test_index = manifest.loc[manifest["role"] == "test", "row_index"].to_numpy()
    pipeline = make_pipeline(model_name)
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    pipeline.fit(data.loc[train_index, feature_columns], data.loc[train_index, "target_label"])
    fit_wall_seconds = time.perf_counter() - wall_start
    fit_cpu_seconds = time.process_time() - cpu_start
    predict_start = time.perf_counter()
    prediction = pipeline.predict(data.loc[test_index, feature_columns])
    predict_wall_seconds = time.perf_counter() - predict_start
    y_true = data.loc[test_index, "target_label"].to_numpy()
    metrics: dict[str, object] = regression_metrics(y_true, prediction)
    metrics.update(
        {
            "label_id": label_id,
            "model": model_name,
            "input_representation": "engineered_features",
            "evaluation": manifest["evaluation"].iloc[0],
            "split_name": manifest["split_name"].iloc[0],
            "holdout_group": manifest["holdout_group"].iloc[0],
            "repeat": int(manifest["repeat"].iloc[0]),
            "n_train": len(train_index),
            "n_validation": len(validation_index),
            "n_test": len(test_index),
            "n_features": len(feature_columns),
            "fit_wall_seconds": fit_wall_seconds,
            "fit_cpu_seconds": fit_cpu_seconds,
            "predict_wall_seconds": predict_wall_seconds,
            "peak_process_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "serialized_model_mb": len(pickle.dumps(pipeline, protocol=pickle.HIGHEST_PROTOCOL))
            / (1024**2),
            "compute_device": f"cuda:{gpu}" if model_name == "xgboost_full" else "cpu",
            "cpu_threads": 4,
        }
    )
    columns = [column for column in ["gene_id", "gene_symbol", "chromosome"] if column in data]
    predictions = data.loc[test_index, columns].copy()
    predictions["label_id"] = label_id
    predictions["model"] = model_name
    predictions["evaluation"] = manifest["evaluation"].iloc[0]
    predictions["split_name"] = manifest["split_name"].iloc[0]
    predictions["y_true"] = y_true
    predictions["y_pred"] = prediction
    predictions["residual"] = prediction - y_true
    return metrics, predictions


def make_pipeline(model_name: str) -> Pipeline:
    if model_name == "elasticnet_full":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
                (
                    "model",
                    ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20000, random_state=13),
                ),
            ]
        )
    if model_name == "random_forest_full":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=400,
                        max_features="sqrt",
                        min_samples_leaf=3,
                        n_jobs=4,
                        random_state=13,
                    ),
                ),
            ]
        )
    if model_name == "xgboost_full":
        from xgboost import XGBRegressor

        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBRegressor(
                        objective="reg:squarederror",
                        n_estimators=600,
                        max_depth=4,
                        learning_rate=0.03,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_alpha=0.0,
                        reg_lambda=1.0,
                        tree_method="hist",
                        device="cuda",
                        n_jobs=4,
                        random_state=13,
                    ),
                ),
            ]
        )
    raise ValueError(model_name)


if __name__ == "__main__":
    main()
