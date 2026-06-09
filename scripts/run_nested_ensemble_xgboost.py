from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from rna_stability_elements.models.evaluation import numeric_feature_columns
from run_fair_classical_benchmark import make_pipeline


LABEL_IDS = (
    "gene_sense_late_chase_6h_2h",
    "gene_sense_total_chase_6h_0h",
    "exon_sense_late_chase_6h_2h",
    "exon_sense_total_chase_6h_0h",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write train-only XGBoost validation and test predictions.")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--label-id")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--out-prefix", default="nested_ensemble_xgboost")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.label_id:
        run_label(
            root,
            label_id=args.label_id,
            gpu=args.gpu,
            processed_dir=args.processed_dir,
            out_prefix=args.out_prefix,
        )
        return

    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if len(gpus) < len(LABEL_IDS):
        raise ValueError(f"Need at least {len(LABEL_IDS)} GPU IDs.")
    processes = []
    for label_id, gpu in zip(LABEL_IDS, gpus):
        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    __file__,
                    "--label-id",
                    label_id,
                    "--gpu",
                    gpu,
                    "--processed-dir",
                    args.processed_dir,
                    "--out-prefix",
                    args.out_prefix,
                ],
                cwd=root,
            )
        )
    codes = [process.wait() for process in processes]
    if any(codes):
        raise SystemExit(f"One or more XGBoost workers failed: {codes}")
    processed = root / args.processed_dir
    paths = [processed / f"{args.out_prefix}_{label_id}_predictions.tsv" for label_id in LABEL_IDS]
    pd.concat([pd.read_csv(path, sep="\t") for path in paths], ignore_index=True).to_csv(
        processed / f"{args.out_prefix}_predictions.tsv", sep="\t", index=False
    )
    print(f"[done] wrote {processed / f'{args.out_prefix}_predictions.tsv'}")


def run_label(
    root: Path,
    *,
    label_id: str,
    gpu: str,
    processed_dir: str,
    out_prefix: str,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    processed = root / processed_dir
    features = pd.read_csv(processed / f"parallel_sequence_model_features_{label_id}.tsv", sep="\t")
    manifest = pd.read_csv(processed / f"fair_benchmark_splits_{label_id}.tsv", sep="\t")
    feature_columns = numeric_feature_columns(features, target_column="target_label")
    frames = []
    for split_name, split_manifest in manifest.groupby("split_name", sort=False):
        print(f"[nested-xgboost] {label_id} / {split_name} / GPU {gpu}", flush=True)
        train_index = split_manifest.loc[split_manifest["role"] == "train", "row_index"].to_numpy()
        pipeline = make_pipeline("xgboost_full")
        pipeline.fit(features.loc[train_index, feature_columns], features.loc[train_index, "target_label"])
        for prediction_role in ["validation", "test"]:
            prediction_index = split_manifest.loc[
                split_manifest["role"] == prediction_role, "row_index"
            ].to_numpy()
            prediction = pipeline.predict(features.loc[prediction_index, feature_columns])
            columns = [
                column
                for column in ["gene_id", "gene_symbol", "chromosome", "replicate_qc_flag"]
                if column in features
            ]
            frame = features.loc[prediction_index, columns].copy()
            frame["label_id"] = label_id
            frame["evaluation"] = split_manifest["evaluation"].iloc[0]
            frame["split_name"] = split_name
            frame["prediction_role"] = prediction_role
            frame["y_true"] = features.loc[prediction_index, "target_label"].to_numpy()
            frame["y_pred"] = prediction
            frame["residual"] = frame["y_pred"] - frame["y_true"]
            frame["model"] = "xgboost_full"
            frames.append(frame)
    out = processed / f"{out_prefix}_{label_id}_predictions.tsv"
    pd.concat(frames, ignore_index=True).to_csv(out, sep="\t", index=False)
    print(f"[done] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
