from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge RAMHT per-split run outputs.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-prefix", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    metrics = concat_outputs(run_dir, "_metrics.tsv")
    predictions = concat_outputs(run_dir, "_predictions.tsv")
    history = concat_outputs(run_dir, "_history.tsv")

    metrics.to_csv(f"{out_prefix}_metrics.tsv", sep="\t", index=False)
    predictions.to_csv(f"{out_prefix}_predictions.tsv", sep="\t", index=False)
    history.to_csv(f"{out_prefix}_history.tsv", sep="\t", index=False)
    summarize(metrics).to_csv(f"{out_prefix}_summary.tsv", sep="\t", index=False)
    print(f"[summary] wrote {out_prefix}_*.tsv")


def concat_outputs(run_dir: Path, suffix: str) -> pd.DataFrame:
    paths = sorted(run_dir.glob(f"*{suffix}"))
    if not paths:
        raise FileNotFoundError(f"No files ending with {suffix!r} in {run_dir}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path, sep="\t")
        frame["source_file"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in metrics.groupby(["label_id", "evaluation"], sort=False):
        label_id, evaluation = keys
        row = {
            "label_id": label_id,
            "evaluation": evaluation,
            "n_splits": int(group["split_name"].nunique()),
        }
        for metric in ["pearson", "spearman", "r2", "rmse", "mae"]:
            values = group[metric].dropna().to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values)) if len(values) else np.nan
            row[f"{metric}_median"] = float(np.median(values)) if len(values) else np.nan
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        for gate in ["gate_sequence_mean", "gate_codon_mean", "gate_engineered_mean"]:
            if gate in group.columns:
                row[gate] = float(group[gate].dropna().mean())
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
