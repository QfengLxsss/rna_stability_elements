from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd


MODELS = (
    ("region_cnn", "train-region-cnn"),
    ("sequence_transformer", "train-sequence-transformer"),
    ("saluki_like", "train-saluki-like"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure one fixed-split GPU cost for each deep model.")
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
    processes = []
    for item, gpu in zip(summary.itertuples(index=False), gpus):
        command = [sys.executable, __file__, "--label-id", item.label_id, "--gpu", gpu]
        if args.force:
            command.append("--force")
        processes.append(subprocess.Popen(command, cwd=root))
    codes = [process.wait() for process in processes]
    if any(codes):
        raise SystemExit(f"One or more deep cost workers failed: {codes}")
    frames = [
        pd.read_csv(path, sep="\t")
        for path in sorted((root / "data/processed").glob("fair_benchmark_deep_cost_*_summary.tsv"))
    ]
    pd.concat(frames, ignore_index=True).to_csv(
        root / "data/processed/fair_benchmark_deep_cost_summary.tsv", sep="\t", index=False
    )


def run_label(root: Path, *, label_id: str, gpu: str, force: bool) -> None:
    processed = root / "data/processed"
    summary_out = processed / f"fair_benchmark_deep_cost_{label_id}_summary.tsv"
    if summary_out.exists() and not force:
        print(f"[skip] deep cost {label_id}", flush=True)
        return
    rows = []
    for model, command_name in MODELS:
        prefix = processed / f"fair_benchmark_deep_cost_{model}_{label_id}"
        metrics_path = Path(f"{prefix}_metrics.tsv")
        predictions_path = Path(f"{prefix}_predictions.tsv")
        history_path = Path(f"{prefix}_history.tsv")
        command = [
            sys.executable,
            "-m",
            "rna_stability_elements.cli",
            command_name,
            "--table",
            str(processed / f"parallel_modeling_master_with_sequences_{label_id}.tsv"),
            "--target-column",
            "target_label",
            "--metrics-out",
            str(metrics_path),
            "--predictions-out",
            str(predictions_path),
            "--history-out",
            str(history_path),
            "--evaluation",
            "repeated_random",
            "--n-repeats",
            "1",
            "--device",
            "cuda",
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["PYTHONPATH"] = str(root / "src")
        print(f"[deep-cost] GPU {gpu} / {label_id} / {model}", flush=True)
        stop = threading.Event()
        samples: list[tuple[float, float]] = []
        monitor = threading.Thread(target=monitor_gpu, args=(gpu, stop, samples), daemon=True)
        monitor.start()
        start = time.perf_counter()
        subprocess.run(command, cwd=root, env=env, check=True)
        wall_seconds = time.perf_counter() - start
        stop.set()
        monitor.join()
        metrics = pd.read_csv(metrics_path, sep="\t").iloc[0]
        expected = set(
            pd.read_csv(processed / f"fair_benchmark_splits_{label_id}.tsv", sep="\t")
            .query("split_name == 'random_repeat_0' and role == 'test'")["gene_id"]
        )
        observed = set(pd.read_csv(predictions_path, sep="\t")["gene_id"])
        rows.append(
            {
                "label_id": label_id,
                "model": model,
                "split_name": "random_repeat_0",
                "wall_seconds": wall_seconds,
                "peak_gpu_memory_mb": max((item[0] for item in samples), default=float("nan")),
                "mean_gpu_utilization_percent": (
                    sum(item[1] for item in samples) / len(samples) if samples else float("nan")
                ),
                "epochs_trained": metrics["epochs_trained"],
                "n_train": metrics["n_train"],
                "n_validation": metrics["n_validation"],
                "n_test": metrics["n_test"],
                "manifest_exact_match": expected == observed,
                "device": metrics["device"],
            }
        )
    pd.DataFrame(rows).to_csv(summary_out, sep="\t", index=False)


def monitor_gpu(gpu: str, stop: threading.Event, samples: list[tuple[float, float]]) -> None:
    while not stop.is_set():
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu}",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            memory, utilization = result.stdout.strip().split(",")
            samples.append((float(memory.strip()), float(utilization.strip())))
        stop.wait(1.0)


if __name__ == "__main__":
    main()
