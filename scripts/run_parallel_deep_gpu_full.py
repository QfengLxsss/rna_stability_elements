from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


MODELS = (
    ("region_cnn", "train-region-cnn"),
    ("sequence_transformer", "train-sequence-transformer"),
    ("saluki_like", "train-saluki-like"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full raw-sequence deep-learning benchmarks across the four parallel labels."
    )
    parser.add_argument("--gpus", default="0,1,2,3", help="Comma-separated physical GPU IDs.")
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=None, help="Override each model's full default.")
    parser.add_argument("--patience", type=int, default=None, help="Override each model's full default.")
    parser.add_argument("--force", action="store_true", help="Rerun completed jobs.")
    parser.add_argument("--worker-label", help=argparse.SUPPRESS)
    return parser.parse_args()


def command_for_job(
    root: Path,
    label_id: str,
    model_name: str,
    cli_command: str,
    n_repeats: int,
    max_epochs: int | None,
    patience: int | None,
) -> list[str]:
    processed = root / "data/processed"
    prefix = processed / f"parallel_deep_gpu_full_{model_name}_{label_id}"
    command = [
        sys.executable,
        "-m",
        "rna_stability_elements.cli",
        cli_command,
        "--table",
        str(processed / f"parallel_modeling_master_with_sequences_{label_id}.tsv"),
        "--target-column",
        "target_label",
        "--metrics-out",
        f"{prefix}_metrics.tsv",
        "--predictions-out",
        f"{prefix}_predictions.tsv",
        "--history-out",
        f"{prefix}_history.tsv",
        "--evaluation",
        "repeated_random",
        "--evaluation",
        "chromosome_holdout",
        "--n-repeats",
        str(n_repeats),
        "--device",
        "cuda",
    ]
    if max_epochs is not None:
        command.extend(["--max-epochs", str(max_epochs)])
    if patience is not None:
        command.extend(["--patience", str(patience)])
    return command


def run_label(
    root: Path,
    label_id: str,
    gpu: str,
    n_repeats: int,
    max_epochs: int | None,
    patience: int | None,
    force: bool,
) -> None:
    log_dir = root / "logs/parallel_deep_gpu_full"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = str(root / "src")
    env.setdefault("MPLCONFIGDIR", str(root / ".matplotlib"))

    for model_name, cli_command in MODELS:
        metrics = root / f"data/processed/parallel_deep_gpu_full_{model_name}_{label_id}_metrics.tsv"
        if metrics.exists() and not force:
            print(f"[skip] {label_id} / {model_name}: {metrics.name} exists", flush=True)
            continue
        command = command_for_job(
            root, label_id, model_name, cli_command, n_repeats, max_epochs, patience
        )
        log_path = log_dir / f"{label_id}_{model_name}.log"
        print(f"[start] GPU {gpu}: {label_id} / {model_name}", flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write("\nCOMMAND: " + " ".join(command) + "\n")
            log.flush()
            subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"[done] GPU {gpu}: {label_id} / {model_name}", flush=True)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.worker_label:
        run_label(
            root,
            args.worker_label,
            args.gpus.split(",")[0].strip(),
            args.n_repeats,
            args.max_epochs,
            args.patience,
            args.force,
        )
        return

    table = pd.read_csv(root / "data/processed/parallel_deep_sequence_tables.tsv", sep="\t")
    label_ids = table["label_id"].tolist()
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if len(gpus) < len(label_ids):
        raise ValueError(f"Need at least {len(label_ids)} GPU IDs, received {len(gpus)}.")

    processes = []
    for label_id, gpu in zip(label_ids, gpus):
        command = [
            sys.executable,
            __file__,
            "--gpus",
            gpu,
            "--n-repeats",
            str(args.n_repeats),
            "--worker-label",
            label_id,
        ]
        if args.max_epochs is not None:
            command.extend(["--max-epochs", str(args.max_epochs)])
        if args.patience is not None:
            command.extend(["--patience", str(args.patience)])
        if args.force:
            command.append("--force")
        processes.append(subprocess.Popen(command, cwd=root))

    failed = [process.wait() for process in processes]
    if any(failed):
        raise SystemExit(f"One or more label workers failed: {failed}")


if __name__ == "__main__":
    main()
