from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


SCREEN_LABELS = (
    "gene_sense_late_chase_6h_2h",
    "exon_sense_late_chase_6h_2h",
)
MODELS = {
    "region_cnn": "train-region-cnn",
    "sequence_transformer": "train-sequence-transformer",
    "saluki_like": "train-saluki-like",
}
CONFIGS = {
    **{
        f"{budget}_{crop}": (*lengths, crop, budget)
        for budget, lengths in {
            "short": (128, 512, 512),
            "medium": (256, 1024, 1024),
            "long": (512, 2048, 2048),
        }.items()
        for crop in ("balanced", "start", "end", "random")
    },
    "medium_cds_heavy_balanced": (128, 1664, 512, "balanced", "medium_cds_heavy"),
    "medium_3utr_heavy_balanced": (128, 512, 1664, "balanced", "medium_3utr_heavy"),
}
REUSED_CONFIG = "medium_balanced"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run two-stage GPU-full hybrid sequence-window and crop-design experiments."
    )
    parser.add_argument("--stage", choices=["screen", "expand"], default="screen")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--config", action="append", choices=sorted(CONFIGS))
    parser.add_argument("--best-config")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--worker-job", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.worker_job:
        config_id, model_id, label_id, gpu = args.worker_job.split("|")
        run_job(
            root,
            config_id=config_id,
            model_id=model_id,
            label_id=label_id,
            gpu=gpu,
            n_repeats=args.n_repeats,
            force=args.force,
        )
        return

    jobs = build_jobs(root, args)
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("At least one GPU ID is required.")
    run_scheduler(root, jobs=jobs, gpus=gpus, n_repeats=args.n_repeats, force=args.force)


def build_jobs(root: Path, args: argparse.Namespace) -> list[tuple[str, str, str]]:
    labels = pd.read_csv(root / "data/processed/fair_benchmark_cohort_summary.tsv", sep="\t")[
        "label_id"
    ].tolist()
    if args.stage == "screen":
        configs = args.config or list(CONFIGS)
        return [
            (config_id, "sequence_transformer", label_id)
            for config_id in configs
            if config_id != REUSED_CONFIG
            for label_id in SCREEN_LABELS
        ]
    best_config = args.best_config or read_best_config(root)
    if best_config not in CONFIGS:
        raise ValueError(f"Unknown best configuration: {best_config}")
    return [
        (best_config, model_id, label_id)
        for model_id in MODELS
        for label_id in labels
        if best_config != REUSED_CONFIG
    ]


def read_best_config(root: Path) -> str:
    path = root / "data/processed/deep_input_design_best_config.txt"
    if not path.exists():
        raise ValueError("Run scripts/summarize_deep_input_design.py after screening first.")
    return path.read_text(encoding="utf-8").strip()


def run_scheduler(
    root: Path,
    *,
    jobs: list[tuple[str, str, str]],
    gpus: list[str],
    n_repeats: int,
    force: bool,
) -> None:
    pending = list(jobs)
    active: dict[str, tuple[subprocess.Popen, tuple[str, str, str]]] = {}
    failures = []
    while pending or active:
        for gpu in gpus:
            if gpu in active or not pending:
                continue
            job = pending.pop(0)
            config_id, model_id, label_id = job
            command = [
                sys.executable,
                __file__,
                "--worker-job",
                "|".join([config_id, model_id, label_id, gpu]),
                "--n-repeats",
                str(n_repeats),
            ]
            if force:
                command.append("--force")
            active[gpu] = (subprocess.Popen(command, cwd=root), job)
        for gpu, (process, job) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            if code:
                failures.append((job, code))
            del active[gpu]
        if active:
            time.sleep(10)
    if failures:
        raise SystemExit(f"One or more deep input-design jobs failed: {failures}")


def run_job(
    root: Path,
    *,
    config_id: str,
    model_id: str,
    label_id: str,
    gpu: str,
    n_repeats: int,
    force: bool,
) -> None:
    processed = root / "data/processed"
    log_dir = root / "logs/deep_input_design_gpu_full"
    log_dir.mkdir(parents=True, exist_ok=True)
    utr5, cds, utr3, crop_strategy, budget_family = CONFIGS[config_id]
    prefix = processed / f"deep_input_design_gpu_full_{config_id}_{model_id}_{label_id}"
    metrics_path = Path(f"{prefix}_metrics.tsv")
    predictions_path = Path(f"{prefix}_predictions.tsv")
    history_path = Path(f"{prefix}_history.tsv")
    if metrics_path.exists() and predictions_path.exists() and history_path.exists() and not force:
        print(f"[skip] GPU {gpu} / {config_id} / {model_id} / {label_id}", flush=True)
        return
    command = [
        sys.executable,
        "-m",
        "rna_stability_elements.cli",
        MODELS[model_id],
        "--table",
        str(processed / f"parallel_modeling_master_with_sequences_{label_id}.tsv"),
        "--feature-table",
        str(processed / f"parallel_sequence_model_features_{label_id}.tsv"),
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
        "--evaluation",
        "chromosome_holdout",
        "--n-repeats",
        str(n_repeats),
        "--max-length-5utr",
        str(utr5),
        "--max-length-cds",
        str(cds),
        "--max-length-3utr",
        str(utr3),
        "--crop-strategy",
        crop_strategy,
        "--device",
        "cuda",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = str(root / "src")
    env.setdefault("MPLCONFIGDIR", str(root / ".matplotlib"))
    log_path = log_dir / f"{config_id}_{model_id}_{label_id}.log"
    print(f"[start] GPU {gpu} / {config_id} / {model_id} / {label_id}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\nCOMMAND: " + " ".join(command) + "\n")
        log.flush()
        subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    annotate_and_audit(
        processed,
        config_id=config_id,
        model_id=model_id,
        label_id=label_id,
        budget_family=budget_family,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        history_path=history_path,
    )
    print(f"[done] GPU {gpu} / {config_id} / {model_id} / {label_id}", flush=True)


def annotate_and_audit(
    processed: Path,
    *,
    config_id: str,
    model_id: str,
    label_id: str,
    budget_family: str,
    metrics_path: Path,
    predictions_path: Path,
    history_path: Path,
) -> None:
    frames = [
        pd.read_csv(metrics_path, sep="\t"),
        pd.read_csv(predictions_path, sep="\t"),
        pd.read_csv(history_path, sep="\t"),
    ]
    for frame in frames:
        frame["config_id"] = config_id
        frame["budget_family"] = budget_family
        frame["model_id"] = model_id
        frame["label_id"] = label_id
        frame["input_representation"] = "raw_sequence_plus_engineered"
    metrics, predictions, history = frames
    manifest = pd.read_csv(processed / f"fair_benchmark_splits_{label_id}.tsv", sep="\t")
    exact = []
    for split_name in metrics["split_name"]:
        expected = set(
            manifest.loc[
                (manifest["split_name"] == split_name) & (manifest["role"] == "test"), "gene_id"
            ]
        )
        observed = set(predictions.loc[predictions["split_name"] == split_name, "gene_id"])
        exact.append(expected == observed)
    metrics["manifest_exact_match"] = exact
    if len(metrics) != 26 or not all(exact):
        raise ValueError(f"Split audit failed: {config_id} / {model_id} / {label_id}")
    metrics.to_csv(metrics_path, sep="\t", index=False)
    predictions.to_csv(predictions_path, sep="\t", index=False)
    history.to_csv(history_path, sep="\t", index=False)


if __name__ == "__main__":
    main()
