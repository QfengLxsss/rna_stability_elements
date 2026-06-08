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
REGION_COLUMNS = ("sequence_5utr", "sequence_cds", "sequence_3utr")
CONDITION_REGIONS = {
    "raw_all_plus_engineered": set(REGION_COLUMNS),
    "raw_cds_only": {"sequence_cds"},
    "raw_no_cds": {"sequence_5utr", "sequence_3utr"},
    "raw_no_5utr": {"sequence_cds", "sequence_3utr"},
    "raw_no_3utr": {"sequence_5utr", "sequence_cds"},
    "raw_5utr_only": {"sequence_5utr"},
    "raw_3utr_only": {"sequence_3utr"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GPU-full raw-sequence region ablation and sequence-plus-engineered hybrid."
    )
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--condition", action="append", choices=sorted(CONDITION_REGIONS))
    parser.add_argument("--model", action="append", choices=[item[0] for item in MODELS])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--worker-label", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.worker_label:
        run_label(
            root,
            label_id=args.worker_label,
            gpu=args.gpus.split(",")[0].strip(),
            conditions=args.condition or list(CONDITION_REGIONS),
            models=args.model,
            n_repeats=args.n_repeats,
            force=args.force,
        )
        return

    labels = pd.read_csv(root / "data/processed/fair_benchmark_cohort_summary.tsv", sep="\t")[
        "label_id"
    ].tolist()
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if len(gpus) < len(labels):
        raise ValueError(f"Need at least {len(labels)} GPU IDs.")
    processes = []
    for label_id, gpu in zip(labels, gpus):
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
        for condition in args.condition or []:
            command.extend(["--condition", condition])
        for model in args.model or []:
            command.extend(["--model", model])
        if args.force:
            command.append("--force")
        processes.append(subprocess.Popen(command, cwd=root))
    codes = [process.wait() for process in processes]
    if any(codes):
        raise SystemExit(f"One or more deep input-ablation workers failed: {codes}")


def run_label(
    root: Path,
    *,
    label_id: str,
    gpu: str,
    conditions: list[str],
    models: list[str] | None,
    n_repeats: int,
    force: bool,
) -> None:
    processed = root / "data/processed"
    log_dir = root / "logs/deep_input_ablation_gpu_full"
    log_dir.mkdir(parents=True, exist_ok=True)
    selected_models = [item for item in MODELS if models is None or item[0] in models]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = str(root / "src")
    env.setdefault("MPLCONFIGDIR", str(root / ".matplotlib"))

    for condition in conditions:
        table_path = prepare_condition_table(processed, label_id=label_id, condition=condition)
        for model, cli_command in selected_models:
            prefix = processed / f"deep_input_ablation_gpu_full_{condition}_{model}_{label_id}"
            metrics_path = Path(f"{prefix}_metrics.tsv")
            predictions_path = Path(f"{prefix}_predictions.tsv")
            history_path = Path(f"{prefix}_history.tsv")
            if metrics_path.exists() and predictions_path.exists() and not force:
                print(f"[skip] GPU {gpu} / {label_id} / {condition} / {model}", flush=True)
                continue
            command = [
                sys.executable,
                "-m",
                "rna_stability_elements.cli",
                cli_command,
                "--table",
                str(table_path),
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
                "--device",
                "cuda",
            ]
            if condition == "raw_all_plus_engineered":
                command.extend(
                    [
                        "--feature-table",
                        str(processed / f"parallel_sequence_model_features_{label_id}.tsv"),
                    ]
                )
            log_path = log_dir / f"{label_id}_{condition}_{model}.log"
            print(f"[start] GPU {gpu} / {label_id} / {condition} / {model}", flush=True)
            with log_path.open("a", encoding="utf-8") as log:
                log.write("\nCOMMAND: " + " ".join(command) + "\n")
                log.flush()
                subprocess.run(
                    command,
                    cwd=root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            annotate_and_audit(
                processed,
                label_id=label_id,
                condition=condition,
                model=model,
                metrics_path=metrics_path,
                predictions_path=predictions_path,
                history_path=history_path,
            )
            print(f"[done] GPU {gpu} / {label_id} / {condition} / {model}", flush=True)


def prepare_condition_table(processed: Path, *, label_id: str, condition: str) -> Path:
    source = processed / f"parallel_modeling_master_with_sequences_{label_id}.tsv"
    if condition == "raw_all_plus_engineered":
        return source
    output = processed / f"deep_input_ablation_table_{condition}_{label_id}.tsv"
    if output.exists():
        return output
    table = pd.read_csv(source, sep="\t")
    keep = CONDITION_REGIONS[condition]
    for column in REGION_COLUMNS:
        if column not in keep:
            table[column] = ""
    table.to_csv(output, sep="\t", index=False)
    return output


def annotate_and_audit(
    processed: Path,
    *,
    label_id: str,
    condition: str,
    model: str,
    metrics_path: Path,
    predictions_path: Path,
    history_path: Path,
) -> None:
    metrics = pd.read_csv(metrics_path, sep="\t")
    predictions = pd.read_csv(predictions_path, sep="\t")
    history = pd.read_csv(history_path, sep="\t")
    for frame in [metrics, predictions, history]:
        frame["label_id"] = label_id
        frame["input_condition"] = condition
        frame["model_id"] = model
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
    if not all(exact):
        raise ValueError(f"Manifest mismatch: {label_id} / {condition} / {model}")
    metrics.to_csv(metrics_path, sep="\t", index=False)
    predictions.to_csv(predictions_path, sep="\t", index=False)
    history.to_csv(history_path, sep="\t", index=False)


if __name__ == "__main__":
    main()
