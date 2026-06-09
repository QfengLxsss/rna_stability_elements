from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rna_stability_elements.models.ramht import (
    LABEL_IDS,
    RamhtConfig,
    build_multitask_table,
    train_ramht_split,
)
from rna_stability_elements.models.multimodal import require_torch
from rna_stability_elements.models.sequence_cnn import RegionLengths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Region-aware Multi-task Hybrid Transformer on fixed fair splits."
    )
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--split-name", action="append", help="Split to run; default: random_repeat_0.")
    parser.add_argument(
        "--split-set",
        choices=["random0", "all"],
        default="random0",
        help="Convenience split selection when --split-name is not provided.",
    )
    parser.add_argument("--out-prefix", default="data/processed/ramht_multitask")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Allow CPU fallback when --device cuda is requested but CUDA is unavailable.",
    )
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-length-5utr", type=int, default=256)
    parser.add_argument("--max-length-cds", type=int, default=1024)
    parser.add_argument("--max-length-3utr", type=int, default=1024)
    parser.add_argument("--codon-length", type=int, default=342)
    parser.add_argument("--model-dim", type=int, default=192)
    parser.add_argument("--codon-dim", type=int, default=128)
    parser.add_argument("--transformer-layers", type=int, default=3)
    parser.add_argument("--codon-layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--feedforward-dim", type=int, default=512)
    parser.add_argument("--feature-hidden-dim", type=int, default=512)
    parser.add_argument("--head-hidden-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--token-dropout", type=float, default=0.02)
    parser.add_argument("--crop-strategy", default="balanced", choices=["balanced", "start", "end", "random"])
    parser.add_argument(
        "--fusion-mode",
        default="gated_sum",
        choices=["gated_sum", "gated_residual"],
    )
    parser.add_argument(
        "--separate-codon-stream",
        action="store_true",
        help="Keep the codon representation out of the nucleotide-region pooling stream.",
    )
    parser.add_argument(
        "--task-specific-gates",
        action="store_true",
        help="Learn separate sequence/codon/engineered fusion weights for each prediction task.",
    )
    parser.add_argument(
        "--mask-padding-attention",
        action="store_true",
        help="Exclude padding tokens from within-region Transformer attention.",
    )
    parser.add_argument(
        "--engineered-output-skip",
        action="store_true",
        help="Add a direct per-task linear prediction path from engineered features.",
    )
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument(
        "--prediction-roles",
        default="test",
        help="Comma-separated prediction roles to write: test or validation,test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed = Path(args.processed_dir)
    if args.device == "cuda" and not args.allow_cpu_fallback:
        torch, _nn = require_torch()
        if not torch.cuda.is_available():
            raise SystemExit(
                "CUDA was requested, but torch.cuda.is_available() is False. "
                "Move to a GPU node or pass --allow-cpu-fallback for a CPU/debug run."
            )
    config = RamhtConfig(
        region_lengths=RegionLengths(
            utr5=args.max_length_5utr,
            cds=args.max_length_cds,
            utr3=args.max_length_3utr,
        ),
        codon_length=args.codon_length,
        model_dim=args.model_dim,
        codon_dim=args.codon_dim,
        transformer_layers=args.transformer_layers,
        codon_layers=args.codon_layers,
        attention_heads=args.attention_heads,
        feedforward_dim=args.feedforward_dim,
        feature_hidden_dim=args.feature_hidden_dim,
        head_hidden_dim=args.head_hidden_dim,
        dropout=args.dropout,
        token_dropout=args.token_dropout,
        crop_strategy=args.crop_strategy,
        fusion_mode=args.fusion_mode,
        separate_codon_stream=args.separate_codon_stream,
        task_specific_gates=args.task_specific_gates,
        mask_padding_attention=args.mask_padding_attention,
        engineered_output_skip=args.engineered_output_skip,
    )
    table, feature_columns = build_multitask_table(processed)
    split_names = args.split_name or default_split_names(processed, args.split_set)
    metric_frames = []
    prediction_frames = []
    history_frames = []
    for split_name in split_names:
        print(f"[start] RAMHT split={split_name}", flush=True)
        metrics, predictions, history = train_ramht_split(
            table,
            feature_columns,
            processed_dir=processed,
            split_name=split_name,
            config=config,
            label_ids=LABEL_IDS,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            random_state=args.random_state,
            device=args.device,
            prediction_roles=[item.strip() for item in args.prediction_roles.split(",") if item.strip()],
        )
        metric_frames.append(metrics)
        prediction_frames.append(predictions)
        history_frames.append(history)
        print(
            metrics[["label_id", "split_name", "pearson", "spearman", "r2"]].to_string(index=False),
            flush=True,
        )
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(metric_frames, ignore_index=True).to_csv(f"{prefix}_metrics.tsv", sep="\t", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        f"{prefix}_predictions.tsv", sep="\t", index=False
    )
    pd.concat(history_frames, ignore_index=True).to_csv(f"{prefix}_history.tsv", sep="\t", index=False)
    print(f"[done] wrote {prefix}_*.tsv", flush=True)


def default_split_names(processed: Path, split_set: str) -> list[str]:
    if split_set == "random0":
        return ["random_repeat_0"]
    manifest = pd.read_csv(
        processed / f"fair_benchmark_splits_{LABEL_IDS[0]}.tsv",
        sep="\t",
    )
    return manifest["split_name"].drop_duplicates().tolist()


if __name__ == "__main__":
    main()
