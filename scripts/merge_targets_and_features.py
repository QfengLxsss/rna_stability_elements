from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", required=True)
    parser.add_argument("--sequence-features", required=True)
    parser.add_argument("--sequence-id-column", default="gene_id")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    targets = pd.read_csv(args.targets, sep="\t")
    sequence = pd.read_csv(args.sequence_features, sep="\t")
    if "sequence_id" not in sequence:
        raise ValueError("Sequence features must contain a sequence_id column.")

    merged = targets.merge(
        sequence,
        left_on=args.sequence_id_column,
        right_on="sequence_id",
        how="inner",
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {len(merged)} rows to {args.out}")


if __name__ == "__main__":
    main()
