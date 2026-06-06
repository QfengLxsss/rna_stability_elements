from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    processed_dir = root / "data/processed"
    base = pd.read_csv(processed_dir / "modeling_master_with_sequences.tsv", sep="\t")
    feature_tables = pd.read_csv(processed_dir / "parallel_label_feature_tables.tsv", sep="\t")

    rows = []
    for item in feature_tables.itertuples(index=False):
        label_id = item.label_id
        consensus = pd.read_csv(processed_dir / f"robust_consensus_{label_id}.tsv", sep="\t")
        labels = consensus[["gene_id", "stability_consensus_median"]].rename(
            columns={"stability_consensus_median": "target_label"}
        )
        table = base.drop(columns=["target_label"], errors="ignore").merge(labels, on="gene_id", how="inner")
        table["target_feature_type"] = item.feature_type
        table["target_label_key"] = item.label_key
        table["target_label_id"] = label_id
        out = processed_dir / f"parallel_modeling_master_with_sequences_{label_id}.tsv"
        table.to_csv(out, sep="\t", index=False)
        rows.append(
            {
                "label_id": label_id,
                "feature_type": item.feature_type,
                "label_key": item.label_key,
                "rows": int(len(table)),
                "table_path": str(out),
                "feature_path": item.feature_path,
            }
        )
    pd.DataFrame(rows).to_csv(processed_dir / "parallel_deep_sequence_tables.tsv", sep="\t", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
