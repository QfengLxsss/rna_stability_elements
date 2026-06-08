from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rna_stability_elements.models.evaluation import build_splits
from rna_stability_elements.models.sequence_cnn import make_train_val_indices


RANDOM_STATE = 13
N_REPEATS = 3
TEST_SIZE = 0.2
MIN_TEST_SAMPLES = 50
DEEP_MODELS = ("region_cnn", "sequence_transformer", "saluki_like")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    processed = root / "data/processed"
    metadata = pd.read_csv(processed / "parallel_label_feature_tables.tsv", sep="\t")
    cohort_rows = []
    audit_rows = []

    for item in metadata.itertuples(index=False):
        label_id = item.label_id
        feature_path = processed / f"parallel_sequence_model_features_{label_id}.tsv"
        sequence_path = processed / f"parallel_modeling_master_with_sequences_{label_id}.tsv"
        feature_data = pd.read_csv(
            feature_path,
            sep="\t",
            usecols=["gene_id", "gene_symbol", "chromosome", "target_label"],
        )
        sequence_data = pd.read_csv(
            sequence_path,
            sep="\t",
            usecols=["gene_id", "gene_symbol", "chromosome", "target_label"],
        )
        target_max_abs_diff = validate_shared_cohort(feature_data, sequence_data, label_id=label_id)
        cohort = sequence_data.copy()
        cohort.insert(0, "row_index", np.arange(len(cohort), dtype=int))
        cohort["label_id"] = label_id
        cohort["feature_type"] = item.feature_type
        cohort["label_key"] = item.label_key
        cohort_path = processed / f"fair_benchmark_cohort_{label_id}.tsv"
        cohort.to_csv(cohort_path, sep="\t", index=False)

        manifest = make_manifest(cohort)
        manifest["label_id"] = label_id
        manifest_path = processed / f"fair_benchmark_splits_{label_id}.tsv"
        manifest.to_csv(manifest_path, sep="\t", index=False)

        cohort_rows.append(
            {
                "label_id": label_id,
                "feature_type": item.feature_type,
                "label_key": item.label_key,
                "n_genes": len(cohort),
                "n_splits": manifest["split_name"].nunique(),
                "n_repeated_random": manifest.loc[
                    manifest["evaluation"] == "repeated_random", "split_name"
                ].nunique(),
                "n_chromosome_holdout": manifest.loc[
                    manifest["evaluation"] == "chromosome_holdout", "split_name"
                ].nunique(),
                "feature_sequence_target_max_abs_diff": target_max_abs_diff,
                "cohort_path": str(cohort_path),
                "split_manifest_path": str(manifest_path),
                "feature_path": str(feature_path),
                "sequence_path": str(sequence_path),
            }
        )
        audit_rows.extend(audit_existing_deep(processed, label_id=label_id, manifest=manifest))

    pd.DataFrame(cohort_rows).to_csv(
        processed / "fair_benchmark_cohort_summary.tsv", sep="\t", index=False
    )
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(processed / "fair_benchmark_deep_reuse_audit.tsv", sep="\t", index=False)
    print(pd.DataFrame(cohort_rows).to_string(index=False))
    print("\nDeep reuse audit:")
    print(audit.groupby(["label_id", "model"])["reusable"].all().to_string())


def validate_shared_cohort(
    feature_data: pd.DataFrame, sequence_data: pd.DataFrame, *, label_id: str
) -> float:
    if feature_data["gene_id"].duplicated().any() or sequence_data["gene_id"].duplicated().any():
        raise ValueError(f"Duplicate gene_id detected for {label_id}.")
    if not feature_data["gene_id"].equals(sequence_data["gene_id"]):
        raise ValueError(f"Feature and sequence cohort row order differs for {label_id}.")
    if not feature_data["chromosome"].equals(sequence_data["chromosome"]):
        raise ValueError(f"Feature and sequence chromosome differs for {label_id}.")
    feature_target = feature_data["target_label"].to_numpy(dtype=float)
    sequence_target = sequence_data["target_label"].to_numpy(dtype=float)
    max_abs_diff = float(np.max(np.abs(feature_target - sequence_target)))
    if not np.allclose(feature_target, sequence_target, rtol=1e-7, atol=1e-9):
        raise ValueError(f"Feature and sequence target_label differs for {label_id}: {max_abs_diff}")
    return max_abs_diff


def make_manifest(cohort: pd.DataFrame) -> pd.DataFrame:
    data = cohort.reset_index(drop=True)
    splits = build_splits(
        data,
        evaluations=["repeated_random", "chromosome_holdout"],
        n_repeats=N_REPEATS,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        chromosome_column="chromosome",
        min_test_samples=MIN_TEST_SAMPLES,
    )
    rows = []
    for split in splits:
        train_index, validation_index = make_train_val_indices(
            split.train_index, random_state=RANDOM_STATE
        )
        role_by_index = {
            **{int(index): "train" for index in train_index},
            **{int(index): "validation" for index in validation_index},
            **{int(index): "test" for index in split.test_index},
        }
        for item in data.itertuples(index=False):
            rows.append(
                {
                    "evaluation": split.evaluation,
                    "split_name": split.split_name,
                    "holdout_group": split.holdout_group,
                    "repeat": split.repeat,
                    "row_index": int(item.row_index),
                    "gene_id": item.gene_id,
                    "chromosome": item.chromosome,
                    "role": role_by_index[int(item.row_index)],
                }
            )
    return pd.DataFrame(rows)


def audit_existing_deep(
    processed: Path, *, label_id: str, manifest: pd.DataFrame
) -> list[dict[str, object]]:
    rows = []
    expected_splits = sorted(manifest["split_name"].unique())
    for model in DEEP_MODELS:
        predictions_path = processed / f"parallel_deep_gpu_full_{model}_{label_id}_predictions.tsv"
        metrics_path = processed / f"parallel_deep_gpu_full_{model}_{label_id}_metrics.tsv"
        if not predictions_path.exists() or not metrics_path.exists():
            rows.append(
                {
                    "label_id": label_id,
                    "model": model,
                    "split_name": "",
                    "reusable": False,
                    "reason": "missing_predictions_or_metrics",
                }
            )
            continue
        predictions = pd.read_csv(predictions_path, sep="\t")
        metrics = pd.read_csv(metrics_path, sep="\t")
        for split_name in expected_splits:
            expected = set(
                manifest.loc[
                    (manifest["split_name"] == split_name) & (manifest["role"] == "test"),
                    "gene_id",
                ]
            )
            observed = set(predictions.loc[predictions["split_name"] == split_name, "gene_id"])
            metric = metrics[metrics["split_name"] == split_name]
            reusable = expected == observed and len(metric) == 1
            rows.append(
                {
                    "label_id": label_id,
                    "model": model,
                    "split_name": split_name,
                    "evaluation": manifest.loc[
                        manifest["split_name"] == split_name, "evaluation"
                    ].iloc[0],
                    "n_expected_test": len(expected),
                    "n_observed_test": len(observed),
                    "missing_test_genes": len(expected - observed),
                    "extra_test_genes": len(observed - expected),
                    "metric_rows": len(metric),
                    "reusable": reusable,
                    "reason": "exact_match" if reusable else "split_mismatch",
                }
            )
    return rows


if __name__ == "__main__":
    main()
