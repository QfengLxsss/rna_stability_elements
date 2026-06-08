from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rna_stability_elements.models.baselines import _make_model, regression_metrics


METADATA_COLUMNS = {
    "gene_id",
    "gene_symbol",
    "canonical_transcript_id",
    "chromosome",
    "strand",
    "gene_biotype",
    "transcript_biotype",
    "sequence_status",
    "replicate_qc_flag",
}


@dataclass(frozen=True)
class Split:
    evaluation: str
    split_name: str
    train_index: np.ndarray
    test_index: np.ndarray
    holdout_group: str = ""
    repeat: int = -1


def evaluate_sequence_models(
    features: pd.DataFrame,
    *,
    target_column: str = "target_label",
    models: Iterable[str] = ("elasticnet", "random_forest"),
    feature_sets: Iterable[str] = ("all",),
    evaluations: Iterable[str] = ("repeated_random", "chromosome_holdout"),
    n_repeats: int = 5,
    test_size: float = 0.2,
    random_state: int = 13,
    chromosome_column: str = "chromosome",
    min_test_samples: int = 50,
    preprocessing: str = "standard",
    pca_components: int = 128,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run strict sequence-model evaluation and return metrics, predictions, summary, and importances."""
    if target_column not in features:
        raise ValueError(f"Missing target column: {target_column}")
    data = features.dropna(subset=[target_column]).reset_index(drop=True).copy()
    if data.empty:
        raise ValueError("No rows remain after dropping missing target values.")

    numeric_columns = numeric_feature_columns(data, target_column=target_column)
    groups = feature_groups(numeric_columns)
    selected_feature_sets = list(feature_sets)
    selected_models = list(models)
    splits = build_splits(
        data,
        evaluations=evaluations,
        n_repeats=n_repeats,
        test_size=test_size,
        random_state=random_state,
        chromosome_column=chromosome_column,
        min_test_samples=min_test_samples,
    )
    if not splits:
        raise ValueError("No valid evaluation splits were produced.")

    metrics_rows = []
    prediction_rows = []
    importance_rows = []
    for feature_set in selected_feature_sets:
        selected_columns = resolve_feature_set(feature_set, numeric_columns, groups)
        if not selected_columns:
            raise ValueError(f"Feature set {feature_set!r} resolved to zero columns.")
        for model_name in selected_models:
            for split in splits:
                metric_row, predictions, importances = evaluate_one_split(
                    data,
                    split=split,
                    feature_columns=selected_columns,
                    target_column=target_column,
                    model_name=model_name,
                    feature_set=feature_set,
                    random_state=random_state,
                    preprocessing=preprocessing,
                    pca_components=pca_components,
                )
                metrics_rows.append(metric_row)
                prediction_rows.append(predictions)
                if not importances.empty:
                    importance_rows.append(importances)

    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    importances = pd.concat(importance_rows, ignore_index=True) if importance_rows else pd.DataFrame()
    summary = summarize_evaluation_metrics(metrics)
    return metrics, predictions, summary, importances


def write_sequence_model_evaluation(
    features_path: str | Path,
    *,
    metrics_out: str | Path,
    predictions_out: str | Path,
    summary_out: str | Path,
    importance_out: str | Path | None = None,
    target_column: str = "target_label",
    models: Iterable[str] = ("elasticnet", "random_forest"),
    feature_sets: Iterable[str] = ("all",),
    evaluations: Iterable[str] = ("repeated_random", "chromosome_holdout"),
    n_repeats: int = 5,
    test_size: float = 0.2,
    random_state: int = 13,
    chromosome_column: str = "chromosome",
    min_test_samples: int = 50,
    preprocessing: str = "standard",
    pca_components: int = 128,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(features_path, sep="\t")
    metrics, predictions, summary, importances = evaluate_sequence_models(
        features,
        target_column=target_column,
        models=models,
        feature_sets=feature_sets,
        evaluations=evaluations,
        n_repeats=n_repeats,
        test_size=test_size,
        random_state=random_state,
        chromosome_column=chromosome_column,
        min_test_samples=min_test_samples,
        preprocessing=preprocessing,
        pca_components=pca_components,
    )
    for path, frame in [
        (metrics_out, metrics),
        (predictions_out, predictions),
        (summary_out, summary),
    ]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, sep="\t", index=False)
    if importance_out is not None:
        Path(importance_out).parent.mkdir(parents=True, exist_ok=True)
        importances.to_csv(importance_out, sep="\t", index=False)
    return metrics, predictions, summary, importances


def numeric_feature_columns(data: pd.DataFrame, *, target_column: str) -> list[str]:
    ignored = set(METADATA_COLUMNS)
    ignored.add(target_column)
    return [
        column
        for column in data.select_dtypes(include=[np.number]).columns
        if column not in ignored
    ]


def feature_groups(numeric_columns: Iterable[str]) -> dict[str, list[str]]:
    columns = list(numeric_columns)
    groups = {
        "all": columns,
        "length": [column for column in columns if column.endswith("_length")],
        "composition": [
            column
            for column in columns
            if column.endswith("_gc_fraction")
            or column.endswith("_au_fraction")
            or column.endswith("_u_fraction")
        ],
        "motif": [column for column in columns if "_motif_" in column],
        "kmer": [column for column in columns if "_kmer_" in column],
        "kmer3": [column for column in columns if _is_kmer_column(column, 3)],
        "kmer4": [column for column in columns if _is_kmer_column(column, 4)],
        "full_region": [column for column in columns if column.startswith("full_")],
        "5utr_region": [column for column in columns if column.startswith("5utr_")],
        "cds_region": [column for column in columns if column.startswith("cds_")],
        "3utr_region": [column for column in columns if column.startswith("3utr_")],
        "lm": [column for column in columns if column.startswith("lm_") and "_emb_" in column],
        "lm_5utr": [column for column in columns if column.startswith("lm_5utr_emb_")],
        "lm_cds": [column for column in columns if column.startswith("lm_cds_emb_")],
        "lm_3utr": [column for column in columns if column.startswith("lm_3utr_emb_")],
    }
    return groups


def input_ablation_feature_sets(numeric_columns: Iterable[str]) -> dict[str, list[str]]:
    """Build interpretable feature sets for fixed-split input-information ablation."""
    columns = list(numeric_columns)
    groups = feature_groups(columns)

    def union(*names: str) -> list[str]:
        selected = set().union(*(groups[name] for name in names))
        return [column for column in columns if column in selected]

    structured = union("5utr_region", "cds_region", "3utr_region")
    return {
        "all_regions": columns,
        "full_only": groups["full_region"],
        "structured_regions": structured,
        "5utr_only": groups["5utr_region"],
        "cds_only": groups["cds_region"],
        "3utr_only": groups["3utr_region"],
        "utr_only": union("5utr_region", "3utr_region"),
        "structured_no_5utr": [
            column for column in structured if not column.startswith("5utr_")
        ],
        "structured_no_cds": [
            column for column in structured if not column.startswith("cds_")
        ],
        "structured_no_3utr": [
            column for column in structured if not column.startswith("3utr_")
        ],
        "simple_length_composition": union("length", "composition"),
        "length_only": groups["length"],
        "composition_only": groups["composition"],
        "motif_only": groups["motif"],
        "kmer3_only": groups["kmer3"],
        "kmer4_only": groups["kmer4"],
    }


def resolve_feature_set(feature_set: str, numeric_columns: list[str], groups: dict[str, list[str]]) -> list[str]:
    if feature_set in groups:
        return groups[feature_set]
    if feature_set.endswith("_only"):
        key = feature_set[: -len("_only")]
        return groups.get(key, [])
    if feature_set.startswith("no_"):
        key = feature_set[len("no_") :]
        drop = set(groups.get(key, []))
        return [column for column in numeric_columns if column not in drop]
    raise ValueError(
        f"Unknown feature set {feature_set!r}. Known sets: {sorted(groups)} plus *_only and no_*."
    )


def build_splits(
    data: pd.DataFrame,
    *,
    evaluations: Iterable[str],
    n_repeats: int,
    test_size: float,
    random_state: int,
    chromosome_column: str,
    min_test_samples: int,
) -> list[Split]:
    splits: list[Split] = []
    for evaluation in evaluations:
        if evaluation == "repeated_random":
            splits.extend(
                repeated_random_splits(
                    data,
                    n_repeats=n_repeats,
                    test_size=test_size,
                    random_state=random_state,
                )
            )
        elif evaluation == "chromosome_holdout":
            splits.extend(
                chromosome_holdout_splits(
                    data,
                    chromosome_column=chromosome_column,
                    min_test_samples=min_test_samples,
                )
            )
        else:
            raise ValueError(f"Unknown evaluation={evaluation!r}")
    return splits


def repeated_random_splits(
    data: pd.DataFrame,
    *,
    n_repeats: int = 5,
    test_size: float = 0.2,
    random_state: int = 13,
) -> list[Split]:
    n_rows = len(data)
    n_test = max(1, int(round(n_rows * test_size)))
    splits = []
    for repeat in range(n_repeats):
        rng = np.random.default_rng(random_state + repeat)
        indices = rng.permutation(n_rows)
        test_index = np.sort(indices[:n_test])
        train_index = np.sort(indices[n_test:])
        splits.append(
            Split(
                evaluation="repeated_random",
                split_name=f"random_repeat_{repeat}",
                train_index=train_index,
                test_index=test_index,
                repeat=repeat,
            )
        )
    return splits


def chromosome_holdout_splits(
    data: pd.DataFrame,
    *,
    chromosome_column: str = "chromosome",
    min_test_samples: int = 50,
) -> list[Split]:
    if chromosome_column not in data:
        raise ValueError(f"Missing chromosome column: {chromosome_column}")
    splits = []
    chromosomes = sorted(data[chromosome_column].dropna().astype(str).unique(), key=_chromosome_sort_key)
    for chromosome in chromosomes:
        test_mask = data[chromosome_column].astype(str) == chromosome
        n_test = int(test_mask.sum())
        if n_test < min_test_samples:
            continue
        test_index = data.index[test_mask].to_numpy()
        train_index = data.index[~test_mask].to_numpy()
        splits.append(
            Split(
                evaluation="chromosome_holdout",
                split_name=f"holdout_{chromosome}",
                train_index=train_index,
                test_index=test_index,
                holdout_group=chromosome,
            )
        )
    return splits


def evaluate_one_split(
    data: pd.DataFrame,
    *,
    split: Split,
    feature_columns: list[str],
    target_column: str,
    model_name: str,
    feature_set: str,
    random_state: int,
    preprocessing: str = "standard",
    pca_components: int = 128,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    pipeline = make_evaluation_pipeline(
        feature_columns,
        model_name=model_name,
        random_state=random_state,
        preprocessing=preprocessing,
        pca_components=pca_components,
    )
    x_train = data.loc[split.train_index, feature_columns]
    y_train = data.loc[split.train_index, target_column]
    x_test = data.loc[split.test_index, feature_columns]
    y_test = data.loc[split.test_index, target_column]
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)

    metric_row: dict[str, object] = regression_metrics(y_test.to_numpy(), y_pred)
    metric_row.update(
        {
            "evaluation": split.evaluation,
            "split_name": split.split_name,
            "holdout_group": split.holdout_group,
            "repeat": split.repeat,
            "model": model_name,
            "feature_set": feature_set,
            "n_train": int(len(split.train_index)),
            "n_test": int(len(split.test_index)),
            "n_features": int(len(feature_columns)),
            "preprocessing": preprocessing,
        }
    )

    prediction_columns = [
        column
        for column in ["gene_id", "gene_symbol", "chromosome", "replicate_qc_flag"]
        if column in data.columns
    ]
    predictions = data.loc[split.test_index, prediction_columns].copy()
    predictions["y_true"] = y_test.to_numpy()
    predictions["y_pred"] = y_pred
    predictions["residual"] = predictions["y_pred"] - predictions["y_true"]
    predictions["evaluation"] = split.evaluation
    predictions["split_name"] = split.split_name
    predictions["holdout_group"] = split.holdout_group
    predictions["repeat"] = split.repeat
    predictions["model"] = model_name
    predictions["feature_set"] = feature_set
    predictions["preprocessing"] = preprocessing
    importances = model_feature_importance(pipeline, feature_columns)
    if not importances.empty:
        importances["evaluation"] = split.evaluation
        importances["split_name"] = split.split_name
        importances["holdout_group"] = split.holdout_group
        importances["repeat"] = split.repeat
        importances["model"] = model_name
        importances["feature_set"] = feature_set
        importances["preprocessing"] = preprocessing
    return metric_row, predictions, importances


def make_evaluation_pipeline(
    feature_columns: list[str],
    *,
    model_name: str,
    random_state: int,
    preprocessing: str,
    pca_components: int,
) -> Pipeline:
    model = _make_model(model_name, random_state=random_state)
    if preprocessing == "standard":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
                ("model", model),
            ]
        )
    if preprocessing == "pca":
        n_components = min(pca_components, len(feature_columns))
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_components, random_state=random_state)),
                ("model", model),
            ]
        )
    if preprocessing == "region_pca":
        transformers = []
        used_columns: set[str] = set()
        for region, prefix in [
            ("lm_5utr", "lm_5utr_emb_"),
            ("lm_cds", "lm_cds_emb_"),
            ("lm_3utr", "lm_3utr_emb_"),
        ]:
            region_columns = [column for column in feature_columns if column.startswith(prefix)]
            if region_columns:
                used_columns.update(region_columns)
                transformers.append(
                    (
                        region,
                        Pipeline(
                            steps=[
                                ("imputer", SimpleImputer(strategy="median")),
                                ("scaler", StandardScaler()),
                                (
                                    "pca",
                                    PCA(
                                        n_components=min(pca_components, len(region_columns)),
                                        random_state=random_state,
                                    ),
                                ),
                            ]
                        ),
                        region_columns,
                    )
                )
        remainder_columns = [column for column in feature_columns if column not in used_columns]
        if remainder_columns:
            transformers.append(
                (
                    "other_features",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    remainder_columns,
                )
            )
        return Pipeline(
            steps=[
                ("preprocess", ColumnTransformer(transformers=transformers)),
                ("model", model),
            ]
        )
    raise ValueError("preprocessing must be one of: standard, pca, region_pca")


def model_feature_importance(pipeline: Pipeline, feature_columns: list[str]) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    if hasattr(model, "coef_"):
        values = np.asarray(model.coef_).reshape(-1)
        importance_kind = "coefficient"
    elif hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_).reshape(-1)
        importance_kind = "feature_importance"
    else:
        return pd.DataFrame()
    if len(values) != len(feature_columns):
        return pd.DataFrame()
    frame = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": values.astype(float),
            "importance_abs": np.abs(values.astype(float)),
            "importance_kind": importance_kind,
        }
    )
    frame["feature_group"] = frame["feature"].map(feature_group_name)
    return frame.sort_values("importance_abs", ascending=False).reset_index(drop=True)


def feature_group_name(feature: str) -> str:
    if feature.endswith("_length"):
        return "length"
    if feature.endswith("_gc_fraction") or feature.endswith("_au_fraction") or feature.endswith("_u_fraction"):
        return "composition"
    if "_motif_" in feature:
        return "motif"
    if "_kmer_" in feature:
        return "kmer"
    if feature.startswith("full_"):
        return "full_region"
    if feature.startswith("5utr_"):
        return "5utr_region"
    if feature.startswith("cds_"):
        return "cds_region"
    if feature.startswith("3utr_"):
        return "3utr_region"
    if feature.startswith("lm_5utr_emb_"):
        return "lm_5utr"
    if feature.startswith("lm_cds_emb_"):
        return "lm_cds"
    if feature.startswith("lm_3utr_emb_"):
        return "lm_3utr"
    if feature.startswith("lm_") and "_emb_" in feature:
        return "lm"
    return "other"


def summarize_evaluation_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["evaluation", "model", "feature_set"]
    value_columns = ["rmse", "mae", "r2", "pearson", "spearman"]
    rows = []
    for keys, group in metrics.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys))
        row["n_splits"] = int(len(group))
        if "n_features" in group:
            row["n_features_median"] = float(group["n_features"].median())
        elif "n_tabular_features" in group:
            row["n_features_median"] = float(group["n_tabular_features"].median())
        else:
            row["n_features_median"] = float("nan")
        for column in value_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=0))
            row[f"{column}_median"] = float(group[column].median())
            row[f"{column}_min"] = float(group[column].min())
            row[f"{column}_max"] = float(group[column].max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["evaluation", "model", "feature_set"]).reset_index(drop=True)


def _is_kmer_column(column: str, k: int) -> bool:
    if "_kmer_" not in column:
        return False
    kmer = column.rsplit("_kmer_", 1)[-1]
    return len(kmer) == k


def _chromosome_sort_key(chromosome: str) -> tuple[int, str]:
    label = chromosome.replace("chr", "")
    if label.isdigit():
        return int(label), label
    special = {"X": 23, "Y": 24, "M": 25, "MT": 25}
    return special.get(label, 99), label
