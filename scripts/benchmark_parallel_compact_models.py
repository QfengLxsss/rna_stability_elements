from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rna_stability_elements.models.baselines import _safe_corr
from rna_stability_elements.models.evaluation import (
    Split,
    numeric_feature_columns,
    repeated_random_splits,
)


MODELS = ["ridge", "random_forest_light", "xgboost_light"]
QUICK_HOLDOUT_CHROMOSOMES = ["chr1", "chr2", "chr17", "chr7"]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    processed_dir = root / "data/processed"
    feature_tables = pd.read_csv(processed_dir / "parallel_label_feature_tables.tsv", sep="\t")

    all_metrics = []
    all_predictions = []
    for item in feature_tables.itertuples(index=False):
        label_id = item.label_id
        print(f"[compact-benchmark] {label_id}", flush=True)
        features = pd.read_csv(item.feature_path, sep="\t")
        metrics, predictions = evaluate_feature_table(
            features,
            label_id=label_id,
            feature_type=item.feature_type,
            label_key=item.label_key,
            target_column=item.target_column,
        )
        all_metrics.append(metrics)
        all_predictions.append(predictions)

    metrics = pd.concat(all_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    summary = summarize_metrics(metrics)

    metrics.to_csv(processed_dir / "parallel_compact_model_benchmark_metrics.tsv", sep="\t", index=False)
    predictions.to_csv(processed_dir / "parallel_compact_model_benchmark_predictions.tsv", sep="\t", index=False)
    summary.to_csv(processed_dir / "parallel_compact_model_benchmark_summary.tsv", sep="\t", index=False)
    print(summary.to_string(index=False))


def evaluate_feature_table(
    features: pd.DataFrame,
    *,
    label_id: str,
    feature_type: str,
    label_key: str,
    target_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = features.dropna(subset=["target_label"]).reset_index(drop=True)
    feature_columns = numeric_feature_columns(data, target_column="target_label")
    splits = repeated_random_splits(
        data,
        n_repeats=3,
        test_size=0.2,
        random_state=13,
    )
    splits.extend(quick_chromosome_holdouts(data, chromosomes=QUICK_HOLDOUT_CHROMOSOMES))
    metric_rows = []
    prediction_rows = []
    for model_name in MODELS:
        print(f"  - {model_name}", flush=True)
        for split in splits:
            model = make_model(model_name)
            pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False)),
                    ("model", model),
                ]
            )
            x_train = data.loc[split.train_index, feature_columns]
            y_train = data.loc[split.train_index, "target_label"].to_numpy()
            x_test = data.loc[split.test_index, feature_columns]
            y_test = data.loc[split.test_index, "target_label"].to_numpy()
            pipeline.fit(x_train, y_train)
            y_pred = pipeline.predict(x_test)
            row = regression_row(y_test, y_pred)
            row.update(
                {
                    "label_id": label_id,
                    "feature_type": feature_type,
                    "label_key": label_key,
                    "target_column": target_column,
                    "model": model_name,
                    "evaluation": split.evaluation,
                    "split_name": split.split_name,
                    "holdout_group": split.holdout_group,
                    "repeat": split.repeat,
                    "n_train": int(len(split.train_index)),
                    "n_test": int(len(split.test_index)),
                    "n_features": int(len(feature_columns)),
                }
            )
            metric_rows.append(row)
            prediction = data.loc[
                split.test_index,
                [c for c in ["gene_id", "gene_symbol", "chromosome"] if c in data],
            ].copy()
            prediction["label_id"] = label_id
            prediction["model"] = model_name
            prediction["evaluation"] = split.evaluation
            prediction["split_name"] = split.split_name
            prediction["y_true"] = y_test
            prediction["y_pred"] = y_pred
            prediction["residual"] = y_pred - y_test
            prediction_rows.append(prediction)
    return pd.DataFrame(metric_rows), pd.concat(prediction_rows, ignore_index=True)


def make_model(model_name: str):
    if model_name == "ridge":
        return Ridge(alpha=1.0)
    if model_name == "random_forest_light":
        return RandomForestRegressor(
            n_estimators=120,
            max_features="sqrt",
            min_samples_leaf=3,
            n_jobs=4,
            random_state=13,
        )
    if model_name == "xgboost_light":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("xgboost is required for xgboost_light.") from exc
        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=160,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            tree_method="hist",
            device="cpu",
            n_jobs=4,
            random_state=13,
        )
    raise ValueError(model_name)


def quick_chromosome_holdouts(data: pd.DataFrame, *, chromosomes: list[str]) -> list[Split]:
    splits = []
    values = data["chromosome"].astype(str)
    for chromosome in chromosomes:
        test_mask = values == chromosome
        if int(test_mask.sum()) < 50:
            continue
        splits.append(
            Split(
                evaluation="chromosome_holdout_quick",
                split_name=f"holdout_{chromosome}",
                train_index=data.index[~test_mask].to_numpy(),
                test_index=data.index[test_mask].to_numpy(),
                holdout_group=chromosome,
            )
        )
    return splits


def regression_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson": _safe_corr(y_true, y_pred),
        "spearman": _safe_corr(
            pd.Series(y_true).rank(method="average").to_numpy(),
            pd.Series(y_pred).rank(method="average").to_numpy(),
        ),
    }


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in metrics.groupby(["label_id", "feature_type", "label_key", "target_column", "model", "evaluation"]):
        label_id, feature_type, label_key, target_column, model, evaluation = keys
        row = {
            "label_id": label_id,
            "feature_type": feature_type,
            "label_key": label_key,
            "target_column": target_column,
            "model": model,
            "evaluation": evaluation,
            "n_splits": int(len(group)),
        }
        for metric in ["pearson", "spearman", "r2", "rmse", "mae"]:
            row[f"{metric}_median"] = float(group[metric].median())
            row[f"{metric}_mean"] = float(group[metric].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["evaluation", "label_id", "model"]).reset_index(drop=True)


if __name__ == "__main__":
    main()
