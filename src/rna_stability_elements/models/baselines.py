from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_baseline(
    features: pd.DataFrame,
    *,
    target_column: str,
    model_name: str = "ridge",
    group_column: str | None = "cell_line",
    leave_group: str | None = None,
    drop_columns: list[str] | None = None,
    random_state: int = 13,
) -> dict[str, Any]:
    """Train a numeric-feature baseline and return metrics."""
    if target_column not in features:
        raise ValueError(f"Missing target column: {target_column}")

    data = features.dropna(subset=[target_column]).copy()
    if data.empty:
        raise ValueError("No rows remain after dropping missing target values.")

    if leave_group is None and group_column and group_column in data:
        leave_group = str(sorted(data[group_column].astype(str).unique())[-1])

    if leave_group is not None and group_column:
        train_mask = data[group_column].astype(str) != str(leave_group)
        split_name = f"leave_{group_column}_{leave_group}"
    else:
        train_mask = pd.Series(np.arange(len(data)) % 5 != 0, index=data.index)
        split_name = "deterministic_80_20"

    test_mask = ~train_mask
    if test_mask.sum() == 0 or train_mask.sum() == 0:
        raise ValueError("Train/test split produced an empty partition.")

    ignore = set(drop_columns or [])
    ignore.update({target_column})
    if group_column:
        ignore.add(group_column)
    numeric_columns = [
        col
        for col in data.select_dtypes(include=[np.number]).columns
        if col not in ignore
    ]
    if not numeric_columns:
        raise ValueError("No numeric feature columns found.")

    model = _make_model(model_name, random_state=random_state)
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
            ("model", model),
        ]
    )

    x_train = data.loc[train_mask, numeric_columns]
    y_train = data.loc[train_mask, target_column]
    x_test = data.loc[test_mask, numeric_columns]
    y_test = data.loc[test_mask, target_column]
    pipeline.fit(x_train, y_train)
    prediction = pipeline.predict(x_test)

    metrics = regression_metrics(y_test.to_numpy(), prediction)
    metrics.update(
        {
            "model": model_name,
            "target_column": target_column,
            "split": split_name,
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "n_features": len(numeric_columns),
            "feature_columns": numeric_columns,
        }
    )
    return metrics


def save_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    serializable = {
        key: (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in metrics.items()
    }
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2, sort_keys=True)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(mean_squared_error(y_true, y_pred, squared=False))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    pearson = _safe_corr(y_true, y_pred)
    spearman = _safe_corr(
        pd.Series(y_true).rank(method="average").to_numpy(),
        pd.Series(y_pred).rank(method="average").to_numpy(),
    )
    return {"rmse": rmse, "mae": mae, "r2": r2, "pearson": pearson, "spearman": spearman}


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0:
        return float(np.corrcoef(x, y)[0, 1])
    return float("nan")


def _make_model(model_name: str, *, random_state: int):
    if model_name == "ridge":
        return Ridge(alpha=1.0)
    if model_name == "elasticnet":
        return ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20000, random_state=random_state)
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=400,
            max_features="sqrt",
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=random_state,
        )
    if model_name == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=(256, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=128,
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=random_state,
        )
    if model_name in {"xgboost", "xgboost_gpu", "xgboost_cpu"}:
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "xgboost is required for model_name='xgboost_gpu' or 'xgboost_cpu'."
            ) from exc
        device = "cpu" if model_name == "xgboost_cpu" else "cuda"
        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=600,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            tree_method="hist",
            device=device,
            n_jobs=-1,
            random_state=random_state,
        )
    raise ValueError(f"Unknown model_name={model_name!r}")
