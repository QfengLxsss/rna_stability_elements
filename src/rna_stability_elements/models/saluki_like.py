from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from rna_stability_elements.models.baselines import regression_metrics
from rna_stability_elements.models.evaluation import Split, build_splits
from rna_stability_elements.models.multimodal import require_torch
from rna_stability_elements.models.sequence_cnn import (
    RegionLengths,
    RegionSequenceDataset,
    align_numeric_features,
    encode_regions,
    make_train_val_indices,
    preprocess_numeric_features,
    run_epoch,
)


def evaluate_saluki_like(
    table: pd.DataFrame,
    *,
    feature_table: pd.DataFrame | None = None,
    target_column: str = "target_label",
    evaluations: Iterable[str] = ("repeated_random",),
    n_repeats: int = 1,
    test_size: float = 0.2,
    random_state: int = 13,
    chromosome_column: str = "chromosome",
    min_test_samples: int = 50,
    region_lengths: RegionLengths = RegionLengths(utr5=512, cds=2048, utr3=2048),
    batch_size: int = 48,
    max_epochs: int = 25,
    patience: int = 5,
    learning_rate: float = 5e-4,
    weight_decay: float = 5e-4,
    embedding_dim: int = 8,
    region_embedding_dim: int = 4,
    channels: int = 96,
    conv_pool_size: int = 4,
    gru_hidden_dim: int = 96,
    gru_layers: int = 1,
    hidden_dim: int = 192,
    tabular_hidden_dim: int = 128,
    dropout: float = 0.35,
    token_dropout: float = 0.02,
    crop_strategy: str = "balanced",
    device: str = "cuda",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate a compact Saluki-like CNN+GRU RNA sequence model."""
    if target_column not in table:
        raise ValueError(f"Missing target column: {target_column}")
    data = table.dropna(subset=[target_column]).reset_index(drop=True).copy()
    if data.empty:
        raise ValueError("No rows remain after dropping missing target values.")

    encoded = encode_regions(
        data,
        region_lengths=region_lengths,
        crop_strategy=crop_strategy,
        random_state=random_state,
    )
    numeric_features, numeric_feature_names = align_numeric_features(
        data, feature_table=feature_table, target_column=target_column
    )
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
    prediction_frames = []
    history_frames = []
    for split in splits:
        metrics, predictions, history = train_saluki_like_split(
            data,
            encoded,
            split=split,
            target_column=target_column,
            region_lengths=region_lengths,
            numeric_features=numeric_features,
            numeric_feature_names=numeric_feature_names,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            embedding_dim=embedding_dim,
            region_embedding_dim=region_embedding_dim,
            channels=channels,
            conv_pool_size=conv_pool_size,
            gru_hidden_dim=gru_hidden_dim,
            gru_layers=gru_layers,
            hidden_dim=hidden_dim,
            tabular_hidden_dim=tabular_hidden_dim,
            dropout=dropout,
            token_dropout=token_dropout,
            crop_strategy=crop_strategy,
            random_state=random_state,
            device=device,
        )
        metrics_rows.append(metrics)
        prediction_frames.append(predictions)
        history_frames.append(history)

    return (
        pd.DataFrame(metrics_rows),
        pd.concat(prediction_frames, ignore_index=True),
        pd.concat(history_frames, ignore_index=True),
    )


def write_saluki_like_evaluation(
    table_path: str | Path,
    *,
    metrics_out: str | Path,
    predictions_out: str | Path,
    history_out: str | Path,
    feature_table_path: str | Path | None = None,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    table = pd.read_csv(table_path, sep="\t")
    feature_table = pd.read_csv(feature_table_path, sep="\t") if feature_table_path else None
    metrics, predictions, history = evaluate_saluki_like(table, feature_table=feature_table, **kwargs)
    for path, frame in [
        (metrics_out, metrics),
        (predictions_out, predictions),
        (history_out, history),
    ]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, sep="\t", index=False)
    return metrics, predictions, history


def train_saluki_like_split(
    data: pd.DataFrame,
    encoded: dict[str, np.ndarray],
    *,
    split: Split,
    target_column: str,
    region_lengths: RegionLengths,
    numeric_features: np.ndarray | None,
    numeric_feature_names: list[str],
    batch_size: int,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    embedding_dim: int,
    region_embedding_dim: int,
    channels: int,
    conv_pool_size: int,
    gru_hidden_dim: int,
    gru_layers: int,
    hidden_dim: int,
    tabular_hidden_dim: int,
    dropout: float,
    token_dropout: float,
    crop_strategy: str,
    random_state: int,
    device: str,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    torch, nn = require_torch()
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    torch.manual_seed(random_state)
    if device == "cuda":
        torch.cuda.manual_seed_all(random_state)

    train_index, val_index = make_train_val_indices(split.train_index, random_state=random_state)
    y_train_raw = data.loc[train_index, target_column].to_numpy(dtype=np.float32)
    y_mean = float(y_train_raw.mean())
    y_std = float(y_train_raw.std()) or 1.0
    numeric_processed = preprocess_numeric_features(
        numeric_features,
        train_index=train_index,
        val_index=val_index,
        test_index=split.test_index,
    )

    train_dataset = RegionSequenceDataset(
        encoded,
        data[target_column],
        train_index,
        y_mean,
        y_std,
        numeric_processed,
        token_dropout=token_dropout,
        training=True,
    )
    val_dataset = RegionSequenceDataset(
        encoded, data[target_column], val_index, y_mean, y_std, numeric_processed
    )
    test_dataset = RegionSequenceDataset(
        encoded, data[target_column], split.test_index, y_mean, y_std, numeric_processed
    )
    pin_memory = device == "cuda"
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory
    )

    model = build_saluki_like_model(
        region_lengths=region_lengths,
        embedding_dim=embedding_dim,
        region_embedding_dim=region_embedding_dim,
        channels=channels,
        conv_pool_size=conv_pool_size,
        gru_hidden_dim=gru_hidden_dim,
        gru_layers=gru_layers,
        hidden_dim=hidden_dim,
        tabular_dim=0 if numeric_processed is None else len(numeric_feature_names),
        tabular_hidden_dim=tabular_hidden_dim,
        dropout=dropout,
    ).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_state = None
    best_val_loss = float("inf")
    stale_epochs = 0
    history_rows = []
    for epoch in range(1, max_epochs + 1):
        train_loss = run_epoch(
            model, train_loader, criterion=criterion, optimizer=optimizer, device=device
        )
        val_loss = run_epoch(model, val_loader, criterion=criterion, optimizer=None, device=device)
        history_rows.append(
            {
                "evaluation": split.evaluation,
                "split_name": split.split_name,
                "holdout_group": split.holdout_group,
                "repeat": split.repeat,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "device": device,
            }
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    y_pred_std = predict_saluki_like(model, test_loader, device=device)
    y_pred = y_pred_std * y_std + y_mean
    y_true = data.loc[split.test_index, target_column].to_numpy(dtype=np.float32)
    metric_row: dict[str, object] = regression_metrics(y_true, y_pred)
    metric_row.update(
        {
            "evaluation": split.evaluation,
            "split_name": split.split_name,
            "holdout_group": split.holdout_group,
            "repeat": split.repeat,
            "model": "saluki_like_cnn_gru",
            "feature_set": "raw_5utr_cds_3utr_plus_tabular" if numeric_processed is not None else "raw_5utr_cds_3utr",
            "n_train": int(len(train_index)),
            "n_validation": int(len(val_index)),
            "n_test": int(len(split.test_index)),
            "max_length_5utr": region_lengths.utr5,
            "max_length_cds": region_lengths.cds,
            "max_length_3utr": region_lengths.utr3,
            "n_tabular_features": len(numeric_feature_names),
            "token_dropout": token_dropout,
            "crop_strategy": crop_strategy,
            "embedding_dim": embedding_dim,
            "region_embedding_dim": region_embedding_dim,
            "channels": channels,
            "conv_pool_size": conv_pool_size,
            "conv_downsample_factor": conv_pool_size * conv_pool_size,
            "gru_hidden_dim": gru_hidden_dim,
            "gru_layers": gru_layers,
            "dropout": dropout,
            "best_val_loss": best_val_loss,
            "epochs_trained": len(history_rows),
            "device": device,
        }
    )

    prediction_columns = [
        column
        for column in ["gene_id", "gene_symbol", "chromosome", "replicate_qc_flag"]
        if column in data.columns
    ]
    predictions = data.loc[split.test_index, prediction_columns].copy()
    predictions["y_true"] = y_true
    predictions["y_pred"] = y_pred
    predictions["residual"] = predictions["y_pred"] - predictions["y_true"]
    predictions["evaluation"] = split.evaluation
    predictions["split_name"] = split.split_name
    predictions["holdout_group"] = split.holdout_group
    predictions["repeat"] = split.repeat
    predictions["model"] = "saluki_like_cnn_gru"
    predictions["feature_set"] = metric_row["feature_set"]
    return metric_row, predictions, pd.DataFrame(history_rows)


def build_saluki_like_model(
    *,
    region_lengths: RegionLengths,
    embedding_dim: int,
    region_embedding_dim: int,
    channels: int,
    conv_pool_size: int,
    gru_hidden_dim: int,
    gru_layers: int,
    hidden_dim: int,
    dropout: float,
    tabular_dim: int = 0,
    tabular_hidden_dim: int = 128,
):
    torch, nn = require_torch()

    class SalukiLikeCNNGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.region_lengths = region_lengths
            self.embedding = nn.Embedding(5, embedding_dim, padding_idx=0)
            self.region_embedding = nn.Embedding(4, region_embedding_dim, padding_idx=0)
            region_ids = make_region_ids(region_lengths)
            self.register_buffer("region_ids", torch.as_tensor(region_ids, dtype=torch.long), persistent=False)
            conv_input_dim = embedding_dim + region_embedding_dim
            self.convolution = nn.Sequential(
                nn.Conv1d(conv_input_dim, channels, kernel_size=9, padding=4),
                nn.BatchNorm1d(channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(kernel_size=conv_pool_size, stride=conv_pool_size),
                nn.Conv1d(channels, channels, kernel_size=9, padding=8, dilation=2),
                nn.BatchNorm1d(channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(kernel_size=conv_pool_size, stride=conv_pool_size),
                nn.Conv1d(channels, channels, kernel_size=5, padding=4, dilation=2),
                nn.GELU(),
            )
            self.gru = nn.GRU(
                input_size=channels,
                hidden_size=gru_hidden_dim,
                num_layers=gru_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if gru_layers > 1 else 0.0,
            )
            self.attention = nn.Sequential(
                nn.Linear(gru_hidden_dim * 2, gru_hidden_dim),
                nn.Tanh(),
                nn.Linear(gru_hidden_dim, 1),
            )
            if tabular_dim > 0:
                self.tabular_encoder = nn.Sequential(
                    nn.Linear(tabular_dim, tabular_hidden_dim),
                    nn.LayerNorm(tabular_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                head_input_dim = gru_hidden_dim * 2 + tabular_hidden_dim
            else:
                self.tabular_encoder = None
                head_input_dim = gru_hidden_dim * 2
            self.head = nn.Sequential(
                nn.Linear(head_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )

        def forward(self, regions):
            tokens = torch.cat([regions["5utr"], regions["cds"], regions["3utr"]], dim=1)
            region_ids = self.region_ids.unsqueeze(0).expand(tokens.shape[0], -1)
            x = torch.cat([self.embedding(tokens), self.region_embedding(region_ids)], dim=-1)
            x = self.convolution(x.transpose(1, 2)).transpose(1, 2)
            x, _hidden = self.gru(x)
            attention_scores = self.attention(x).squeeze(-1)
            attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(-1)
            pooled = torch.sum(x * attention_weights, dim=1)
            if self.tabular_encoder is not None:
                pooled = torch.cat([pooled, self.tabular_encoder(regions["_numeric"])], dim=1)
            return self.head(pooled).squeeze(-1)

    return SalukiLikeCNNGRU()


def make_region_ids(region_lengths: RegionLengths) -> np.ndarray:
    """Return fixed region ids aligned to concatenated 5'UTR/CDS/3'UTR token tensors."""
    return np.concatenate(
        [
            np.full(region_lengths.utr5, 1, dtype=np.int64),
            np.full(region_lengths.cds, 2, dtype=np.int64),
            np.full(region_lengths.utr3, 3, dtype=np.int64),
        ]
    )


def predict_saluki_like(model, loader, *, device: str) -> np.ndarray:
    torch, _ = require_torch()
    model.eval()
    predictions = []
    with torch.no_grad():
        for regions, _target in loader:
            regions = {name: value.to(device, non_blocking=True) for name, value in regions.items()}
            predictions.append(model(regions).detach().cpu().numpy())
    return np.concatenate(predictions)
