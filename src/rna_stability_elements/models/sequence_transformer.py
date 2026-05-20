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
from rna_stability_elements.models.saluki_like import make_region_ids


def evaluate_sequence_transformer(
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
    region_lengths: RegionLengths = RegionLengths(utr5=256, cds=1024, utr3=1024),
    batch_size: int = 48,
    max_epochs: int = 20,
    patience: int = 5,
    learning_rate: float = 3e-4,
    weight_decay: float = 5e-4,
    embedding_dim: int = 8,
    region_embedding_dim: int = 4,
    model_dim: int = 128,
    conv_pool_size: int = 4,
    transformer_layers: int = 2,
    attention_heads: int = 4,
    feedforward_dim: int = 256,
    hidden_dim: int = 192,
    tabular_hidden_dim: int = 128,
    dropout: float = 0.25,
    token_dropout: float = 0.02,
    crop_strategy: str = "balanced",
    device: str = "cuda",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate a compact Conv-tokenized Transformer for RNA stability prediction."""
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
        metrics, predictions, history = train_sequence_transformer_split(
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
            model_dim=model_dim,
            conv_pool_size=conv_pool_size,
            transformer_layers=transformer_layers,
            attention_heads=attention_heads,
            feedforward_dim=feedforward_dim,
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


def write_sequence_transformer_evaluation(
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
    metrics, predictions, history = evaluate_sequence_transformer(
        table, feature_table=feature_table, **kwargs
    )
    for path, frame in [
        (metrics_out, metrics),
        (predictions_out, predictions),
        (history_out, history),
    ]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, sep="\t", index=False)
    return metrics, predictions, history


def train_sequence_transformer_split(
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
    model_dim: int,
    conv_pool_size: int,
    transformer_layers: int,
    attention_heads: int,
    feedforward_dim: int,
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

    model = build_sequence_transformer_model(
        region_lengths=region_lengths,
        embedding_dim=embedding_dim,
        region_embedding_dim=region_embedding_dim,
        model_dim=model_dim,
        conv_pool_size=conv_pool_size,
        transformer_layers=transformer_layers,
        attention_heads=attention_heads,
        feedforward_dim=feedforward_dim,
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

    y_pred_std = predict_sequence_transformer(model, test_loader, device=device)
    y_pred = y_pred_std * y_std + y_mean
    y_true = data.loc[split.test_index, target_column].to_numpy(dtype=np.float32)
    metric_row: dict[str, object] = regression_metrics(y_true, y_pred)
    metric_row.update(
        {
            "evaluation": split.evaluation,
            "split_name": split.split_name,
            "holdout_group": split.holdout_group,
            "repeat": split.repeat,
            "model": "conv_sequence_transformer",
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
            "model_dim": model_dim,
            "conv_pool_size": conv_pool_size,
            "conv_downsample_factor": conv_pool_size * conv_pool_size,
            "transformer_layers": transformer_layers,
            "attention_heads": attention_heads,
            "feedforward_dim": feedforward_dim,
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
    predictions["model"] = "conv_sequence_transformer"
    predictions["feature_set"] = metric_row["feature_set"]
    return metric_row, predictions, pd.DataFrame(history_rows)


def build_sequence_transformer_model(
    *,
    region_lengths: RegionLengths,
    embedding_dim: int,
    region_embedding_dim: int,
    model_dim: int,
    conv_pool_size: int,
    transformer_layers: int,
    attention_heads: int,
    feedforward_dim: int,
    hidden_dim: int,
    dropout: float,
    tabular_dim: int = 0,
    tabular_hidden_dim: int = 128,
):
    torch, nn = require_torch()
    if model_dim % attention_heads != 0:
        raise ValueError("model_dim must be divisible by attention_heads.")

    class ConvSequenceTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(5, embedding_dim, padding_idx=0)
            self.region_embedding = nn.Embedding(4, region_embedding_dim, padding_idx=0)
            region_ids = make_region_ids(region_lengths)
            self.register_buffer("region_ids", torch.as_tensor(region_ids, dtype=torch.long), persistent=False)
            conv_input_dim = embedding_dim + region_embedding_dim
            self.tokenizer = nn.Sequential(
                nn.Conv1d(conv_input_dim, model_dim, kernel_size=9, padding=4),
                nn.BatchNorm1d(model_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(kernel_size=conv_pool_size, stride=conv_pool_size),
                nn.Conv1d(model_dim, model_dim, kernel_size=7, padding=6, dilation=2),
                nn.BatchNorm1d(model_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(kernel_size=conv_pool_size, stride=conv_pool_size),
            )
            reduced_length = max(1, total_length(region_lengths) // (conv_pool_size * conv_pool_size))
            self.position_embedding = nn.Parameter(torch.zeros(1, reduced_length + 1, model_dim))
            self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=attention_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
            self.attention = nn.Sequential(
                nn.Linear(model_dim, model_dim // 2),
                nn.Tanh(),
                nn.Linear(model_dim // 2, 1),
            )
            if tabular_dim > 0:
                self.tabular_encoder = nn.Sequential(
                    nn.Linear(tabular_dim, tabular_hidden_dim),
                    nn.LayerNorm(tabular_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                head_input_dim = model_dim * 2 + tabular_hidden_dim
            else:
                self.tabular_encoder = None
                head_input_dim = model_dim * 2
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
            x = self.tokenizer(x.transpose(1, 2)).transpose(1, 2)
            cls = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)
            x = x + self.position_embedding[:, : x.shape[1], :]
            x = self.transformer(x)
            cls_pooled = x[:, 0, :]
            token_values = x[:, 1:, :]
            attention_scores = self.attention(token_values).squeeze(-1)
            attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(-1)
            attention_pooled = torch.sum(token_values * attention_weights, dim=1)
            pooled = torch.cat([cls_pooled, attention_pooled], dim=1)
            if self.tabular_encoder is not None:
                pooled = torch.cat([pooled, self.tabular_encoder(regions["_numeric"])], dim=1)
            return self.head(pooled).squeeze(-1)

    return ConvSequenceTransformer()


def total_length(region_lengths: RegionLengths) -> int:
    return int(region_lengths.utr5 + region_lengths.cds + region_lengths.utr3)


def predict_sequence_transformer(model, loader, *, device: str) -> np.ndarray:
    torch, _ = require_torch()
    model.eval()
    predictions = []
    with torch.no_grad():
        for regions, _target in loader:
            regions = {name: value.to(device, non_blocking=True) for name, value in regions.items()}
            predictions.append(model(regions).detach().cpu().numpy())
    return np.concatenate(predictions)
