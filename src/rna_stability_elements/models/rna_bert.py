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
    align_numeric_features,
    encode_regions,
    make_train_val_indices,
    preprocess_numeric_features,
    run_epoch,
)


def evaluate_rna_bert(
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
    kmer_size: int = 4,
    kmer_stride: int = 4,
    batch_size: int = 32,
    max_epochs: int = 20,
    patience: int = 5,
    learning_rate: float = 3e-4,
    weight_decay: float = 5e-4,
    model_dim: int = 128,
    region_embedding_dim: int = 8,
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
    """Evaluate a DNABERT-style k-mer RNA Transformer encoder."""
    if target_column not in table:
        raise ValueError(f"Missing target column: {target_column}")
    data = table.dropna(subset=[target_column]).reset_index(drop=True).copy()
    if data.empty:
        raise ValueError("No rows remain after dropping missing target values.")

    encoded_regions = encode_regions(
        data,
        region_lengths=region_lengths,
        crop_strategy=crop_strategy,
        random_state=random_state,
    )
    encoded_kmers = encode_kmer_regions(
        encoded_regions,
        region_lengths=region_lengths,
        kmer_size=kmer_size,
        kmer_stride=kmer_stride,
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
        metrics, predictions, history = train_rna_bert_split(
            data,
            encoded_kmers,
            split=split,
            target_column=target_column,
            region_lengths=region_lengths,
            numeric_features=numeric_features,
            numeric_feature_names=numeric_feature_names,
            kmer_size=kmer_size,
            kmer_stride=kmer_stride,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            model_dim=model_dim,
            region_embedding_dim=region_embedding_dim,
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


def write_rna_bert_evaluation(
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
    metrics, predictions, history = evaluate_rna_bert(table, feature_table=feature_table, **kwargs)
    for path, frame in [
        (metrics_out, metrics),
        (predictions_out, predictions),
        (history_out, history),
    ]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, sep="\t", index=False)
    return metrics, predictions, history


def train_rna_bert_split(
    data: pd.DataFrame,
    encoded_kmers: dict[str, np.ndarray],
    *,
    split: Split,
    target_column: str,
    region_lengths: RegionLengths,
    numeric_features: np.ndarray | None,
    numeric_feature_names: list[str],
    kmer_size: int,
    kmer_stride: int,
    batch_size: int,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    model_dim: int,
    region_embedding_dim: int,
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

    train_dataset = KmerSequenceDataset(
        encoded_kmers,
        data[target_column],
        train_index,
        y_mean,
        y_std,
        numeric_processed,
        token_dropout=token_dropout,
        training=True,
    )
    val_dataset = KmerSequenceDataset(
        encoded_kmers, data[target_column], val_index, y_mean, y_std, numeric_processed
    )
    test_dataset = KmerSequenceDataset(
        encoded_kmers, data[target_column], split.test_index, y_mean, y_std, numeric_processed
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

    model = build_rna_bert_model(
        n_tokens=encoded_kmers["tokens"].shape[1],
        vocab_size=2 + 4**kmer_size,
        model_dim=model_dim,
        region_embedding_dim=region_embedding_dim,
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

    y_pred_std = predict_rna_bert(model, test_loader, device=device)
    y_pred = y_pred_std * y_std + y_mean
    y_true = data.loc[split.test_index, target_column].to_numpy(dtype=np.float32)
    metric_row: dict[str, object] = regression_metrics(y_true, y_pred)
    metric_row.update(
        {
            "evaluation": split.evaluation,
            "split_name": split.split_name,
            "holdout_group": split.holdout_group,
            "repeat": split.repeat,
            "model": "rna_bert_kmer",
            "feature_set": "kmer_tokens_plus_tabular" if numeric_processed is not None else "kmer_tokens",
            "n_train": int(len(train_index)),
            "n_validation": int(len(val_index)),
            "n_test": int(len(split.test_index)),
            "max_length_5utr": region_lengths.utr5,
            "max_length_cds": region_lengths.cds,
            "max_length_3utr": region_lengths.utr3,
            "n_tabular_features": len(numeric_feature_names),
            "kmer_size": kmer_size,
            "kmer_stride": kmer_stride,
            "n_kmer_tokens": encoded_kmers["tokens"].shape[1],
            "model_dim": model_dim,
            "region_embedding_dim": region_embedding_dim,
            "transformer_layers": transformer_layers,
            "attention_heads": attention_heads,
            "feedforward_dim": feedforward_dim,
            "dropout": dropout,
            "token_dropout": token_dropout,
            "crop_strategy": crop_strategy,
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
    predictions["model"] = "rna_bert_kmer"
    predictions["feature_set"] = metric_row["feature_set"]
    return metric_row, predictions, pd.DataFrame(history_rows)


def encode_kmer_regions(
    encoded_regions: dict[str, np.ndarray],
    *,
    region_lengths: RegionLengths,
    kmer_size: int,
    kmer_stride: int,
) -> dict[str, np.ndarray]:
    region_order = ["5utr", "cds", "3utr"]
    token_blocks = []
    region_blocks = []
    for region_id, region in enumerate(region_order, start=1):
        tokens = encode_kmer_block(encoded_regions[region], kmer_size=kmer_size, kmer_stride=kmer_stride)
        token_blocks.append(tokens)
        region_ids = np.where(tokens > 0, region_id, 0).astype(np.int64)
        region_blocks.append(region_ids)
    return {
        "tokens": np.concatenate(token_blocks, axis=1),
        "region_ids": np.concatenate(region_blocks, axis=1),
    }


def encode_kmer_block(sequences: np.ndarray, *, kmer_size: int, kmer_stride: int) -> np.ndarray:
    if kmer_size < 1:
        raise ValueError("kmer_size must be >= 1.")
    if kmer_stride < 1:
        raise ValueError("kmer_stride must be >= 1.")
    n_rows, length = sequences.shape
    n_tokens = max(1, 1 + max(0, length - kmer_size) // kmer_stride)
    encoded = np.zeros((n_rows, n_tokens), dtype=np.int64)
    powers = (4 ** np.arange(kmer_size - 1, -1, -1)).astype(np.int64)
    for token_idx in range(n_tokens):
        start = token_idx * kmer_stride
        stop = start + kmer_size
        if stop > length:
            break
        block = sequences[:, start:stop]
        valid = np.all(block > 0, axis=1)
        if not np.any(valid):
            continue
        kmer_index = ((block[valid] - 1) * powers).sum(axis=1)
        encoded[valid, token_idx] = kmer_index + 2
    return encoded


class KmerSequenceDataset:
    def __init__(
        self,
        encoded_kmers: dict[str, np.ndarray],
        target: pd.Series,
        indices: np.ndarray,
        y_mean: float,
        y_std: float,
        numeric_features: np.ndarray | None = None,
        token_dropout: float = 0.0,
        training: bool = False,
    ):
        self.tokens = encoded_kmers["tokens"]
        self.region_ids = encoded_kmers["region_ids"]
        self.target = target.to_numpy(dtype=np.float32)
        self.indices = np.asarray(indices)
        self.y_mean = y_mean
        self.y_std = y_std
        self.numeric_features = numeric_features
        self.token_dropout = token_dropout
        self.training = training

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        torch, _ = require_torch()
        idx = self.indices[item]
        tokens = torch.as_tensor(self.tokens[idx], dtype=torch.long)
        region_ids = torch.as_tensor(self.region_ids[idx], dtype=torch.long)
        if self.training and self.token_dropout > 0:
            maskable = tokens.ne(0)
            drop = torch.rand(tokens.shape).lt(self.token_dropout) & maskable
            tokens = tokens.masked_fill(drop, 0)
            region_ids = region_ids.masked_fill(drop, 0)
        features = {"tokens": tokens, "region_ids": region_ids}
        if self.numeric_features is not None:
            features["_numeric"] = torch.as_tensor(self.numeric_features[idx], dtype=torch.float32)
        y = (self.target[idx] - self.y_mean) / self.y_std
        return features, torch.tensor(y, dtype=torch.float32)


def build_rna_bert_model(
    *,
    n_tokens: int,
    vocab_size: int,
    model_dim: int,
    region_embedding_dim: int,
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

    class RNABertRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_embedding = nn.Embedding(vocab_size, model_dim, padding_idx=0)
            self.region_embedding = nn.Embedding(4, region_embedding_dim, padding_idx=0)
            self.region_projection = nn.Linear(region_embedding_dim, model_dim, bias=False)
            self.position_embedding = nn.Parameter(torch.zeros(1, n_tokens + 1, model_dim))
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

        def forward(self, batch):
            tokens = batch["tokens"]
            region_ids = batch["region_ids"]
            x = self.token_embedding(tokens) + self.region_projection(self.region_embedding(region_ids))
            cls = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)
            x = x + self.position_embedding[:, : x.shape[1], :]
            padding_mask = torch.cat(
                [torch.zeros(tokens.shape[0], 1, dtype=torch.bool, device=tokens.device), tokens.eq(0)],
                dim=1,
            )
            x = self.transformer(x, src_key_padding_mask=padding_mask)
            cls_pooled = x[:, 0, :]
            token_values = x[:, 1:, :]
            attention_scores = self.attention(token_values).squeeze(-1).masked_fill(tokens.eq(0), -1e4)
            attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(-1)
            attention_pooled = torch.sum(token_values * attention_weights, dim=1)
            pooled = torch.cat([cls_pooled, attention_pooled], dim=1)
            if self.tabular_encoder is not None:
                pooled = torch.cat([pooled, self.tabular_encoder(batch["_numeric"])], dim=1)
            return self.head(pooled).squeeze(-1)

    return RNABertRegressor()


def predict_rna_bert(model, loader, *, device: str) -> np.ndarray:
    torch, _ = require_torch()
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch, _target in loader:
            batch = {name: value.to(device, non_blocking=True) for name, value in batch.items()}
            predictions.append(model(batch).detach().cpu().numpy())
    return np.concatenate(predictions)
