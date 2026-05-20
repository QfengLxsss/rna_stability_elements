from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from rna_stability_elements.models.baselines import regression_metrics
from rna_stability_elements.models.evaluation import Split, build_splits, numeric_feature_columns
from rna_stability_elements.models.multimodal import require_torch


REGION_COLUMNS = {
    "5utr": "sequence_5utr",
    "cds": "sequence_cds",
    "3utr": "sequence_3utr",
}


@dataclass(frozen=True)
class RegionLengths:
    utr5: int = 512
    cds: int = 4096
    utr3: int = 4096

    def as_dict(self) -> dict[str, int]:
        return {"5utr": self.utr5, "cds": self.cds, "3utr": self.utr3}


def evaluate_region_cnn(
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
    region_lengths: RegionLengths = RegionLengths(),
    batch_size: int = 64,
    max_epochs: int = 30,
    patience: int = 6,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    embedding_dim: int = 8,
    channels: int = 96,
    hidden_dim: int = 192,
    tabular_hidden_dim: int = 128,
    dropout: float = 0.2,
    region_dropout: float = 0.0,
    token_dropout: float = 0.0,
    crop_strategy: str = "balanced",
    device: str = "cuda",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate a region-aware sequence CNN on repeated random or chromosome holdout splits."""
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
        metrics, predictions, history = train_region_cnn_split(
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
            channels=channels,
            hidden_dim=hidden_dim,
            tabular_hidden_dim=tabular_hidden_dim,
            dropout=dropout,
            region_dropout=region_dropout,
            token_dropout=token_dropout,
            crop_strategy=crop_strategy,
            random_state=random_state,
            device=device,
        )
        metrics_rows.append(metrics)
        prediction_frames.append(predictions)
        history_frames.append(history)

    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    history = pd.concat(history_frames, ignore_index=True)
    return metrics, predictions, history


def write_region_cnn_evaluation(
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
    metrics, predictions, history = evaluate_region_cnn(table, feature_table=feature_table, **kwargs)
    for path, frame in [
        (metrics_out, metrics),
        (predictions_out, predictions),
        (history_out, history),
    ]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, sep="\t", index=False)
    return metrics, predictions, history


def encode_regions(
    table: pd.DataFrame,
    *,
    region_lengths: RegionLengths,
    crop_strategy: str = "balanced",
    random_state: int = 13,
) -> dict[str, np.ndarray]:
    lengths = region_lengths.as_dict()
    encoded = {}
    rng = np.random.default_rng(random_state)
    for region, column in REGION_COLUMNS.items():
        if column not in table:
            raise ValueError(f"Missing sequence column: {column}")
        encoded[region] = np.stack(
            [
                encode_sequence(value, lengths[region], crop_strategy=crop_strategy, rng=rng)
                for value in table[column].fillna("")
            ]
        )
    return encoded


def align_numeric_features(
    data: pd.DataFrame,
    *,
    feature_table: pd.DataFrame | None,
    target_column: str,
) -> tuple[np.ndarray | None, list[str]]:
    if feature_table is None:
        return None, []
    if "gene_id" not in data or "gene_id" not in feature_table:
        raise ValueError("Both sequence table and feature table need a gene_id column.")
    feature_columns = numeric_feature_columns(feature_table, target_column=target_column)
    if not feature_columns:
        return None, []
    aligned = feature_table.set_index("gene_id").reindex(data["gene_id"])
    return aligned[feature_columns].to_numpy(dtype=np.float32), feature_columns


def encode_sequence(
    sequence: object,
    max_length: int,
    *,
    crop_strategy: str = "balanced",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    values = np.zeros(max_length, dtype=np.int64)
    if not isinstance(sequence, str) or not sequence:
        return values
    seq = sequence.upper().replace("T", "U")
    if len(seq) > max_length:
        seq = crop_sequence(seq, max_length, crop_strategy=crop_strategy, rng=rng)
    mapping = {"A": 1, "C": 2, "G": 3, "U": 4}
    for idx, base in enumerate(seq[:max_length]):
        values[idx] = mapping.get(base, 0)
    return values


def crop_sequence(
    sequence: str,
    max_length: int,
    *,
    crop_strategy: str = "balanced",
    rng: np.random.Generator | None = None,
) -> str:
    if crop_strategy == "balanced":
        left = max_length // 2
        right = max_length - left
        return sequence[:left] + sequence[-right:]
    if crop_strategy == "start":
        return sequence[:max_length]
    if crop_strategy == "end":
        return sequence[-max_length:]
    if crop_strategy == "random":
        rng = rng or np.random.default_rng()
        start = int(rng.integers(0, len(sequence) - max_length + 1))
        return sequence[start : start + max_length]
    raise ValueError("crop_strategy must be one of: balanced, start, end, random")


def train_region_cnn_split(
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
    channels: int,
    hidden_dim: int,
    tabular_hidden_dim: int,
    dropout: float,
    region_dropout: float,
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

    model = build_region_aware_cnn(
        region_lengths=region_lengths,
        embedding_dim=embedding_dim,
        channels=channels,
        hidden_dim=hidden_dim,
        tabular_dim=0 if numeric_processed is None else len(numeric_feature_names),
        tabular_hidden_dim=tabular_hidden_dim,
        dropout=dropout,
        region_dropout=region_dropout,
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

    y_pred_std = predict_region_cnn(model, test_loader, device=device)
    y_pred = y_pred_std * y_std + y_mean
    y_true = data.loc[split.test_index, target_column].to_numpy(dtype=np.float32)
    metric_row: dict[str, object] = regression_metrics(y_true, y_pred)
    metric_row.update(
        {
            "evaluation": split.evaluation,
            "split_name": split.split_name,
            "holdout_group": split.holdout_group,
            "repeat": split.repeat,
            "model": "region_cnn",
            "feature_set": "raw_5utr_cds_3utr",
            "n_train": int(len(train_index)),
            "n_validation": int(len(val_index)),
            "n_test": int(len(split.test_index)),
            "max_length_5utr": region_lengths.utr5,
            "max_length_cds": region_lengths.cds,
            "max_length_3utr": region_lengths.utr3,
            "n_tabular_features": len(numeric_feature_names),
            "region_dropout": region_dropout,
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
    predictions["model"] = "region_cnn"
    predictions["feature_set"] = "raw_5utr_cds_3utr"
    return metric_row, predictions, pd.DataFrame(history_rows)


def make_train_val_indices(
    train_index: np.ndarray, *, random_state: int, val_fraction: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(train_index)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    val_index = np.sort(shuffled[:n_val])
    train_index = np.sort(shuffled[n_val:])
    return train_index, val_index


def preprocess_numeric_features(
    numeric_features: np.ndarray | None,
    *,
    train_index: np.ndarray,
    val_index: np.ndarray,
    test_index: np.ndarray,
) -> np.ndarray | None:
    if numeric_features is None:
        return None
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_values = scaler.fit_transform(imputer.fit_transform(numeric_features[train_index]))
    processed = np.zeros_like(numeric_features, dtype=np.float32)
    processed[train_index] = train_values.astype(np.float32)
    for indices in [val_index, test_index]:
        processed[indices] = scaler.transform(imputer.transform(numeric_features[indices])).astype(np.float32)
    return processed


class RegionSequenceDataset:
    def __init__(
        self,
        encoded: dict[str, np.ndarray],
        target: pd.Series,
        indices: np.ndarray,
        y_mean: float,
        y_std: float,
        numeric_features: np.ndarray | None = None,
        token_dropout: float = 0.0,
        training: bool = False,
    ):
        self.encoded = encoded
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
        regions = {name: torch.as_tensor(values[idx], dtype=torch.long) for name, values in self.encoded.items()}
        if self.training and self.token_dropout > 0:
            regions = {name: apply_token_dropout(value, self.token_dropout) for name, value in regions.items()}
        if self.numeric_features is not None:
            regions["_numeric"] = torch.as_tensor(self.numeric_features[idx], dtype=torch.float32)
        y = (self.target[idx] - self.y_mean) / self.y_std
        return regions, torch.tensor(y, dtype=torch.float32)


def build_region_aware_cnn(
    *,
    region_lengths: RegionLengths,
    embedding_dim: int,
    channels: int,
    hidden_dim: int,
    dropout: float,
    tabular_dim: int = 0,
    tabular_hidden_dim: int = 128,
    region_dropout: float = 0.0,
):
    torch, nn = require_torch()

    class RegionEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(embedding_dim, channels, kernel_size=9, padding=4),
                nn.GELU(),
                nn.Conv1d(channels, channels, kernel_size=9, padding=8, dilation=2),
                nn.GELU(),
                nn.AdaptiveMaxPool1d(1),
                nn.Flatten(),
            )

        def forward(self, x):
            return self.net(x)

    class RegionAwareCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(5, embedding_dim, padding_idx=0)
            self.region_lengths = region_lengths
            self.region_dropout = region_dropout
            self.encoders = nn.ModuleDict(
                {"5utr": RegionEncoder(), "cds": RegionEncoder(), "3utr": RegionEncoder()}
            )
            if tabular_dim > 0:
                self.tabular_encoder = nn.Sequential(
                    nn.Linear(tabular_dim, tabular_hidden_dim),
                    nn.LayerNorm(tabular_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                head_input_dim = channels * 3 + tabular_hidden_dim
            else:
                self.tabular_encoder = None
                head_input_dim = channels * 3
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
            encoded_regions = []
            for region in ["5utr", "cds", "3utr"]:
                x = self.embedding(regions[region]).transpose(1, 2)
                latent = self.encoders[region](x)
                if self.training and self.region_dropout > 0:
                    keep = torch.empty(latent.shape[0], 1, device=latent.device).bernoulli_(
                        1 - self.region_dropout
                    )
                    latent = latent * keep / max(1e-6, 1 - self.region_dropout)
                encoded_regions.append(latent)
            if self.tabular_encoder is not None:
                encoded_regions.append(self.tabular_encoder(regions["_numeric"]))
            return self.head(torch.cat(encoded_regions, dim=1)).squeeze(-1)

    return RegionAwareCNN()


def apply_token_dropout(tokens, dropout: float):
    torch, _ = require_torch()
    if dropout <= 0:
        return tokens
    maskable = tokens.ne(0)
    drop = torch.rand(tokens.shape).lt(dropout) & maskable
    return tokens.masked_fill(drop, 0)


def run_epoch(model, loader, *, criterion, optimizer, device: str) -> float:
    torch, _ = require_torch()
    model.train(optimizer is not None)
    losses = []
    with torch.set_grad_enabled(optimizer is not None):
        for regions, target in loader:
            regions = {name: value.to(device, non_blocking=True) for name, value in regions.items()}
            target = target.to(device, non_blocking=True)
            prediction = model(regions)
            loss = criterion(prediction, target)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach().cpu()) * len(target))
    return float(np.sum(losses) / max(1, len(loader.dataset)))


def predict_region_cnn(model, loader, *, device: str) -> np.ndarray:
    torch, _ = require_torch()
    model.eval()
    predictions = []
    with torch.no_grad():
        for regions, _target in loader:
            regions = {name: value.to(device, non_blocking=True) for name, value in regions.items()}
            predictions.append(model(regions).detach().cpu().numpy())
    return np.concatenate(predictions)
