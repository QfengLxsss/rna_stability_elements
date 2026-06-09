from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from rna_stability_elements.models.baselines import regression_metrics
from rna_stability_elements.models.evaluation import numeric_feature_columns
from rna_stability_elements.models.multimodal import require_torch
from rna_stability_elements.models.sequence_cnn import (
    RegionLengths,
    apply_token_dropout,
    crop_sequence,
    encode_regions,
    preprocess_numeric_features,
)


LABEL_IDS = (
    "gene_sense_late_chase_6h_2h",
    "gene_sense_total_chase_6h_0h",
    "exon_sense_late_chase_6h_2h",
    "exon_sense_total_chase_6h_0h",
)

TARGET_COLUMNS = {label_id: f"target_{label_id}" for label_id in LABEL_IDS}
CODON_TO_INDEX = {
    "".join(codon): index + 1
    for index, codon in enumerate(
        (a, b, c) for a in "ACGU" for b in "ACGU" for c in "ACGU"
    )
}


@dataclass(frozen=True)
class RamhtConfig:
    region_lengths: RegionLengths = RegionLengths(utr5=256, cds=1024, utr3=1024)
    codon_length: int = 342
    embedding_dim: int = 12
    model_dim: int = 192
    codon_dim: int = 128
    transformer_layers: int = 3
    codon_layers: int = 2
    attention_heads: int = 4
    feedforward_dim: int = 512
    feature_hidden_dim: int = 512
    head_hidden_dim: int = 192
    dropout: float = 0.15
    token_dropout: float = 0.02
    crop_strategy: str = "balanced"
    fusion_mode: str = "gated_sum"
    separate_codon_stream: bool = False
    task_specific_gates: bool = False
    mask_padding_attention: bool = False
    engineered_output_skip: bool = False


def build_multitask_table(
    processed_dir: str | Path,
    *,
    label_ids: Iterable[str] = LABEL_IDS,
) -> tuple[pd.DataFrame, list[str]]:
    """Build one gene-level table with four target columns and one engineered feature matrix."""
    processed = Path(processed_dir)
    label_ids = list(label_ids)
    sequence_frames = []
    for label_id in label_ids:
        path = processed / f"parallel_modeling_master_with_sequences_{label_id}.tsv"
        frame = pd.read_csv(path, sep="\t")
        keep_columns = [
            column
            for column in [
                "gene_id",
                "gene_symbol",
                "chromosome",
                "strand",
                "gene_biotype",
                "replicate_qc_flag",
                "sequence_5utr",
                "sequence_cds",
                "sequence_3utr",
                "target_label",
            ]
            if column in frame.columns
        ]
        frame = frame[keep_columns].rename(columns={"target_label": TARGET_COLUMNS[label_id]})
        sequence_frames.append(frame)

    table = sequence_frames[0]
    sequence_columns = ["sequence_5utr", "sequence_cds", "sequence_3utr"]
    metadata_columns = ["gene_symbol", "chromosome", "strand", "gene_biotype", "replicate_qc_flag"]
    for frame in sequence_frames[1:]:
        target_column = [column for column in frame.columns if column.startswith("target_")][0]
        table = table.merge(frame[["gene_id", target_column]], on="gene_id", how="outer")
        for column in metadata_columns + sequence_columns:
            if column in frame.columns and column not in table.columns:
                table = table.merge(frame[["gene_id", column]], on="gene_id", how="left")

    table = table.sort_values("gene_id").reset_index(drop=True)

    feature_frames = []
    for label_id in label_ids:
        path = processed / f"parallel_sequence_model_features_{label_id}.tsv"
        frame = pd.read_csv(path, sep="\t")
        feature_frames.append(frame)
    feature_columns = numeric_feature_columns(feature_frames[0], target_column="target_label")
    feature_table = (
        pd.concat([frame[["gene_id", *feature_columns]] for frame in feature_frames], ignore_index=True)
        .drop_duplicates("gene_id")
        .set_index("gene_id")
    )
    table = table.merge(feature_table, on="gene_id", how="left")
    return table, feature_columns


def encode_codons(
    table: pd.DataFrame,
    *,
    codon_length: int,
    crop_strategy: str,
    random_state: int = 13,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    encoded = np.zeros((len(table), codon_length), dtype=np.int64)
    max_nt_length = codon_length * 3
    for row_index, sequence in enumerate(table["sequence_cds"].fillna("")):
        if not isinstance(sequence, str) or not sequence:
            continue
        seq = sequence.upper().replace("T", "U")
        if len(seq) > max_nt_length:
            seq = crop_sequence(seq, max_nt_length, crop_strategy=crop_strategy, rng=rng)
        seq = seq[: len(seq) - (len(seq) % 3)]
        for codon_index in range(min(codon_length, len(seq) // 3)):
            codon = seq[codon_index * 3 : codon_index * 3 + 3]
            encoded[row_index, codon_index] = CODON_TO_INDEX.get(codon, 0)
    return encoded


class RamhtDataset:
    def __init__(
        self,
        encoded_regions: dict[str, np.ndarray],
        encoded_codons: np.ndarray,
        numeric_features: np.ndarray,
        targets: np.ndarray,
        target_mask: np.ndarray,
        indices: np.ndarray,
        y_mean: np.ndarray,
        y_std: np.ndarray,
        *,
        token_dropout: float = 0.0,
        training: bool = False,
    ):
        self.encoded_regions = encoded_regions
        self.encoded_codons = encoded_codons
        self.numeric_features = numeric_features
        self.targets = targets
        self.target_mask = target_mask
        self.indices = np.asarray(indices)
        self.y_mean = y_mean
        self.y_std = y_std
        self.token_dropout = token_dropout
        self.training = training

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        torch, _ = require_torch()
        idx = self.indices[item]
        inputs = {
            name: torch.as_tensor(values[idx], dtype=torch.long)
            for name, values in self.encoded_regions.items()
        }
        inputs["codon"] = torch.as_tensor(self.encoded_codons[idx], dtype=torch.long)
        inputs["_numeric"] = torch.as_tensor(self.numeric_features[idx], dtype=torch.float32)
        if self.training and self.token_dropout > 0:
            for name in ["5utr", "cds", "3utr", "codon"]:
                inputs[name] = apply_token_dropout(inputs[name], self.token_dropout)
        y = (self.targets[idx] - self.y_mean) / self.y_std
        y = np.where(self.target_mask[idx], y, 0.0)
        return (
            inputs,
            torch.as_tensor(y, dtype=torch.float32),
            torch.as_tensor(self.target_mask[idx], dtype=torch.float32),
        )


def build_ramht_model(config: RamhtConfig, *, feature_dim: int, n_tasks: int = 4):
    torch, nn = require_torch()
    if config.model_dim % config.attention_heads != 0:
        raise ValueError("model_dim must be divisible by attention_heads.")
    if config.codon_dim % config.attention_heads != 0:
        raise ValueError("codon_dim must be divisible by attention_heads.")
    if config.fusion_mode not in {"gated_sum", "gated_residual"}:
        raise ValueError("fusion_mode must be 'gated_sum' or 'gated_residual'.")

    class AttentionPool(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.score = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))

        def forward(self, x, mask):
            valid_rows = mask.any(dim=1, keepdim=True)
            safe_mask = mask.clone()
            safe_mask[:, 0] |= ~valid_rows.squeeze(1)
            scores = self.score(x).squeeze(-1).masked_fill(~safe_mask, -1e9)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            pooled = torch.sum(x * weights, dim=1)
            return pooled * valid_rows.to(dtype=pooled.dtype)

    class RegionEncoder(nn.Module):
        def __init__(self, length: int):
            super().__init__()
            self.length = length
            self.embedding = nn.Embedding(5, config.embedding_dim, padding_idx=0)
            self.projection = nn.Linear(config.embedding_dim, config.model_dim)
            self.position = nn.Parameter(torch.zeros(1, length, config.model_dim))
            layer = nn.TransformerEncoderLayer(
                d_model=config.model_dim,
                nhead=config.attention_heads,
                dim_feedforward=config.feedforward_dim,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=config.transformer_layers)
            self.pool = AttentionPool(config.model_dim)

        def forward(self, tokens):
            mask = tokens.ne(0)
            x = self.projection(self.embedding(tokens)) + self.position[:, : tokens.shape[1], :]
            if config.mask_padding_attention:
                safe_mask = mask.clone()
                safe_mask[:, 0] |= ~mask.any(dim=1)
                x = x * mask.unsqueeze(-1)
                x = self.encoder(x, src_key_padding_mask=~safe_mask)
                x = x * mask.unsqueeze(-1)
            else:
                x = self.encoder(x)
            return self.pool(x, mask)

    class CodonEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(65, config.codon_dim, padding_idx=0)
            self.position = nn.Parameter(torch.zeros(1, config.codon_length, config.codon_dim))
            layer = nn.TransformerEncoderLayer(
                d_model=config.codon_dim,
                nhead=config.attention_heads,
                dim_feedforward=config.feedforward_dim,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=config.codon_layers)
            self.pool = AttentionPool(config.codon_dim)
            self.output = nn.Linear(config.codon_dim, config.model_dim)

        def forward(self, tokens):
            mask = tokens.ne(0)
            x = self.embedding(tokens) + self.position[:, : tokens.shape[1], :]
            if config.mask_padding_attention:
                safe_mask = mask.clone()
                safe_mask[:, 0] |= ~mask.any(dim=1)
                x = x * mask.unsqueeze(-1)
                x = self.encoder(x, src_key_padding_mask=~safe_mask)
                x = x * mask.unsqueeze(-1)
            else:
                x = self.encoder(x)
            return self.output(self.pool(x, mask))

    class Ramht(nn.Module):
        def __init__(self):
            super().__init__()
            self.region_encoders = nn.ModuleDict(
                {
                    "5utr": RegionEncoder(config.region_lengths.utr5),
                    "cds": RegionEncoder(config.region_lengths.cds),
                    "3utr": RegionEncoder(config.region_lengths.utr3),
                }
            )
            self.region_type_embedding = nn.Parameter(torch.zeros(1, 4, config.model_dim))
            region_layer = nn.TransformerEncoderLayer(
                d_model=config.model_dim,
                nhead=config.attention_heads,
                dim_feedforward=config.feedforward_dim,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.region_attention = nn.TransformerEncoder(region_layer, num_layers=1)
            self.region_pool = AttentionPool(config.model_dim)
            self.codon_encoder = CodonEncoder()
            self.feature_encoder = nn.Sequential(
                nn.Linear(feature_dim, config.feature_hidden_dim),
                nn.LayerNorm(config.feature_hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.feature_hidden_dim, config.model_dim),
                nn.LayerNorm(config.model_dim),
                nn.GELU(),
            )
            gate_outputs = n_tasks * 3 if config.task_specific_gates else 3
            self.gate = nn.Sequential(
                nn.Linear(config.model_dim * 3, config.model_dim),
                nn.GELU(),
                nn.Linear(config.model_dim, gate_outputs),
            )
            self.concat_fusion = nn.Sequential(
                nn.Linear(config.model_dim * 3, config.model_dim),
                nn.LayerNorm(config.model_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            self.shared = nn.Sequential(
                nn.LayerNorm(config.model_dim),
                nn.Dropout(config.dropout),
                nn.Linear(config.model_dim, config.head_hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            self.heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(config.head_hidden_dim, config.head_hidden_dim // 2),
                        nn.GELU(),
                        nn.Dropout(config.dropout),
                        nn.Linear(config.head_hidden_dim // 2, 1),
                    )
                    for _ in range(n_tasks)
                ]
            )
            self.engineered_skip = (
                nn.Linear(feature_dim, n_tasks) if config.engineered_output_skip else None
            )

        def forward(self, inputs):
            nucleotide_vectors = [
                self.region_encoders["5utr"](inputs["5utr"]),
                self.region_encoders["cds"](inputs["cds"]),
                self.region_encoders["3utr"](inputs["3utr"]),
            ]
            h_codon = self.codon_encoder(inputs["codon"])
            region_inputs = nucleotide_vectors if config.separate_codon_stream else nucleotide_vectors + [h_codon]
            region_vectors = torch.stack(region_inputs, dim=1)
            region_vectors = self.region_attention(
                region_vectors + self.region_type_embedding[:, : region_vectors.shape[1], :]
            )
            region_mask = torch.ones(
                region_vectors.shape[:2], dtype=torch.bool, device=region_vectors.device
            )
            h_seq = self.region_pool(region_vectors, region_mask)
            if not config.separate_codon_stream:
                h_codon = region_vectors[:, 3, :]
            h_feat = self.feature_encoder(inputs["_numeric"])
            streams = torch.stack([h_seq, h_codon, h_feat], dim=1)
            concatenated = torch.cat([h_seq, h_codon, h_feat], dim=1)
            gate_logits = self.gate(concatenated)
            if config.task_specific_gates:
                gate_weights = torch.softmax(gate_logits.view(-1, n_tasks, 3), dim=2)
                fused = torch.sum(gate_weights.unsqueeze(-1) * streams.unsqueeze(1), dim=2)
                if config.fusion_mode == "gated_residual":
                    fused = fused + self.concat_fusion(concatenated).unsqueeze(1)
                shared = self.shared(fused)
                predictions = torch.cat(
                    [head(shared[:, task_index, :]) for task_index, head in enumerate(self.heads)],
                    dim=1,
                )
            else:
                gate_weights = torch.softmax(gate_logits, dim=1)
                fused = torch.sum(gate_weights.unsqueeze(-1) * streams, dim=1)
                if config.fusion_mode == "gated_residual":
                    fused = fused + self.concat_fusion(concatenated)
                shared = self.shared(fused)
                predictions = torch.cat([head(shared) for head in self.heads], dim=1)
            if self.engineered_skip is not None:
                predictions = predictions + self.engineered_skip(inputs["_numeric"])
            return predictions, gate_weights

    return Ramht()


def make_multitask_split_indices(
    table: pd.DataFrame,
    processed_dir: str | Path,
    *,
    split_name: str,
    label_ids: Iterable[str] = LABEL_IDS,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, dict[str, object]]]:
    processed = Path(processed_dir)
    label_ids = list(label_ids)
    all_test_genes: set[str] = set()
    all_validation_genes: set[str] = set()
    test_by_label: dict[str, np.ndarray] = {}
    split_metadata: dict[str, dict[str, object]] = {}
    for label_id in label_ids:
        manifest = pd.read_csv(processed / f"fair_benchmark_splits_{label_id}.tsv", sep="\t")
        split_manifest = manifest[manifest["split_name"] == split_name]
        if split_manifest.empty:
            raise ValueError(f"Missing split {split_name!r} for label {label_id!r}.")
        test_genes = set(split_manifest.loc[split_manifest["role"] == "test", "gene_id"])
        val_genes = set(split_manifest.loc[split_manifest["role"] == "validation", "gene_id"])
        all_test_genes.update(test_genes)
        all_validation_genes.update(val_genes)
        test_by_label[label_id] = table.index[table["gene_id"].isin(test_genes)].to_numpy()
        row = split_manifest.iloc[0]
        split_metadata[label_id] = {
            "evaluation": row["evaluation"],
            "holdout_group": row["holdout_group"],
            "repeat": int(row["repeat"]),
        }
    validation_genes = all_validation_genes - all_test_genes
    train_mask = ~table["gene_id"].isin(all_test_genes | validation_genes)
    validation_mask = table["gene_id"].isin(validation_genes)
    return (
        table.index[train_mask].to_numpy(),
        table.index[validation_mask].to_numpy(),
        test_by_label,
        split_metadata,
    )


def train_ramht_split(
    table: pd.DataFrame,
    feature_columns: list[str],
    *,
    processed_dir: str | Path,
    split_name: str,
    config: RamhtConfig = RamhtConfig(),
    label_ids: Iterable[str] = LABEL_IDS,
    batch_size: int = 32,
    max_epochs: int = 30,
    patience: int = 6,
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-4,
    task_weights: Iterable[float] = (1.2, 1.0, 1.2, 1.0),
    random_state: int = 13,
    device: str = "cuda",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    torch, nn = require_torch()
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    torch.manual_seed(random_state)
    if device == "cuda":
        torch.cuda.manual_seed_all(random_state)
    label_ids = list(label_ids)
    target_columns = [TARGET_COLUMNS[label_id] for label_id in label_ids]
    train_index, val_index, test_by_label, split_metadata = make_multitask_split_indices(
        table, processed_dir, split_name=split_name, label_ids=label_ids
    )

    target_values = table[target_columns].to_numpy(dtype=np.float32)
    target_mask = ~np.isnan(target_values)
    y_mean = np.nanmean(target_values[train_index], axis=0).astype(np.float32)
    y_std = np.nanstd(target_values[train_index], axis=0).astype(np.float32)
    y_std[y_std == 0] = 1.0
    target_values = np.nan_to_num(target_values, nan=0.0)

    encoded_regions = encode_regions(
        table,
        region_lengths=config.region_lengths,
        crop_strategy=config.crop_strategy,
        random_state=random_state,
    )
    encoded_codons = encode_codons(
        table,
        codon_length=config.codon_length,
        crop_strategy=config.crop_strategy,
        random_state=random_state,
    )
    raw_numeric = table[feature_columns].to_numpy(dtype=np.float32)
    numeric = preprocess_numeric_features(
        raw_numeric,
        train_index=train_index,
        val_index=val_index,
        test_index=np.asarray(sorted(set().union(*[set(v) for v in test_by_label.values()]))),
    )

    train_dataset = RamhtDataset(
        encoded_regions,
        encoded_codons,
        numeric,
        target_values,
        target_mask,
        train_index,
        y_mean,
        y_std,
        token_dropout=config.token_dropout,
        training=True,
    )
    val_dataset = RamhtDataset(
        encoded_regions, encoded_codons, numeric, target_values, target_mask, val_index, y_mean, y_std
    )
    pin_memory = device == "cuda"
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory
    )

    model = build_ramht_model(config, feature_dim=len(feature_columns), n_tasks=len(label_ids)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    weights = torch.as_tensor(list(task_weights), dtype=torch.float32, device=device)
    best_state = None
    best_val_loss = float("inf")
    stale_epochs = 0
    history_rows = []
    for epoch in range(1, max_epochs + 1):
        train_loss = _run_ramht_epoch(
            model, train_loader, optimizer=optimizer, device=device, task_weights=weights
        )
        val_loss = _run_ramht_epoch(
            model, val_loader, optimizer=None, device=device, task_weights=weights
        )
        history_rows.append(
            {
                "split_name": split_name,
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

    metrics_rows = []
    prediction_frames = []
    gate_rows = []
    for task_index, label_id in enumerate(label_ids):
        test_index = test_by_label[label_id]
        test_dataset = RamhtDataset(
            encoded_regions,
            encoded_codons,
            numeric,
            target_values,
            target_mask,
            test_index,
            y_mean,
            y_std,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory
        )
        pred_std, gate_values = _predict_ramht(model, test_loader, device=device)
        y_pred = pred_std[:, task_index] * y_std[task_index] + y_mean[task_index]
        y_true = table.loc[test_index, target_columns[task_index]].to_numpy(dtype=np.float32)
        valid = ~np.isnan(y_true)
        metric_row = regression_metrics(y_true[valid], y_pred[valid])
        metadata = split_metadata[label_id]
        metric_row.update(
            {
                "label_id": label_id,
                "evaluation": metadata["evaluation"],
                "split_name": split_name,
                "holdout_group": metadata["holdout_group"],
                "repeat": metadata["repeat"],
                "model": _ramht_model_id(config),
                "feature_set": _ramht_feature_set(config),
                "n_train": int(len(train_index)),
                "n_validation": int(len(val_index)),
                "n_test": int(valid.sum()),
                "n_total_test_genes_for_label": int(len(test_index)),
                "max_length_5utr": config.region_lengths.utr5,
                "max_length_cds": config.region_lengths.cds,
                "max_length_3utr": config.region_lengths.utr3,
                "codon_length": config.codon_length,
                "n_tabular_features": len(feature_columns),
                "best_val_loss": best_val_loss,
                "epochs_trained": len(history_rows),
                "device": device,
            }
        )
        metrics_rows.append(metric_row)
        prediction_columns = [
            column
            for column in ["gene_id", "gene_symbol", "chromosome", "replicate_qc_flag"]
            if column in table.columns
        ]
        predictions = table.loc[test_index, prediction_columns].copy()
        predictions["label_id"] = label_id
        predictions["y_true"] = y_true
        predictions["y_pred"] = y_pred
        predictions["residual"] = predictions["y_pred"] - predictions["y_true"]
        predictions["evaluation"] = metadata["evaluation"]
        predictions["split_name"] = split_name
        predictions["holdout_group"] = metadata["holdout_group"]
        predictions["repeat"] = metadata["repeat"]
        predictions["model"] = metric_row["model"]
        predictions["feature_set"] = metric_row["feature_set"]
        prediction_frames.append(predictions[predictions["y_true"].notna()])
        gate_rows.append(
            {
                "label_id": label_id,
                "split_name": split_name,
                "gate_sequence_mean": float(_task_gate_values(gate_values, valid, task_index)[:, 0].mean()),
                "gate_codon_mean": float(_task_gate_values(gate_values, valid, task_index)[:, 1].mean()),
                "gate_engineered_mean": float(_task_gate_values(gate_values, valid, task_index)[:, 2].mean()),
            }
        )

    history = pd.DataFrame(history_rows)
    gates = pd.DataFrame(gate_rows)
    history = history.merge(gates, on="split_name", how="left")
    return pd.DataFrame(metrics_rows), pd.concat(prediction_frames, ignore_index=True), history


def _ramht_model_id(config: RamhtConfig) -> str:
    if (
        config.task_specific_gates
        or config.separate_codon_stream
        or config.fusion_mode != "gated_sum"
        or config.mask_padding_attention
        or config.engineered_output_skip
    ):
        return "ramht_v2"
    return "ramht"


def _ramht_feature_set(config: RamhtConfig) -> str:
    parts = ["raw_regions", "codon", "engineered", "multitask"]
    if config.separate_codon_stream:
        parts.append("separate_codon")
    if config.task_specific_gates:
        parts.append("task_gates")
    if config.mask_padding_attention:
        parts.append("padding_mask")
    if config.engineered_output_skip:
        parts.append("engineered_skip")
    parts.append(config.fusion_mode)
    return "_".join(parts)


def _task_gate_values(gate_values: np.ndarray, valid: np.ndarray, task_index: int) -> np.ndarray:
    if gate_values.ndim == 3:
        return gate_values[valid, task_index, :]
    return gate_values[valid, :]


def _run_ramht_epoch(model, loader, *, optimizer, device: str, task_weights) -> float:
    torch, nn = require_torch()
    model.train(optimizer is not None)
    losses = []
    with torch.set_grad_enabled(optimizer is not None):
        for inputs, target, mask in loader:
            inputs = {name: value.to(device, non_blocking=True) for name, value in inputs.items()}
            target = target.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            prediction, _gate = model(inputs)
            loss_matrix = nn.functional.smooth_l1_loss(prediction, target, reduction="none")
            weighted_mask = mask * task_weights.unsqueeze(0)
            loss = (loss_matrix * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach().cpu()) * len(target))
    return float(np.sum(losses) / max(1, len(loader.dataset)))


def _predict_ramht(model, loader, *, device: str) -> tuple[np.ndarray, np.ndarray]:
    torch, _ = require_torch()
    model.eval()
    predictions = []
    gates = []
    with torch.no_grad():
        for inputs, _target, _mask in loader:
            inputs = {name: value.to(device, non_blocking=True) for name, value in inputs.items()}
            prediction, gate = model(inputs)
            predictions.append(prediction.detach().cpu().numpy())
            gates.append(gate.detach().cpu().numpy())
    return np.concatenate(predictions), np.concatenate(gates)


def write_ramht_split(
    processed_dir: str | Path,
    *,
    split_name: str,
    metrics_out: str | Path,
    predictions_out: str | Path,
    history_out: str | Path,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    table, feature_columns = build_multitask_table(processed_dir)
    metrics, predictions, history = train_ramht_split(
        table, feature_columns, processed_dir=processed_dir, split_name=split_name, **kwargs
    )
    for path, frame in [
        (metrics_out, metrics),
        (predictions_out, predictions),
        (history_out, history),
    ]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, sep="\t", index=False)
    return metrics, predictions, history
