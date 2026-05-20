from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def write_rna_lm_embeddings(
    table_path: str | Path,
    *,
    out: str | Path,
    model_name_or_path: str,
    sequence_column: str = "sequence_full",
    target_column: str = "target_label",
    sequence_format: str = "raw",
    alphabet: str = "rna",
    kmer_size: int = 6,
    kmer_stride: int = 1,
    max_length: int = 512,
    chunk_size: int = 1024,
    chunk_stride: int = 1024,
    batch_size: int = 8,
    device: str = "cuda",
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    disable_safetensors: bool = False,
    limit: int | None = None,
    resume: bool = False,
    flush_every: int = 100,
) -> pd.DataFrame:
    """Extract frozen HuggingFace RNA/DNA language-model embeddings for each transcript."""
    transformers = require_transformers()
    torch = require_torch_module()
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    table = pd.read_csv(table_path, sep="\t")
    if limit is not None:
        table = table.head(limit).copy()
    if sequence_column not in table:
        raise ValueError(f"Missing sequence column: {sequence_column}")
    out = Path(out)
    completed_gene_ids: set[str] = set()
    if resume and out.exists() and "gene_id" in table.columns:
        completed = pd.read_csv(out, sep="\t", usecols=["gene_id"])
        completed_gene_ids = set(completed["gene_id"].astype(str))
        table = table[~table["gene_id"].astype(str).isin(completed_gene_ids)].reset_index(drop=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    model = transformers.AutoModel.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        use_safetensors=False if disable_safetensors else None,
    ).to(device)
    model.eval()

    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not (resume and out.exists())
    batch_frames = []
    for start in range(0, len(table), batch_size):
        batch = table.iloc[start : start + batch_size]
        embeddings = [
            embed_sequence(
                normalize_sequence(value, alphabet=alphabet),
                tokenizer=tokenizer,
                model=model,
                torch=torch,
                sequence_format=sequence_format,
                kmer_size=kmer_size,
                kmer_stride=kmer_stride,
                max_length=max_length,
                chunk_size=chunk_size,
                chunk_stride=chunk_stride,
                device=device,
            )
            for value in batch[sequence_column]
        ]
        batch_frames.append(
            embedding_output_frame(
                batch,
                embeddings,
                target_column=target_column,
                model_name_or_path=model_name_or_path,
                sequence_column=sequence_column,
                sequence_format=sequence_format,
                alphabet=alphabet,
            )
        )
        pending_rows = sum(len(frame) for frame in batch_frames)
        if pending_rows >= flush_every:
            write_embedding_frames(out, batch_frames, header=write_header)
            write_header = False
            batch_frames = []

    if batch_frames:
        write_embedding_frames(out, batch_frames, header=write_header)

    return pd.read_csv(out, sep="\t")


def write_multi_region_rna_lm_embeddings(
    *,
    utr5_path: str | Path,
    cds_path: str | Path,
    utr3_path: str | Path,
    out: str | Path,
    target_column: str = "target_label",
    join: str = "inner",
) -> pd.DataFrame:
    """Merge separately extracted 5'UTR/CDS/3'UTR LM embeddings into one feature table."""
    if join not in {"inner", "outer"}:
        raise ValueError("join must be one of: inner, outer")
    region_paths = {
        "5utr": Path(utr5_path),
        "cds": Path(cds_path),
        "3utr": Path(utr3_path),
    }
    merged: pd.DataFrame | None = None
    for index, (region, path) in enumerate(region_paths.items()):
        frame = read_region_embedding_table(path, region=region, target_column=target_column)
        if index > 0:
            region_embedding_columns = [column for column in frame.columns if column.startswith(f"lm_{region}_emb_")]
            frame = frame[["gene_id"] + region_embedding_columns]
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on="gene_id", how=join)
    if merged is None:
        raise ValueError("No embedding tables were provided.")
    metadata_columns = [
        column
        for column in metadata_frame(merged, target_column=target_column).columns
        if column in merged.columns
    ]
    embedding_columns = sorted(column for column in merged.columns if column.startswith("lm_") and "_emb_" in column)
    output = merged[metadata_columns + embedding_columns].copy()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out, sep="\t", index=False)
    return output


def read_region_embedding_table(path: Path, *, region: str, target_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    if "gene_id" not in frame:
        raise ValueError(f"Missing gene_id column in {path}")
    if frame["gene_id"].duplicated().any():
        duplicated = frame.loc[frame["gene_id"].duplicated(), "gene_id"].head(5).tolist()
        raise ValueError(f"Duplicate gene_id values in {path}: {duplicated}")
    embedding_columns = [column for column in frame.columns if column.startswith("lm_emb_")]
    if not embedding_columns:
        raise ValueError(f"No lm_emb_* columns found in {path}")
    metadata_columns = [
        column
        for column in [
            "gene_id",
            "gene_symbol",
            "canonical_transcript_id",
            "chromosome",
            "strand",
            "gene_biotype",
            "transcript_biotype",
            "replicate_qc_flag",
            target_column,
        ]
        if column in frame.columns
    ]
    output = frame[metadata_columns + embedding_columns].copy()
    return output.rename(columns={column: f"lm_{region}_{column[len('lm_'):]}" for column in embedding_columns})


def embedding_output_frame(
    batch: pd.DataFrame,
    embeddings: list[np.ndarray],
    *,
    target_column: str,
    model_name_or_path: str,
    sequence_column: str,
    sequence_format: str,
    alphabet: str,
) -> pd.DataFrame:
    embedding = np.vstack(embeddings).astype(np.float32)
    output = metadata_frame(batch, target_column=target_column)
    embedding_frame = pd.DataFrame(
        embedding,
        columns=[f"lm_emb_{idx:04d}" for idx in range(embedding.shape[1])],
    )
    output = pd.concat([output.reset_index(drop=True), embedding_frame], axis=1)
    output["lm_model"] = model_name_or_path
    output["lm_sequence_column"] = sequence_column
    output["lm_sequence_format"] = sequence_format
    output["lm_alphabet"] = alphabet
    return output


def write_embedding_frames(path: Path, frames: list[pd.DataFrame], *, header: bool) -> None:
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(path, sep="\t", index=False, mode="a" if path.exists() and not header else "w", header=header)


def embed_sequence(
    sequence: str,
    *,
    tokenizer,
    model,
    torch,
    sequence_format: str,
    kmer_size: int,
    kmer_stride: int,
    max_length: int,
    chunk_size: int,
    chunk_stride: int,
    device: str,
) -> np.ndarray:
    chunks = sequence_chunks(sequence, chunk_size=chunk_size, chunk_stride=chunk_stride)
    embeddings = []
    with torch.no_grad():
        for chunk in chunks:
            formatted = format_sequence_for_lm(
                chunk,
                sequence_format=sequence_format,
                kmer_size=kmer_size,
                kmer_stride=kmer_stride,
            )
            inputs = tokenizer(
                formatted,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state
            attention_mask = inputs.get("attention_mask")
            if attention_mask is None:
                pooled = hidden.mean(dim=1)
            else:
                mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            embeddings.append(pooled.squeeze(0).detach().cpu().numpy())
    return np.mean(np.vstack(embeddings), axis=0)


def metadata_frame(table: pd.DataFrame, *, target_column: str) -> pd.DataFrame:
    columns = [
        column
        for column in [
            "gene_id",
            "gene_symbol",
            "canonical_transcript_id",
            "chromosome",
            "strand",
            "gene_biotype",
            "transcript_biotype",
            "replicate_qc_flag",
            target_column,
        ]
        if column in table.columns
    ]
    return table[columns].copy()


def normalize_sequence(value: object, *, alphabet: str) -> str:
    if not isinstance(value, str):
        return ""
    sequence = value.upper().replace("N", "")
    if alphabet == "rna":
        return sequence.replace("T", "U")
    if alphabet == "dna":
        return sequence.replace("U", "T")
    raise ValueError("alphabet must be one of: rna, dna")


def sequence_chunks(sequence: str, *, chunk_size: int, chunk_stride: int) -> list[str]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1.")
    if chunk_stride < 1:
        raise ValueError("chunk_stride must be >= 1.")
    if not sequence:
        return [""]
    chunks = []
    for start in range(0, len(sequence), chunk_stride):
        chunk = sequence[start : start + chunk_size]
        if not chunk:
            break
        chunks.append(chunk)
        if start + chunk_size >= len(sequence):
            break
    return chunks


def format_sequence_for_lm(
    sequence: str,
    *,
    sequence_format: str,
    kmer_size: int = 6,
    kmer_stride: int = 1,
) -> str:
    if sequence_format == "raw":
        return sequence
    if sequence_format == "spaced_chars":
        return " ".join(sequence)
    if sequence_format == "kmer":
        return " ".join(sequence_kmers(sequence, k=kmer_size, stride=kmer_stride))
    raise ValueError("sequence_format must be one of: raw, spaced_chars, kmer")


def sequence_kmers(sequence: str, *, k: int, stride: int = 1) -> list[str]:
    if k < 1:
        raise ValueError("k must be >= 1.")
    if stride < 1:
        raise ValueError("stride must be >= 1.")
    if len(sequence) < k:
        return [sequence] if sequence else []
    return [sequence[start : start + k] for start in range(0, len(sequence) - k + 1, stride)]


def require_transformers():
    try:
        import transformers  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Missing optional dependency 'transformers'. Install with: "
            "pip install 'transformers>=4.30,<4.41' 'tokenizers>=0.13' safetensors"
        ) from exc
    return transformers


def require_torch_module():
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Missing optional dependency 'torch'.") from exc
    return torch
