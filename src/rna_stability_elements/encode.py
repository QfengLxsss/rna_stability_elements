from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode

import pandas as pd
import requests
from tqdm import tqdm

BASE_URL = "https://www.encodeproject.org"


@dataclass(frozen=True)
class EncodeQuery:
    base_url: str = BASE_URL
    lab_title: str = "Mats Ljungman, UMichigan"
    status: str = "released"
    series_type: str = "PulseChaseTimeSeries"


def encode_get(
    path_or_url: str,
    *,
    session: requests.Session | None = None,
    base_url: str = BASE_URL,
    retries: int = 3,
    timeout: int = 60,
) -> dict[str, Any]:
    """Fetch JSON from ENCODE with small retry handling."""
    http = session or requests.Session()
    if path_or_url.startswith("http"):
        url = path_or_url
    else:
        url = base_url.rstrip("/") + "/" + path_or_url.strip("/")
    if "format=json" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}format=json"

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = http.get(url, headers={"Accept": "application/json"}, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"Failed to fetch ENCODE JSON: {url}") from last_error


def _first_term_name(series: dict[str, Any]) -> str:
    terms = series.get("biosample_ontology") or []
    if not terms:
        return ""
    first = terms[0]
    if isinstance(first, dict):
        return str(first.get("term_name", ""))
    return str(first)


def _dataset_time_h(dataset: dict[str, Any]) -> float:
    """Infer pulse-chase time from embedded replicates; Bru-seq controls become 0h."""
    for replicate in dataset.get("replicates", []):
        biosample = replicate.get("library", {}).get("biosample", {})
        if "pulse_chase_time" in biosample:
            return float(biosample["pulse_chase_time"])
    return 0.0


def _accession_from_id(item_id: str) -> str:
    return item_id.strip("/").split("/")[-1]


def discover_pulse_chase_series(
    query: EncodeQuery = EncodeQuery(),
    *,
    expected_terms: set[str] | None = None,
    aliases: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Discover Ljungman lab BrU/BruChase pulse-chase series and experiments.

    Returns two tables:
    - series_wide: one row per cell line with 0h/2h/6h experiment accessions.
    - experiments_long: one row per experiment time point.
    """
    params = {
        "type": query.series_type,
        "status": query.status,
        "lab.title": query.lab_title,
        "format": "json",
        "limit": "all",
    }
    url = f"{query.base_url.rstrip('/')}/search/?{urlencode(params)}"
    payload = encode_get(url, base_url=query.base_url)
    graph = payload.get("@graph", [])

    long_rows: list[dict[str, Any]] = []
    wide_rows: list[dict[str, Any]] = []

    for series in graph:
        encode_term = _first_term_name(series)
        if expected_terms and encode_term not in expected_terms:
            continue
        paper_name = (aliases or {}).get(encode_term, encode_term)
        series_accession = series["accession"]
        by_time: dict[float, str] = {}

        for dataset in series.get("related_datasets", []):
            item_id = dataset.get("@id", "")
            if not item_id:
                continue
            experiment_accession = _accession_from_id(item_id)
            time_h = _dataset_time_h(dataset)
            by_time[time_h] = experiment_accession
            long_rows.append(
                {
                    "paper_name": paper_name,
                    "encode_term": encode_term,
                    "series_accession": series_accession,
                    "experiment_accession": experiment_accession,
                    "time_h": time_h,
                    "assays": ",".join(series.get("assay_term_name", [])),
                    "status": series.get("status", ""),
                }
            )

        wide_rows.append(
            {
                "paper_name": paper_name,
                "encode_term": encode_term,
                "series_accession": series_accession,
                "experiment_0h": by_time.get(0.0, ""),
                "experiment_2h": by_time.get(2.0, ""),
                "experiment_6h": by_time.get(6.0, ""),
            }
        )

    series_wide = pd.DataFrame(wide_rows).sort_values(["paper_name"]).reset_index(drop=True)
    experiments_long = pd.DataFrame(long_rows).sort_values(
        ["paper_name", "time_h", "experiment_accession"]
    )
    return series_wide, experiments_long.reset_index(drop=True)


def collect_experiment_files(
    experiment_accessions: Iterable[str],
    *,
    query: EncodeQuery = EncodeQuery(),
    file_formats: set[str] | None = None,
    output_types: set[str] | None = None,
) -> pd.DataFrame:
    """Fetch ENCODE experiment pages and flatten released files."""
    rows: list[dict[str, Any]] = []
    http = requests.Session()

    for experiment_accession in tqdm(list(experiment_accessions), desc="ENCODE experiments"):
        experiment = encode_get(
            f"/experiments/{experiment_accession}/",
            session=http,
            base_url=query.base_url,
        )
        cell_line = _extract_experiment_cell_line(experiment)
        time_h = _extract_experiment_time_h(experiment)
        for file_item in experiment.get("files", []):
            if file_item.get("status") != query.status:
                continue
            file_format = file_item.get("file_format", "")
            output_type = file_item.get("output_type", "")
            if file_formats and file_format not in file_formats:
                continue
            if output_types and output_type not in output_types:
                continue
            href = file_item.get("href", "")
            portal_url = query.base_url.rstrip("/") + href if href.startswith("/") else href
            cloud_url = file_item.get("cloud_metadata", {}).get("url", "")
            download_url = cloud_url or portal_url
            rows.append(
                {
                    "cell_line": cell_line,
                    "time_h": time_h,
                    "experiment_accession": experiment_accession,
                    "file_accession": file_item.get("accession") or file_item.get("title", ""),
                    "file_format": file_format,
                    "file_type": file_item.get("file_type", ""),
                    "output_type": output_type,
                    "output_category": file_item.get("output_category", ""),
                    "assembly": file_item.get("assembly", ""),
                    "biological_replicates": ",".join(
                        str(x) for x in file_item.get("biological_replicates", [])
                    ),
                    "technical_replicates": ",".join(
                        str(x) for x in file_item.get("technical_replicates", [])
                    ),
                    "file_size": file_item.get("file_size", ""),
                    "md5sum": file_item.get("md5sum", ""),
                    "portal_url": portal_url,
                    "cloud_url": cloud_url,
                    "download_url": download_url,
                    "s3_uri": file_item.get("s3_uri", ""),
                    "local_path": "",
                }
            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["cell_line", "time_h", "experiment_accession", "output_type", "file_accession"]
    )


def _extract_experiment_cell_line(experiment: dict[str, Any]) -> str:
    biosample_ontology = experiment.get("biosample_ontology", {})
    if isinstance(biosample_ontology, dict):
        return str(biosample_ontology.get("term_name", ""))
    summary = experiment.get("biosample_summary", "")
    parts = str(summary).replace("Homo sapiens", "").split(",")
    return parts[0].strip()


def _extract_experiment_time_h(experiment: dict[str, Any]) -> float:
    for replicate in experiment.get("replicates", []):
        biosample = replicate.get("library", {}).get("biosample", {})
        if "pulse_chase_time" in biosample:
            return float(biosample["pulse_chase_time"])
    return 0.0


def download_files(
    files: pd.DataFrame,
    out_dir: str | Path,
    *,
    file_format: str | None = None,
    output_type: str | None = None,
    overwrite: bool = False,
    workers: int = 1,
) -> pd.DataFrame:
    """Download files listed in a manifest and update local_path."""
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = files.copy()

    if file_format:
        manifest = manifest[manifest["file_format"] == file_format].copy()
    if output_type:
        manifest = manifest[manifest["output_type"] == output_type].copy()

    local_paths: dict[int, str] = {}
    rows = [(index, row.to_dict()) for index, row in manifest.iterrows()]
    if workers <= 1:
        http = requests.Session()
        for index, row in tqdm(rows, total=len(rows), desc="Downloading"):
            result_index, path = _download_manifest_row(
                index,
                row,
                out_root,
                overwrite=overwrite,
                http=http,
                show_progress=True,
            )
            local_paths[result_index] = path
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _download_manifest_row,
                    index,
                    row,
                    out_root,
                    overwrite=overwrite,
                    http=None,
                    show_progress=False,
                )
                for index, row in rows
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
                result_index, path = future.result()
                local_paths[result_index] = path

    files = files.copy()
    for index, path in local_paths.items():
        files.loc[index, "local_path"] = path
    return files


def _download_manifest_row(
    index: int,
    row: dict[str, Any],
    out_root: Path,
    *,
    overwrite: bool,
    http: requests.Session | None,
    show_progress: bool,
) -> tuple[int, str]:
    ext = str(row["file_format"]).replace(" ", ".")
    file_name = f"{row['cell_line']}_{int(float(row['time_h']))}h_{row['file_accession']}.{ext}"
    path = out_root / file_name.replace("/", "_").replace(" ", "_")
    md5sum = str(row.get("md5sum", ""))
    if path.exists() and not overwrite:
        if not md5sum or md5sum == _md5(path):
            return index, str(path)
    session = http or requests.Session()
    url = str(row.get("download_url") or row.get("cloud_url") or row.get("portal_url"))
    _download_one(url, path, http=session, show_progress=show_progress)
    if md5sum and md5sum != _md5(path):
        raise ValueError(f"MD5 mismatch for {path}")
    return index, str(path)


def _download_one(
    url: str,
    path: Path,
    *,
    http: requests.Session,
    show_progress: bool = True,
) -> None:
    with http.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with path.open("wb") as handle:
            progress = None
            if show_progress:
                progress = tqdm(total=total, unit="B", unit_scale=True, leave=False, desc=path.name)
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    if progress is not None:
                        progress.update(len(chunk))
            if progress is not None:
                progress.close()


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
