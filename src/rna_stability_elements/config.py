from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def cell_line_aliases(config: dict[str, Any]) -> dict[str, str]:
    """Return ENCODE term -> paper-friendly cell line name."""
    aliases: dict[str, str] = {}
    for item in config.get("cell_lines", []):
        encode_term = item["encode_term"]
        paper_name = item["paper_name"]
        aliases[encode_term] = paper_name
    return aliases


def expected_encode_terms(config: dict[str, Any]) -> set[str]:
    return {item["encode_term"] for item in config.get("cell_lines", [])}
