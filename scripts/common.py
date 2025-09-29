#!/usr/bin/env python3
"""Shared helpers for the generalized literature search workflow."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import re


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "project.json"


@dataclass(frozen=True)
class ColumnConfig:
    doi: str
    title: str
    status: str
    path: str
    notes: str


@dataclass(frozen=True)
class ReferenceConfig:
    bib: Path
    ris: Path
    log: Path


@dataclass(frozen=True)
class ProjectConfig:
    root_dir: Path
    csv_path: Path
    full_text_dir: Path
    columns: ColumnConfig
    open_access_sources: Path
    references: ReferenceConfig
    search_plan: Path


def _resolve_path(raw_path: str, project_root: Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (project_root / path)


def load_project_config(config_path: Path | None = None) -> ProjectConfig:
    """Load the JSON configuration and convert relative paths."""

    cfg_path = config_path or DEFAULT_CONFIG_PATH
    cfg_path = cfg_path.expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")

    data: Dict[str, Any] = json.loads(cfg_path.read_text(encoding="utf-8"))
    project_root = cfg_path.parents[1]

    columns_raw = data.get("columns") or {}
    columns = ColumnConfig(
        doi=columns_raw.get("doi", "DOI"),
        title=columns_raw.get("title", "Title"),
        status=columns_raw.get("status", "Full Text Status"),
        path=columns_raw.get("path", "Full Text Path"),
        notes=columns_raw.get("notes", "Full Text Notes"),
    )

    references_raw = data.get("references") or {}
    references = ReferenceConfig(
        bib=_resolve_path(references_raw.get("bib", "data/processed/library.bib"), project_root),
        ris=_resolve_path(references_raw.get("ris", "data/processed/library.ris"), project_root),
        log=_resolve_path(references_raw.get("log", "data/processed/fetch_references.log"), project_root),
    )

    return ProjectConfig(
        root_dir=project_root,
        csv_path=_resolve_path(data.get("csv_path", "data/processed/library.csv"), project_root),
        full_text_dir=_resolve_path(data.get("full_text_dir", "data/full_text"), project_root),
        columns=columns,
        open_access_sources=_resolve_path(
            data.get("open_access_sources", "config/open_access_sources.json"), project_root
        ),
        references=references,
        search_plan=_resolve_path(data.get("search_plan", "config/search_plan.json"), project_root),
    )


def load_open_access_sources(path: Path) -> Dict[str, Dict[str, Any]]:
    """Return the DOI→metadata mapping used for deterministic downloads."""

    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data
    raise ValueError(
        "open_access_sources must be a JSON object mapping DOI→{url,note}."
    )


def slugify(value: str) -> str:
    """Generate a filesystem-friendly slug from a title or DOI."""

    normalized = re.sub(r"[^A-Za-z0-9]+", "-", value)
    return normalized.strip("-")[:120] or "paper"
