#!/usr/bin/env python3
"""Fetch BibTeX and RIS records for all DOIs in the configured dataset."""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List
from urllib.parse import quote

import requests

try:  # Support running as both module and script
    from .common import DEFAULT_CONFIG_PATH, load_project_config
except ImportError:  # pragma: no cover - fallback for direct execution
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.append(str(CURRENT_DIR))
    from common import DEFAULT_CONFIG_PATH, load_project_config  # type: ignore  # noqa: E402

BIBTEX_FMT = "application/x-bibtex"
RIS_FMT = "application/x-research-info-systems"
DEFAULT_DELAY = 0.3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export BibTeX and RIS files for every DOI in the dataset.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Path to project config (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        help="Seconds to wait between resolver requests (default: 0.3).",
    )
    parser.add_argument(
        "--skip-bibtex",
        action="store_true",
        help="Do not write the BibTeX output file.",
    )
    parser.add_argument(
        "--skip-ris",
        action="store_true",
        help="Do not write the RIS output file.",
    )
    return parser.parse_args()


def _read_dois(csv_path: Path, doi_column: str) -> Iterable[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        seen = set()
        for row in reader:
            doi = (row.get(doi_column) or "").strip()
            if not doi or doi in seen:
                continue
            seen.add(doi)
            yield doi


def _resolver_request(session: requests.Session, doi: str, mime: str) -> str | None:
    encoded = quote(doi, safe="/")
    url = f"https://doi.org/{encoded}"
    try:
        response = session.get(url, timeout=30, headers={"Accept": mime})
        response.raise_for_status()
    except Exception:
        return None
    text = response.text.strip()
    return text or None


def _crossref_request(session: requests.Session, doi: str, fmt: str) -> str | None:
    encoded = quote(doi, safe="/")
    url = f"https://api.crossref.org/works/{encoded}/transform/{fmt}"
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        print(f"{doi}: failed Crossref fetch for {fmt} ({exc})")
        return None
    return response.text.strip() or None


def main() -> int:
    args = _parse_args()
    delay = args.delay if args.delay is not None else float(os.getenv("CROSSREF_DELAY", DEFAULT_DELAY))

    config = load_project_config(args.config)
    references = config.references
    mailto = os.getenv("CROSSREF_MAILTO", "you@example.com")
    user_agent = f"GenericLiteratureSearch/0.1 (mailto:{mailto})"

    dois = list(_read_dois(config.csv_path, config.columns.doi))
    if not dois:
        print("No DOIs found; nothing to fetch.")
        return 0

    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})

    if not args.skip_bibtex:
        references.bib.parent.mkdir(parents=True, exist_ok=True)
    if not args.skip_ris:
        references.ris.parent.mkdir(parents=True, exist_ok=True)
    references.log.parent.mkdir(parents=True, exist_ok=True)
    references.log.write_text("", encoding="utf-8")

    def log(message: str) -> None:
        with references.log.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")
        print(message)

    log(f"Fetching references for {len(dois)} DOIs...")

    bib_entries: List[str] = []
    ris_entries: List[str] = []
    for doi in dois:
        if not args.skip_bibtex:
            bib = _resolver_request(session, doi, BIBTEX_FMT) or _crossref_request(session, doi, BIBTEX_FMT)
            if bib:
                bib_entries.append(bib)
        if not args.skip_ris:
            ris = _resolver_request(session, doi, RIS_FMT) or _crossref_request(session, doi, RIS_FMT)
            if ris:
                ris_entries.append(ris)
        time.sleep(delay)

    if not args.skip_bibtex and bib_entries:
        references.bib.write_text("\n\n".join(bib_entries) + "\n", encoding="utf-8")
        log(f"Wrote {len(bib_entries)} BibTeX entries to {references.bib}")
    if not args.skip_ris and ris_entries:
        references.ris.write_text("\n\n".join(ris_entries) + "\n", encoding="utf-8")
        log(f"Wrote {len(ris_entries)} RIS entries to {references.ris}")
    log(f"Processed {len(dois)} DOIs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
