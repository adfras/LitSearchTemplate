#!/usr/bin/env python3
"""Locate and download PDFs via Serper-powered Google searches."""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import requests

try:  # Allow running both as a module and as a script
    from .common import (
        DEFAULT_CONFIG_PATH,
        load_project_config,
        slugify,
    )
except ImportError:  # pragma: no cover - fallback for direct execution
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.append(str(CURRENT_DIR))
    from common import (  # type: ignore  # noqa: E402
        DEFAULT_CONFIG_PATH,
        load_project_config,
        slugify,
    )

SERPER_ENDPOINT = "https://google.serper.dev/search"
BLOCKED_HOST_KEYWORDS = {"sci-hub", "scihub", "libgen"}
DEFAULT_DELAY = 0.25
MAX_URLS_PER_ROW = 12
PDF_URL_RE = re.compile(r"href=[\"']([^\"']+\.pdf[^\"']*)", re.IGNORECASE)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover PDFs using Serper search results and update the dataset.",
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
        help="Override delay (in seconds) between Serper requests.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on how many dataset rows to process.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Explore candidate URLs without downloading or updating the CSV.",
    )
    return parser.parse_args()


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    )
    return session


def _serper_search(api_key: str, query: str) -> List[str]:
    payload = {"q": query, "num": 10}
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    try:
        response = requests.post(SERPER_ENDPOINT, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        print(f"Serper request failed for '{query}': {exc}")
        return []
    data = response.json()
    urls: List[str] = []
    for section in ("organic", "answerBox", "topStories"):
        entries = data.get(section)
        if not entries:
            continue
        if isinstance(entries, dict):
            entries = [entries]
        for entry in entries:
            url = entry.get("link") if isinstance(entry, dict) else None
            if isinstance(url, str):
                urls.append(url)
            for subkey in ("source", "preview", "image"):
                maybe = entry.get(subkey) if isinstance(entry, dict) else None
                if isinstance(maybe, dict):
                    link = maybe.get("link")
                    if isinstance(link, str):
                        urls.append(link)
    seen = set()
    unique_urls: List[str] = []
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        unique_urls.append(url)
    return unique_urls


def _looks_like_pdf(url: str) -> bool:
    lowered = url.lower()
    return lowered.endswith(".pdf") or ".pdf?" in lowered or lowered.endswith(".pdf/")


def _find_pdf_in_page(session: requests.Session, url: str) -> Optional[str]:
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        print(f"  Fetch failed for page {url}: {exc}")
        return None
    matches = PDF_URL_RE.findall(response.text)
    for match in matches:
        pdf_url = requests.compat.urljoin(response.url, match)
        if not _looks_like_pdf(pdf_url):
            continue
        lowered = pdf_url.lower()
        if any(keyword in lowered for keyword in BLOCKED_HOST_KEYWORDS):
            continue
        return pdf_url
    return None


def _download_pdf(session: requests.Session, pdf_url: str, dest: Path) -> Optional[str]:
    headers = {
        "Referer": pdf_url.rsplit("/", 1)[0],
        "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
    }
    try:
        response = session.get(pdf_url, timeout=60, headers=headers)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        print(f"  Download failed for {pdf_url}: {exc}")
        return None
    content_type = (response.headers.get("Content-Type") or "").lower()
    content = response.content.lstrip()
    if "application/pdf" not in content_type and not content.startswith(b"%PDF"):
        print(f"  Not a PDF according to content-type for {pdf_url}")
        return None
    dest.write_bytes(response.content)
    return pdf_url


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _candidate_urls(
    api_key: str,
    row: dict,
    columns: Tuple[str, str],
    delay_seconds: float,
) -> Iterable[str]:
    doi_col, title_col = columns
    doi = (row.get(doi_col) or "").strip()
    title = (row.get(title_col) or "").strip()
    queries: List[str] = []
    if doi:
        queries.append(f'"{doi}" pdf')
        queries.append(f'"{doi}" filetype:pdf')
    if title:
        queries.append(f'"{title}" pdf')
    seen = set()
    yielded = 0
    for query in queries:
        for url in _serper_search(api_key, query):
            lowered = url.lower()
            if any(keyword in lowered for keyword in BLOCKED_HOST_KEYWORDS):
                continue
            if url in seen:
                continue
            seen.add(url)
            yield url
            yielded += 1
            if yielded >= MAX_URLS_PER_ROW:
                return
            time.sleep(delay_seconds)


def _process_row(
    api_key: str,
    session: requests.Session,
    config,
    row: dict,
    column_titles: Tuple[str, str, str, str, str],
    delay_seconds: float,
    dry_run: bool,
) -> Optional[Tuple[str, Optional[str]]]:
    doi_col, title_col, status_col, path_col, notes_col = column_titles
    title = (row.get(title_col) or row.get(doi_col) or "untitled").strip()
    slug = slugify(title)
    dest = config.full_text_dir / f"{slug}.pdf"

    if dest.exists():
        return _relative_to_root(dest, config.root_dir), None

    for url in _candidate_urls(api_key, row, (doi_col, title_col), delay_seconds):
        pdf_url = url if _looks_like_pdf(url) else _find_pdf_in_page(session, url)
        if not pdf_url:
            continue
        print(f"  Attempting {pdf_url}")
        if dry_run:
            return _relative_to_root(dest, config.root_dir), "dry-run"
        saved_from = _download_pdf(session, pdf_url, dest)
        if saved_from:
            from urllib.parse import urlparse

            parsed = urlparse(saved_from)
            host = parsed.netloc or parsed.path.split("/")[0]
            print(f"  Saved {dest}")
            return _relative_to_root(dest, config.root_dir), host
    return None


def main() -> int:
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        print("SERPER_API_KEY environment variable must be set.", file=sys.stderr)
        return 1

    args = _parse_args()
    delay_seconds = args.delay if args.delay is not None else float(os.getenv("SERPER_DELAY", DEFAULT_DELAY))
    max_rows_env = int(os.getenv("SERPER_MAX_ROWS", "0"))
    max_rows = args.max_rows if args.max_rows is not None else max_rows_env

    config = load_project_config(args.config)
    config.full_text_dir.mkdir(parents=True, exist_ok=True)

    csv_path = config.csv_path
    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}", file=sys.stderr)
        return 1

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    column_titles: Tuple[str, str, str, str, str] = (
        config.columns.doi,
        config.columns.title,
        config.columns.status,
        config.columns.path,
        config.columns.notes,
    )

    session = _session()

    updated = 0
    processed = 0
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        status = (row.get(column_titles[2]) or "").strip().lower()
        if status == "downloaded":
            continue
        doi = row.get(column_titles[0])
        title = row.get(column_titles[1])
        print(f"Processing [{index}/{total}]: {title} ({doi})")
        result = _process_row(
            api_key,
            session,
            config,
            row,
            column_titles,
            delay_seconds,
            args.dry_run,
        )
        if result:
            pdf_path, source_host = result
            row[column_titles[2]] = "downloaded"
            row[column_titles[3]] = pdf_path
            existing = row.get(column_titles[4]) or ""
            extras: List[str] = []
            if source_host and source_host != "dry-run":
                extras.append(f"Serper ({source_host})")
            elif source_host == "dry-run":
                extras.append("Serper (dry-run)")
            if existing:
                extras.append(existing)
            row[column_titles[4]] = "; ".join(dict.fromkeys(filter(None, extras)))
            updated += 1
            if not args.dry_run:
                with csv_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
        else:
            print("  No accessible PDF located via Serper.")
            if not args.dry_run and index % 5 == 0:
                with csv_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
        processed += 1
        if max_rows and processed >= max_rows:
            break

    if args.dry_run:
        print("Dry run complete; CSV not modified.")
        return 0

    if updated:
        print(f"Updated CSV with {updated} new PDFs.")
    else:
        print("No new PDFs found via Serper.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
