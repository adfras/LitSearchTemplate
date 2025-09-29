#!/usr/bin/env python3
"""Download deterministic open-access PDFs defined in the project config."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Tuple

import requests

try:  # Allow running as `python scripts/download_open_pdfs.py`
    from .common import (
        DEFAULT_CONFIG_PATH,
        load_open_access_sources,
        load_project_config,
        slugify,
    )
except ImportError:  # pragma: no cover - fallback for direct execution
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.append(str(CURRENT_DIR))
    from common import (  # type: ignore  # noqa: E402
        DEFAULT_CONFIG_PATH,
        load_open_access_sources,
        load_project_config,
        slugify,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download known open-access PDFs using a DOI→URL mapping.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Path to project config (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=None,
        help="Optional override for the DOI→URL mapping JSON file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without downloading or modifying the CSV.",
    )
    return parser.parse_args()


def _download_pdf(session: requests.Session, url: str, dest: Path) -> None:
    headers = {"Accept": "application/pdf,*/*;q=0.8"}
    if "mdpi.com" in url:
        headers["Referer"] = url.split("/pdf", 1)[0]
    response = session.get(url, timeout=60, headers=headers)
    response.raise_for_status()
    dest.write_bytes(response.content)


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> int:
    args = _parse_args()
    config = load_project_config(args.config)
    sources_path = args.sources or config.open_access_sources
    sources = load_open_access_sources(sources_path)

    if not sources:
        print(f"No open-access sources defined at {sources_path}")
        return 0

    csv_path = config.csv_path
    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}", file=sys.stderr)
        return 1

    config.full_text_dir.mkdir(parents=True, exist_ok=True)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            print("CSV is missing a header row.", file=sys.stderr)
            return 1
        rows = list(reader)

    column_titles: Tuple[str, str, str, str, str] = (
        config.columns.doi,
        config.columns.title,
        config.columns.status,
        config.columns.path,
        config.columns.notes,
    )

    session = requests.Session()
    session.headers.update({"User-Agent": "curl/8.4"})

    total = 0
    updated = 0
    for row in rows:
        doi = (row.get(column_titles[0]) or "").strip()
        if not doi:
            continue
        entry = sources.get(doi)
        if not entry:
            continue
        total += 1
        url = entry.get("url")
        if not isinstance(url, str) or not url.strip():
            print(f"Skipping {doi}: missing URL in sources mapping.")
            continue
        title = (row.get(column_titles[1]) or doi or "untitled").strip()
        dest = config.full_text_dir / f"{slugify(title)}.pdf"
        if dest.exists():
            row[column_titles[2]] = "downloaded"
            row[column_titles[3]] = _relative_to_root(dest, config.root_dir)
            note = row.get(column_titles[4]) or ""
            extra = entry.get("note")
            if isinstance(extra, str) and extra and extra not in note:
                row[column_titles[4]] = (note + "; " if note else "") + extra
            continue
        print(f"Downloading {doi} -> {dest}")
        if args.dry_run:
            continue
        try:
            _download_pdf(session, url.strip(), dest)
        except Exception as exc:  # noqa: BLE001
            print(f"  Failed: {exc}")
            continue
        row[column_titles[2]] = "downloaded"
        row[column_titles[3]] = _relative_to_root(dest, config.root_dir)
        base_note = (row.get(column_titles[4]) or "").strip()
        extra_note = (entry.get("note") or "").strip() if isinstance(entry.get("note"), str) else ""
        notes = [text for text in (base_note, extra_note) if text]
        row[column_titles[4]] = "; ".join(dict.fromkeys(notes))
        updated += 1

    if args.dry_run:
        print("Dry run complete; no files written.")
        return 0

    if updated:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Updated CSV with {updated} downloaded entries (out of {total} candidates).")
    else:
        print("No new PDFs downloaded; CSV left untouched.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
