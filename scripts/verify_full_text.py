#!/usr/bin/env python3
"""Verify downloaded PDFs against the target topic using OpenAI."""
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from openai import OpenAI
from pypdf import PdfReader

try:  # Support both module and script execution
    from .common import load_project_config
except ImportError:  # pragma: no cover
    from common import load_project_config  # type: ignore


DEFAULT_MODEL = "gpt-5-chat-latest"
MAX_EXCERPT_CHARS = 6000
PAGES_TO_SAMPLE = 12
VERIFICATION_FIELD = "Verification Status"
NOTES_FIELD = "Verification Notes"

PROMPT_TEMPLATE = (
    "You are reviewing a research paper to confirm it is relevant to the bibliographic search topic below.\n\n"
    "Target topic:\n{topic}\n\n"
    "Extracted excerpts from the PDF:\n---\n{excerpt}\n---\n\n"
    "Task: Decide whether this paper makes a substantial contribution to the target topic. "
    "Answer in exactly one of the following formats:\n"
    "YES - <one sentence justification>\n"
    "NO - <one sentence justification>"
)


@dataclass(frozen=True)
class VerificationResult:
    verdict: str
    notes: str


def _sample_pdf_text(pdf_path: Path) -> str:
    """Extract text slices from across the document to give the LLM enough context."""

    try:
        reader = PdfReader(pdf_path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to open PDF {pdf_path}: {exc}") from exc

    total_pages = len(reader.pages)
    if total_pages == 0:
        return ""

    indices = {0, total_pages - 1}
    if total_pages > 1:
        step = max(1, total_pages // max(3, PAGES_TO_SAMPLE))
        for idx in range(0, min(total_pages, PAGES_TO_SAMPLE * step), step):
            indices.add(idx)

    excerpts: List[str] = []
    accumulated = 0
    for idx in sorted(indices):
        if idx >= total_pages:
            continue
        try:
            text = reader.pages[idx].extract_text() or ""
        except Exception:  # noqa: BLE001
            text = ""
        text = text.strip()
        if text:
            excerpts.append(text)
            accumulated += len(text)
            if accumulated >= MAX_EXCERPT_CHARS:
                break

    return ("\n\n".join(excerpts))[:MAX_EXCERPT_CHARS]


def _call_openai(client: OpenAI, model: str, topic: str, excerpt: str) -> VerificationResult:
    if not excerpt.strip():
        return VerificationResult("NO", "Unable to extract text from PDF.")

    prompt = PROMPT_TEMPLATE.format(topic=topic, excerpt=excerpt)
    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=0,
    )
    answer = response.output_text.strip()
    upper = answer.upper()
    if upper.startswith("YES"):
        return VerificationResult("YES", answer)
    if upper.startswith("NO"):
        return VerificationResult("NO", answer)
    return VerificationResult("NO", f"Unexpected response: {answer}")


def _load_rows(csv_path: Path) -> List[dict]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _write_rows(csv_path: Path, fieldnames: Iterable[str], rows: Iterable[dict]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def verify_pdfs(topic: str, model: str, config_path: Path | None, dry_run: bool) -> None:
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("OPENAI_API_KEY environment variable is required.")

    config = load_project_config(config_path)
    csv_path = config.csv_path
    pdf_dir = config.full_text_dir

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not pdf_dir.exists():
        raise FileNotFoundError(f"Full-text directory not found: {pdf_dir}")

    rows = _load_rows(csv_path)
    if not rows:
        print("No rows to verify; exiting.")
        return

    fieldnames = list(rows[0].keys())
    if VERIFICATION_FIELD not in fieldnames:
        fieldnames.append(VERIFICATION_FIELD)
    if NOTES_FIELD not in fieldnames:
        fieldnames.append(NOTES_FIELD)

    client = OpenAI()

    verified = 0
    rejected = 0

    for row in rows:
        pdf_rel = row.get(config.columns.path, "").strip()
        if not pdf_rel:
            continue
        pdf_path = pdf_dir / Path(pdf_rel).name
        if not pdf_path.exists():
            continue

        try:
            excerpt = _sample_pdf_text(pdf_path)
        except Exception as exc:  # noqa: BLE001
            row[VERIFICATION_FIELD] = "NO"
            row[NOTES_FIELD] = f"Extraction failed: {exc}"
            rejected += 1
            continue

        result = _call_openai(client, model, topic, excerpt)
        row[VERIFICATION_FIELD] = result.verdict
        row[NOTES_FIELD] = result.notes

        if result.verdict == "YES":
            verified += 1
        else:
            rejected += 1

        if dry_run:
            print(f"[DRY-RUN] {pdf_path.name}: {result.verdict} -> {result.notes}")
        else:
            print(f"{pdf_path.name}: {result.verdict} -> {result.notes}")

    if not dry_run:
        _write_rows(csv_path, fieldnames, rows)

    print(f"Verification complete: YES={verified}, NO={rejected}.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify downloaded PDFs with OpenAI for topical relevance.",
    )
    parser.add_argument("topic", help="Target topic description for the verifier prompt.")
    parser.add_argument("--config", type=Path, help="Custom project config (defaults to config/project.json).")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model to use (default: {DEFAULT_MODEL}).")
    parser.add_argument("--dry-run", action="store_true", help="Run without writing CSV changes.")
    args = parser.parse_args()

    try:
        verify_pdfs(topic=args.topic, model=args.model, config_path=args.config, dry_run=args.dry_run)
    except Exception as exc:  # noqa: BLE001
        print(f"Verification failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
