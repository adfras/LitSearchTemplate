#!/usr/bin/env python3
"""End-to-end orchestrator for the literature search pipeline.

Usage example:

    python -m scripts.run_pipeline "individual differences for virtual reality social interactions" \
        --year-min 2018 --year-max 2025 --min-citations 5 --top-n 150

This command:

1. Updates ``config/search_plan.json`` with generated OpenAlex queries for the topic.
2. Runs the harvesting, PDF discovery, and reference export scripts in sequence.
3. Summarises differences versus the previous dataset (row/DOI changes, full-text coverage).
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

from .common import DEFAULT_CONFIG_PATH, load_project_config

DEFAULT_FILTER = "type:article,language:en"
DEFAULT_PER_PAGE = 200
DEFAULT_MAX_RESULTS = 200


PROFILES = {
    "social-vr-individual-diff": [
        {
            "label": "vr-social-indivdiff",
            "search": '"virtual reality" "social interaction" "individual difference"',
            "filters": DEFAULT_FILTER,
            "sort": "cited_by_count:desc",
            "per_page": DEFAULT_PER_PAGE,
            "max_results": DEFAULT_MAX_RESULTS,
        },
        {
            "label": "social-vr-personality",
            "search": '"social virtual reality" (personality OR trait OR "individual differences")',
            "filters": DEFAULT_FILTER,
            "sort": "relevance_score:desc",
            "per_page": DEFAULT_PER_PAGE,
            "max_results": DEFAULT_MAX_RESULTS,
        },
        {
            "label": "vr-social-presence",
            "search": '"virtual reality" "social presence" personality',
            "filters": DEFAULT_FILTER,
            "sort": "relevance_score:desc",
            "per_page": DEFAULT_PER_PAGE,
            "max_results": DEFAULT_MAX_RESULTS,
        },
    ],
}

def _run(*cmd: Sequence[str | Path], timeout: int | None = None) -> None:
    """Run a subprocess, echoing the command for transparency."""

    printable = " ".join(str(part) for part in cmd)
    print(f"→ {printable}")
    subprocess.run([str(part) for part in cmd], check=True, timeout=timeout)


def _load_rows(csv_path: Path) -> List[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _normalise_doi(raw: str | None) -> str | None:
    if not raw:
        return None
    doi = raw.strip()
    if not doi:
        return None
    prefix = "https://doi.org/"
    if doi.lower().startswith(prefix):
        return doi.lower()[len(prefix) :]
    return doi.lower()


def _count_missing(rows: Iterable[dict]) -> int:
    return sum(1 for row in rows if (row.get("Full Text Status") or "").lower() != "downloaded")


def _generate_queries(
    topic: str,
    extra_queries: Sequence[str],
    profile: str | None,
    required_terms: Sequence[str],
    required_near: Sequence[Sequence[str]],
) -> List[dict]:
    if profile and profile in PROFILES:
        return PROFILES[profile]

    def _prepare_terms(terms: Sequence[str]) -> List[str]:
        prepared: List[str] = []
        for term in terms:
            cleaned = term.strip()
            if not cleaned:
                continue
            prepared.append(f'"{cleaned}"' if any(ch.isspace() for ch in cleaned) else cleaned)
        return prepared

    cleaned = topic.strip()
    lowered = cleaned.lower()
    if all(keyword in lowered for keyword in ("virtual reality", "social", "individual", "difference")):
        return PROFILES["social-vr-individual-diff"]
    queries: List[dict] = []

    def add_query(label: str, search: str, sort: str = "cited_by_count:desc") -> None:
        queries.append(
            {
                "label": label,
                "search": search,
                "filters": DEFAULT_FILTER,
                "sort": sort,
                "per_page": DEFAULT_PER_PAGE,
                "max_results": DEFAULT_MAX_RESULTS,
            }
        )

    if cleaned:
        add_query("primary", f'"{cleaned}"', "relevance_score:desc")
        add_query("broad", cleaned, "cited_by_count:desc")

    if "virtual reality" in lowered:
        remainder = lowered.replace("virtual reality", "").strip()
        if remainder:
            add_query("vr-or", f'("virtual reality" OR VR) {remainder}', "cited_by_count:desc")
        add_query("vr-social", '("social virtual reality" OR "social VR")', "relevance_score:desc")

    if "social" in lowered and "virtual reality" not in lowered:
        add_query("social", f'"social" {cleaned}', "relevance_score:desc")

    if "individual" in lowered and "difference" in lowered:
        add_query("individual-diff", cleaned.replace("individual", '"individual"'), "relevance_score:desc")

    # User-specified extras
    for index, query in enumerate(extra_queries, start=1):
        label = f"extra-{index}"
        add_query(label, query, "relevance_score:desc")

    required_tokens = _prepare_terms(required_terms)
    if required_tokens:
        add_query("required-all", " ".join(required_tokens), "relevance_score:desc")
        if len(required_tokens) > 1:
            add_query("required-and", " AND ".join(required_tokens), "relevance_score:desc")
            if len(required_tokens) > 2:
                for combo_index, combo in enumerate(itertools.combinations(required_tokens, 2), start=1):
                    add_query(
                        f"required-pair-{combo_index}",
                        " AND ".join(combo),
                        "relevance_score:desc",
                    )

    for idx, requirement in enumerate(required_near, start=1):
        near_tokens = _prepare_terms(requirement)
        if len(near_tokens) >= 2:
            add_query(
                f"near-{idx}",
                " AND ".join(near_tokens),
                "relevance_score:desc",
            )

    # De-duplicate while keeping order
    unique = []
    seen = set()
    for entry in queries:
        key_data = {k: entry[k] for k in entry if k != "label"}
        key = json.dumps(key_data, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)
    return unique


def _write_search_plan(
    topic: str,
    config_path: Path,
    year_min: int | None,
    year_max: int | None,
    min_citations: int,
    top_n: int | None,
    required_terms: Sequence[str],
    required_near: Sequence[Sequence[str]],
    near_window: int,
    extra_queries: Sequence[str],
    profile: str | None,
    providers: Sequence[str] | None,
    provider_thresholds: dict[str, int] | None,
) -> Path:
    config = load_project_config(config_path)
    plan_path = config.search_plan
    queries = _generate_queries(
        topic,
        extra_queries,
        profile,
        required_terms,
        required_near,
    )
    if not queries:
        raise ValueError("No queries generated; supply topic or --extra-query.")

    must_include = [term.strip().lower() for term in required_terms if term.strip()]
    if not must_include and "virtual reality" in topic.lower():
        must_include = ["virtual reality"]

    near_requirements = []
    clean_window = max(0, near_window)
    for requirement in required_near:
        cleaned = [term.strip().lower() for term in requirement if term.strip()]
        if len(cleaned) >= 2:
            near_requirements.append({"terms": cleaned, "window": clean_window})

    provider_list = [prov.strip().lower() for prov in providers or [] if prov.strip()]
    if not provider_list:
        provider_list = [
            "openalex",
            "crossref",
            "semantic_scholar",
            "serper_scholar",
            "core",
            "arxiv",
            "doaj",
        ]

    plan = {
        "provider": "openalex",
        "queries": queries,
        "year_min": year_min,
        "year_max": year_max,
        "min_citations": min_citations,
        "top_n": top_n,
        "sleep": 0.5,
        "must_include": must_include,
        "must_include_near": near_requirements,
        "providers": provider_list,
    }
    if provider_thresholds:
        plan["provider_min_citations"] = provider_thresholds
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan_path


def _ensure_full_text(config_path: Path, rounds: int) -> None:
    config = load_project_config(config_path)
    csv_path = config.csv_path
    for attempt in range(1, rounds + 1):
        rows = _load_rows(csv_path)
        missing = _count_missing(rows)
        if missing == 0:
            print("All rows already have full text; skipping Serper.")
            return
        print(f"Full-text discovery round {attempt} (remaining: {missing})")
        _run(sys.executable, "-m", "scripts.serper_download_pdfs", "--config", config_path)
        new_missing = _count_missing(_load_rows(csv_path))
        if new_missing == 0:
            print("✓ All PDFs located via Serper.")
            return
        if new_missing >= missing:
            print("No further progress locating PDFs; stopping early.")
            return
    print("Serper rounds exhausted; some PDFs may still require manual retrieval.")


def _summarise_changes(old_rows: List[dict], new_rows: List[dict]) -> None:
    old_dois = {_normalise_doi(row.get("DOI")) for row in old_rows}
    new_dois = {_normalise_doi(row.get("DOI")) for row in new_rows}
    old_dois.discard(None)
    new_dois.discard(None)

    added = sorted(new_dois - old_dois)
    removed = sorted(old_dois - new_dois)

    downloaded = sum(1 for row in new_rows if (row.get("Full Text Status") or "").lower() == "downloaded")
    missing = len(new_rows) - downloaded

    print("\n=== Pipeline summary ===")
    print(f"Rows: {len(new_rows)} (was {len(old_rows)})")
    print(f"Full-text coverage: {downloaded}/{len(new_rows)} downloaded; {missing} missing")
    if added:
        print(f"New DOIs ({len(added)}): {', '.join(added[:5])}{'…' if len(added) > 5 else ''}")
    if removed:
        print(f"Removed DOIs ({len(removed)}): {', '.join(removed[:5])}{'…' if len(removed) > 5 else ''}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the full literature pipeline for a topic/question.",
    )
    parser.add_argument("topic", help="Natural-language topic or question.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        type=Path,
        help=f"Project config path (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument("--year-min", type=int, default=2018, help="Lower publication year bound (inclusive).")
    parser.add_argument("--year-max", type=int, default=2025, help="Upper publication year bound (inclusive).")
    parser.add_argument("--min-citations", type=int, default=5, help="Minimum citation count filter.")
    parser.add_argument(
        "--top-n",
        type=int,
        default=150,
        help="Keep only the top-N records after sorting (0 = no limit).",
    )
    parser.add_argument(
        "--require",
        action="append",
        default=[],
        help="Keyword that must appear in title/abstract (can be supplied multiple times).",
    )
    parser.add_argument(
        "--require-near",
        action="append",
        nargs=2,
        metavar=("TERM_A", "TERM_B"),
        default=[],
        help="Ensure the given pair of terms appear within the proximity window in title/abstract (repeatable).",
    )
    parser.add_argument(
        "--near-window",
        type=int,
        default=120,
        help="Maximum character span for --require-near matches (0 disables the proximity constraint).",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES.keys()),
        help="Optional pre-defined query template to use.",
    )
    parser.add_argument(
        "--extra-query",
        action="append",
        default=[],
        help="Additional OpenAlex search string(s) to include alongside the generated ones.",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        help=(
            "Which data providers to query (defaults to openalex, crossref, "
            "semantic_scholar, serper_scholar, core, arxiv, doaj)."
        ),
    )
    parser.add_argument(
        "--provider-min-citations",
        action="append",
        default=[],
        metavar="PROVIDER=N",
        help=(
            "Optional provider-specific citation floors (e.g. arxiv=10). "
            "Can be supplied multiple times."
        ),
    )
    parser.add_argument(
        "--serper-rounds",
        type=int,
        default=3,
        help="How many passes to attempt when locating PDFs via Serper (default: 3).",
    )

    args = parser.parse_args(argv)

    top_n = args.top_n if args.top_n > 0 else None

    old_rows = _load_rows(load_project_config(args.config).csv_path)

    provider_thresholds: dict[str, int] = {}
    for item in args.provider_min_citations:
        if not item or "=" not in item:
            print(f"Ignoring invalid --provider-min-citations entry: {item}")
            continue
        provider, value = item.split("=", 1)
        provider_clean = provider.strip().lower()
        if not provider_clean:
            continue
        try:
            provider_thresholds[provider_clean] = max(0, int(value))
        except ValueError:
            print(f"Ignoring invalid citation threshold '{value}' for provider '{provider}'.")

    plan_path = _write_search_plan(
        topic=args.topic,
        config_path=args.config,
        year_min=args.year_min,
        year_max=args.year_max,
        min_citations=args.min_citations,
        top_n=top_n,
        required_terms=args.require,
        required_near=args.require_near,
        near_window=args.near_window,
        extra_queries=args.extra_query,
        profile=args.profile,
        providers=args.providers,
        provider_thresholds=provider_thresholds or None,
    )
    print(f"Updated search plan at {plan_path}")

    # Harvest latest candidates
    _run(sys.executable, "-m", "scripts.collect_openalex", "--config", args.config)

    # Deterministic downloads first (if mapping provided)
    _run(sys.executable, "-m", "scripts.download_open_pdfs", "--config", args.config)

    # Serper-assisted discovery (multiple passes if needed)
    _ensure_full_text(args.config, rounds=args.serper_rounds)

    # Bibliography export
    _run(sys.executable, "-m", "scripts.fetch_references", "--config", args.config)

    new_rows = _load_rows(load_project_config(args.config).csv_path)
    _summarise_changes(old_rows, new_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
