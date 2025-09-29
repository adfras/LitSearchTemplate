#!/usr/bin/env python3
"""Harvest OpenAlex search results into the project dataset."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests
import unicodedata


try:  # Allow running as module or script
    from .common import DEFAULT_CONFIG_PATH, load_project_config
except ImportError:  # pragma: no cover - fallback for direct execution
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.append(str(CURRENT_DIR))
    from common import DEFAULT_CONFIG_PATH, load_project_config  # type: ignore  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SEARCH_PLAN = ROOT_DIR / "config" / "search_plan.json"
OPENALEX_WORKS_ENDPOINT = "https://api.openalex.org/works"


@dataclass
class SearchQuery:
    label: str
    search: str
    filters: Optional[str]
    sort: str
    per_page: int
    max_results: int


@dataclass
class NearRequirement:
    terms: List[str]
    window: int


@dataclass
class SearchPlan:
    provider: str
    queries: List[SearchQuery]
    year_min: Optional[int]
    year_max: Optional[int]
    min_citations: int
    top_n: Optional[int]
    sleep: float
    must_include: List[str]
    must_include_near: List[NearRequirement]


@dataclass
class Record:
    openalex_id: str
    doi: Optional[str]
    doi_url: Optional[str]
    title: str
    publication_year: Optional[int]
    cited_by_count: int
    landing_page_url: Optional[str]
    authors: List[str]
    abstract: str
    source_queries: List[str]


def _sanitize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", value)
    ascii_bytes = normalized.encode("ascii", "ignore")
    collapsed = ascii_bytes.decode("ascii").strip()
    return " ".join(collapsed.split())


class OpenAlexError(RuntimeError):
    """Raised when the OpenAlex API returns an unexpected response."""


def _load_search_plan(path: Path | None) -> SearchPlan:
    plan_path = (path or DEFAULT_SEARCH_PLAN).expanduser().resolve()
    if not plan_path.exists():
        raise FileNotFoundError(f"Search plan not found: {plan_path}")
    data = json.loads(plan_path.read_text(encoding="utf-8"))
    provider = str(data.get("provider", "openalex"))
    if provider.lower() != "openalex":
        raise ValueError("Only the 'openalex' provider is currently supported.")
    raw_queries = data.get("queries")
    if not isinstance(raw_queries, list) or not raw_queries:
        raise ValueError("Search plan must include a non-empty 'queries' list.")
    queries = []
    for item in raw_queries:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or item.get("search") or "query")
        search = item.get("search")
        if not isinstance(search, str) or not search.strip():
            raise ValueError("Each query must define a non-empty 'search' string.")
        filters = item.get("filters")
        filters = str(filters).strip() if filters else None
        sort = str(item.get("sort") or "cited_by_count:desc")
        per_page = int(item.get("per_page") or 25)
        per_page = max(1, min(per_page, 200))
        max_results = int(item.get("max_results") or per_page)
        max_results = max(1, max_results)
        queries.append(
            SearchQuery(
                label=label,
                search=search.strip(),
                filters=filters,
                sort=sort.strip(),
                per_page=per_page,
                max_results=max_results,
            )
        )
    year_min = data.get("year_min")
    year_max = data.get("year_max")
    min_citations = int(data.get("min_citations") or 0)
    top_n = data.get("top_n")
    top_n = int(top_n) if top_n is not None else None
    sleep = float(data.get("sleep") or 0.5)
    must_include_raw = data.get("must_include") or []
    must_include = [str(term).strip().lower() for term in must_include_raw if str(term).strip()]

    near_raw = data.get("must_include_near") or []
    must_include_near: List[NearRequirement] = []
    for entry in near_raw:
        if isinstance(entry, dict):
            raw_terms = entry.get("terms")
            window = int(entry.get("window") or 0)
        else:
            raw_terms = entry
            window = 0
        if isinstance(raw_terms, str):
            terms = [raw_terms]
        else:
            terms = list(raw_terms) if isinstance(raw_terms, (list, tuple)) else []
        cleaned = [str(term).strip().lower() for term in terms if str(term).strip()]
        if len(cleaned) < 2:
            continue
        window = max(0, int(window) or 0)
        must_include_near.append(NearRequirement(terms=cleaned, window=window))
    return SearchPlan(
        provider=provider,
        queries=queries,
        year_min=int(year_min) if year_min is not None else None,
        year_max=int(year_max) if year_max is not None else None,
        min_citations=max(0, min_citations),
        top_n=top_n,
        sleep=max(0.0, sleep),
        must_include=must_include,
        must_include_near=must_include_near,
    )


def _reconstruct_abstract(abstract_index: Optional[dict]) -> str:
    if not abstract_index:
        return ""
    positions = []
    for word, indices in abstract_index.items():
        for pos in indices:
            positions.append((pos, word))
    if not positions:
        return ""
    positions.sort()
    length = positions[-1][0] + 1
    tokens = [""] * length
    for pos, word in positions:
        tokens[pos] = word
    return " ".join(token for token in tokens if token)


def _extract_record(raw: dict, label: str) -> Record:
    openalex_id = raw.get("id") or ""
    doi_value = raw.get("doi")
    doi_url = None
    if doi_value:
        doi_value = doi_value.lower()
        doi_url = doi_value if doi_value.startswith("http") else f"https://doi.org/{doi_value}"
    title = _sanitize_text(raw.get("display_name") or raw.get("title") or "untitled")
    year = raw.get("publication_year")
    cited_by = int(raw.get("cited_by_count") or 0)
    primary = raw.get("primary_location") or {}
    landing = primary.get("landing_page_url") or primary.get("source_url")
    if not landing:
        landing = doi_url
    authors = []
    for auth in raw.get("authorships") or []:
        name = _sanitize_text(auth.get("author", {}).get("display_name"))
        if name:
            authors.append(name)
    abstract = raw.get("abstract")
    if not abstract:
        abstract = _reconstruct_abstract(raw.get("abstract_inverted_index"))
    abstract = _sanitize_text(abstract)
    return Record(
        openalex_id=str(openalex_id),
        doi=doi_value,
        doi_url=doi_url,
        title=title,
        publication_year=int(year) if year is not None else None,
        cited_by_count=cited_by,
        landing_page_url=landing,
        authors=authors,
        abstract=abstract,
        source_queries=[_sanitize_text(label)],
    )


def _request_openalex(params: Dict[str, str]) -> dict:
    try:
        response = requests.get(OPENALEX_WORKS_ENDPOINT, params=params, timeout=30)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        raise OpenAlexError(f"OpenAlex request failed: {exc}") from exc
    data = response.json()
    if not isinstance(data, dict):
        raise OpenAlexError("Unexpected OpenAlex response structure.")
    return data


def _fetch_query(plan: SearchPlan, query: SearchQuery) -> List[Record]:
    params: Dict[str, str] = {
        "search": query.search,
        "per-page": str(query.per_page),
        "sort": query.sort,
        "cursor": "*",
    }
    if query.filters:
        params["filter"] = query.filters
    results: List[Record] = []
    fetched = 0
    while fetched < query.max_results:
        data = _request_openalex(params)
        works = data.get("results") or []
        for item in works:
            record = _extract_record(item, query.label)
            results.append(record)
            fetched += 1
            if fetched >= query.max_results:
                break
        next_cursor = data.get("meta", {}).get("next_cursor")
        if not next_cursor or fetched >= query.max_results:
            break
        params["cursor"] = next_cursor
        if plan.sleep:
            time.sleep(plan.sleep)
    return results


def _deduplicate(records: Iterable[Record]) -> List[Record]:
    merged: Dict[str, Record] = {}
    for record in records:
        key = record.doi or record.openalex_id
        key = key.lower() if isinstance(key, str) else record.openalex_id
        existing = merged.get(key)
        if existing:
            combined_queries = sorted(set(existing.source_queries + record.source_queries))
            existing.source_queries = combined_queries
            if record.cited_by_count > existing.cited_by_count:
                existing.cited_by_count = record.cited_by_count
            if not existing.abstract and record.abstract:
                existing.abstract = record.abstract
            if not existing.landing_page_url and record.landing_page_url:
                existing.landing_page_url = record.landing_page_url
            if not existing.doi_url and record.doi_url:
                existing.doi_url = record.doi_url
            continue
        merged[key] = record
    return list(merged.values())


def _find_term_positions(text: str, term: str) -> List[int]:
    positions: List[int] = []
    start = 0
    while True:
        index = text.find(term, start)
        if index == -1:
            break
        positions.append(index)
        start = index + max(1, len(term))
    return positions


def _matches_near_requirement(text: str, requirement: NearRequirement) -> bool:
    terms = requirement.terms
    if not terms:
        return True
    occurrences: List[tuple[int, str]] = []
    for term in terms:
        positions = _find_term_positions(text, term)
        if not positions:
            return False
        occurrences.extend((position, term) for position in positions)
    occurrences.sort(key=lambda item: item[0])
    counts = {term: 0 for term in terms}
    left = 0
    for right, (position, term) in enumerate(occurrences):
        counts[term] += 1
        while all(count > 0 for count in counts.values()) and left <= right:
            span = occurrences[right][0] - occurrences[left][0]
            if requirement.window <= 0 or span <= requirement.window:
                return True
            left_term = occurrences[left][1]
            counts[left_term] -= 1
            left += 1
    return False


def _satisfies_near_requirements(text: str, requirements: List[NearRequirement]) -> bool:
    return all(_matches_near_requirement(text, requirement) for requirement in requirements)


def _apply_filters(plan: SearchPlan, records: Iterable[Record]) -> List[Record]:
    filtered: List[Record] = []
    for record in records:
        if plan.year_min is not None and (record.publication_year or 0) < plan.year_min:
            continue
        if plan.year_max is not None and (record.publication_year or 9999) > plan.year_max:
            continue
        if record.cited_by_count < plan.min_citations:
            continue
        if plan.must_include:
            haystack = f"{record.title} {record.abstract}".lower()
            if not all(term in haystack for term in plan.must_include):
                continue
        if plan.must_include_near:
            haystack = f"{record.title} {record.abstract}".lower()
            if not _satisfies_near_requirements(haystack, plan.must_include_near):
                continue
        filtered.append(record)
    return filtered


def _sort_and_limit(plan: SearchPlan, records: List[Record]) -> List[Record]:
    records.sort(key=lambda r: (r.cited_by_count, r.publication_year or 0), reverse=True)
    if plan.top_n is not None:
        return records[: plan.top_n]
    return records


def _write_json(path: Path, data: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, records: List[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Rank",
        "Title",
        "Publication Year",
        "Citations (OpenAlex)",
        "DOI",
        "Landing Page URL",
        "Authors",
        "Source Query",
        "Abstract",
        "Full Text Status",
        "Full Text Path",
        "Full Text Notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, record in enumerate(records, start=1):
            writer.writerow(
                {
                    "Rank": index,
                    "Title": record.title,
                    "Publication Year": record.publication_year or "",
                    "Citations (OpenAlex)": record.cited_by_count,
                    "DOI": record.doi_url or record.doi or "",
                    "Landing Page URL": record.landing_page_url or record.doi_url or "",
                    "Authors": "; ".join(record.authors),
                    "Source Query": "; ".join(record.source_queries),
                    "Abstract": record.abstract,
                    "Full Text Status": "missing",
                    "Full Text Path": "",
                    "Full Text Notes": "",
                }
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Harvest OpenAlex results and rebuild the dataset CSV.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Project config path (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--search-plan",
        type=Path,
        default=None,
        help=f"Search plan path (default: {DEFAULT_SEARCH_PLAN}).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        help="Override the number of records retained after sorting.",
    )
    parser.add_argument(
        "--min-citations",
        type=int,
        help="Override the minimum citation count filter.",
    )
    parser.add_argument(
        "--year-min",
        type=int,
        help="Override the inclusive lower bound for publication year.",
    )
    parser.add_argument(
        "--year-max",
        type=int,
        help="Override the inclusive upper bound for publication year.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = load_project_config(args.config)
    plan = _load_search_plan(args.search_plan or config.search_plan)

    if args.top_n is not None:
        plan.top_n = max(1, args.top_n)
    if args.min_citations is not None:
        plan.min_citations = max(0, args.min_citations)
    if args.year_min is not None:
        plan.year_min = args.year_min
    if args.year_max is not None:
        plan.year_max = args.year_max

    all_records: List[Record] = []
    for query in plan.queries:
        print(f"Querying OpenAlex for '{query.label}' ...")
        fetched = _fetch_query(plan, query)
        print(f"  Retrieved {len(fetched)} records.")
        all_records.extend(fetched)

    if not all_records:
        print("No records retrieved; aborting.")
        return 0

    candidates_path = config.csv_path.with_name(f"{config.csv_path.stem}_candidates.json")
    dedup_path = config.csv_path.with_name(f"{config.csv_path.stem}_dedup.json")

    _write_json(
        candidates_path,
        [
            {
                "openalex_id": record.openalex_id,
                "doi": record.doi,
                "doi_url": record.doi_url,
                "title": record.title,
                "publication_year": record.publication_year,
                "cited_by_count": record.cited_by_count,
                "landing_page_url": record.landing_page_url,
                "authors": record.authors,
                "abstract": record.abstract,
                "source_queries": record.source_queries,
            }
            for record in all_records
        ],
    )

    deduped = _deduplicate(all_records)
    filtered = _apply_filters(plan, deduped)
    ordered = _sort_and_limit(plan, filtered)

    _write_json(
        dedup_path,
        [
            {
                "openalex_id": record.openalex_id,
                "doi": record.doi,
                "doi_url": record.doi_url,
                "title": record.title,
                "publication_year": record.publication_year,
                "cited_by_count": record.cited_by_count,
                "landing_page_url": record.landing_page_url,
                "authors": record.authors,
                "abstract": record.abstract,
                "source_queries": record.source_queries,
            }
            for record in ordered
        ],
    )

    _write_csv(config.csv_path, ordered)
    print(
        "Wrote",
        len(ordered),
        "records to",
        config.csv_path,
        "(candidates:",
        candidates_path,
        "deduplicated:",
        dedup_path,
        ")",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
