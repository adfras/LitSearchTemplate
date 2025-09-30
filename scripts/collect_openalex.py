#!/usr/bin/env python3
"""Harvest scholarly search results into the project dataset."""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests


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
SEMANTIC_SCHOLAR_ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/search"
SERPER_SCHOLAR_ENDPOINT = "https://google.serper.dev/scholar"
CROSSREF_WORKS_ENDPOINT = "https://api.crossref.org/works"
CORE_SEARCH_ENDPOINT = "https://core.ac.uk/api-v2/articles/search/"
ARXIV_API_ENDPOINT = "http://export.arxiv.org/api/query"
DOAJ_SEARCH_ENDPOINT = "https://doaj.org/api/search/articles/"
SEMANTIC_SCHOLAR_MIN_INTERVAL = 3.2  # seconds between unauthenticated requests


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
    providers: List[str]
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
    record_id: str
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


def _strip_html_tags(value: str) -> str:
    if not value:
        return ""
    return re.sub(r"<[^>]+>", " ", value)


def _normalise_doi_value(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    cleaned = doi.strip().lower()
    prefix = "https://doi.org/"
    if cleaned.startswith(prefix):
        cleaned = cleaned[len(prefix) :]
    return cleaned or None


def _create_record(
    record_id: str,
    doi: Optional[str],
    title: str,
    year: Optional[int],
    citations: int,
    landing_page: Optional[str],
    authors: Iterable[str],
    abstract: str,
    source_label: str,
) -> Record:
    doi_norm = _normalise_doi_value(doi)
    doi_url = None
    if doi_norm:
        doi_url = f"https://doi.org/{doi_norm}"
    author_list = [_sanitize_text(name) for name in authors if _sanitize_text(name)]
    return Record(
        record_id=record_id,
        doi=doi_norm,
        doi_url=doi_url,
        title=_sanitize_text(title) or "untitled",
        publication_year=year,
        cited_by_count=max(0, citations),
        landing_page_url=landing_page or doi_url,
        authors=author_list,
        abstract=_sanitize_text(abstract),
        source_queries=[_sanitize_text(source_label) or source_label],
    )


def _load_search_plan(path: Path | None) -> SearchPlan:
    plan_path = (path or DEFAULT_SEARCH_PLAN).expanduser().resolve()
    if not plan_path.exists():
        raise FileNotFoundError(f"Search plan not found: {plan_path}")
    data = json.loads(plan_path.read_text(encoding="utf-8"))
    providers_raw = data.get("providers")
    providers: List[str]
    if isinstance(providers_raw, list) and providers_raw:
        providers = [str(item).strip().lower() for item in providers_raw if str(item).strip()]
    else:
        provider = str(data.get("provider", "openalex")).strip().lower() or "openalex"
        providers = [provider]
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
    if not providers:
        providers = ["openalex"]
    return SearchPlan(
        providers=providers,
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
    title = _sanitize_text(raw.get("display_name") or raw.get("title") or "untitled")
    year = raw.get("publication_year")
    cited_by = int(raw.get("cited_by_count") or 0)
    primary = raw.get("primary_location") or {}
    landing = primary.get("landing_page_url") or primary.get("source_url")
    authors = []
    for auth in raw.get("authorships") or []:
        name = _sanitize_text(auth.get("author", {}).get("display_name"))
        if name:
            authors.append(name)
    abstract = raw.get("abstract")
    if not abstract:
        abstract = _reconstruct_abstract(raw.get("abstract_inverted_index"))
    return _create_record(
        record_id=str(openalex_id) or f"openalex:{title}"[:100],
        doi=doi_value.lower() if isinstance(doi_value, str) else None,
        title=title,
        year=int(year) if year is not None else None,
        citations=cited_by,
        landing_page=landing,
        authors=authors,
        abstract=abstract or "",
        source_label=f"openalex:{label}",
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


def _fetch_openalex(plan: SearchPlan, query: SearchQuery) -> List[Record]:
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


def _fetch_crossref(plan: SearchPlan, query: SearchQuery) -> List[Record]:
    mailto = os.environ.get("CROSSREF_MAILTO")
    headers = {
        "User-Agent": f"LitSearchTemplate/1.0 ({mailto})" if mailto else "LitSearchTemplate/1.0",
    }
    results: List[Record] = []
    retrieved = 0
    while retrieved < query.max_results:
        rows = min(query.per_page, query.max_results - retrieved)
        params = {
            "query": query.search,
            "rows": rows,
            "offset": retrieved,
            "select": "DOI,title,author,issued,abstract,is-referenced-by-count,URL",
        }
        filters: List[str] = []
        if plan.year_min is not None:
            filters.append(f"from-pub-date:{plan.year_min}-01-01")
        if plan.year_max is not None:
            filters.append(f"until-pub-date:{plan.year_max}-12-31")
        if filters:
            params["filter"] = ",".join(filters)
        try:
            response = requests.get(CROSSREF_WORKS_ENDPOINT, params=params, headers=headers, timeout=30)
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            print(f"Crossref request failed for '{query.label}': {exc}")
            break
        payload = response.json()
        items = payload.get("message", {}).get("items", [])
        if not items:
            break
        for item in items:
            title_values = item.get("title") or []
            title = title_values[0] if title_values else ""
            doi = item.get("DOI")
            landing = item.get("URL")
            year = None
            date_parts = item.get("issued", {}).get("date-parts") or []
            if date_parts and isinstance(date_parts[0], list) and date_parts[0]:
                year = date_parts[0][0]
            citations = int(item.get("is-referenced-by-count") or 0)
            abstract_raw = item.get("abstract") or ""
            abstract_text = html.unescape(_strip_html_tags(abstract_raw))
            authors_raw = item.get("author") or []
            authors = []
            for person in authors_raw:
                given = person.get("given") or ""
                family = person.get("family") or ""
                full = f"{given} {family}".strip()
                if not full and person.get("name"):
                    full = person.get("name")
                if full:
                    authors.append(full)
            record_id = f"crossref:{doi.lower()}" if doi else f"crossref:{title[:50]}:{retrieved}"
            results.append(
                _create_record(
                    record_id=record_id,
                    doi=doi,
                    title=title,
                    year=int(year) if year else None,
                    citations=citations,
                    landing_page=landing,
                    authors=authors,
                    abstract=abstract_text,
                    source_label=f"crossref:{query.label}",
                )
            )
        retrieved += len(items)
        if len(items) < rows:
            break
        if plan.sleep:
            time.sleep(plan.sleep)
    return results


def _fetch_semantic_scholar(plan: SearchPlan, query: SearchQuery) -> List[Record]:
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    headers = {
        "User-Agent": "LitSearchTemplate/1.0",
    }
    if api_key:
        headers["x-api-key"] = api_key
    results: List[Record] = []
    offset = 0
    min_sleep = max(plan.sleep, 3.1)
    last_request = getattr(_fetch_semantic_scholar, "_last_request", 0.0)
    while offset < query.max_results:
        limit = min(query.per_page, query.max_results - offset, 80)
        params = {
            "query": query.search,
            "offset": offset,
            "limit": limit,
            "fields": "title,abstract,year,citationCount,externalIds,url,authors,openAccessPdf",
        }
        for attempt in range(3):
            wait_for = SEMANTIC_SCHOLAR_MIN_INTERVAL - (time.time() - last_request)
            if wait_for > 0:
                time.sleep(wait_for)
            try:
                response = requests.get(SEMANTIC_SCHOLAR_ENDPOINT, params=params, headers=headers, timeout=30)
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after and retry_after.isdigit() else 45.0
                    print(f"Semantic Scholar rate limit hit for '{query.label}', skipping remaining pages (set SEMANTIC_SCHOLAR_API_KEY for higher quota).")
                    last_request = time.time()
                    _fetch_semantic_scholar._last_request = last_request
                    return results
                response.raise_for_status()
                last_request = time.time()
            except Exception as exc:  # noqa: BLE001
                if attempt < 2:
                    backoff = min_sleep * (attempt + 1)
                    print(f"Semantic Scholar request error for '{query.label}' (attempt {attempt + 1}): {exc}; retrying in {backoff:.1f}s")
                    time.sleep(backoff)
                    continue
                print(f"Semantic Scholar request failed for '{query.label}': {exc}")
                _fetch_semantic_scholar._last_request = last_request
                return results
            break
        else:
            _fetch_semantic_scholar._last_request = last_request
            return results
        payload = response.json()
        papers = payload.get("data") or []
        if not papers:
            break
        for paper in papers:
            title = paper.get("title") or ""
            year = paper.get("year")
            citations = int(paper.get("citationCount") or 0)
            external_ids = paper.get("externalIds") or {}
            doi = external_ids.get("DOI") or paper.get("doi")
            url = None
            if isinstance(paper.get("openAccessPdf"), dict):
                url = paper["openAccessPdf"].get("url")
            url = url or paper.get("url")
            authors_raw = paper.get("authors") or []
            authors = [author.get("name") for author in authors_raw if author.get("name")]
            abstract = paper.get("abstract") or ""
            record_id = paper.get("paperId") or f"semanticscholar:{doi or title[:50]}"
            results.append(
                _create_record(
                    record_id=f"semanticscholar:{record_id}",
                    doi=doi,
                    title=title,
                    year=int(year) if year else None,
                    citations=citations,
                    landing_page=url,
                    authors=authors,
                    abstract=abstract,
                    source_label=f"semantic_scholar:{query.label}",
                )
            )
        offset += len(papers)
        # respect public rate limits by keeping only the first page without a key
        break
    _fetch_semantic_scholar._last_request = last_request
    return results


def _fetch_serper_scholar(plan: SearchPlan, query: SearchQuery) -> List[Record]:
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        print(f"Skipping Serper Scholar for '{query.label}' (SERPER_API_KEY not set).")
        return []

    per_page = max(1, min(query.per_page, 20))
    payload_base = {
        "q": query.search,
        "num": per_page,
    }
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }

    results: List[Record] = []
    seen_ids: set[str] = set()
    max_pages_env = int(os.getenv("SERPER_SCHOLAR_MAX_PAGES", "5"))
    max_pages = max(1, max_pages_env)

    doi_pattern = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
    year_pattern = re.compile(r"(19|20)\d{2}")

    def _extract_doi(*values: Optional[str]) -> Optional[str]:
        for value in values:
            if not value or not isinstance(value, str):
                continue
            match = doi_pattern.search(value)
            if match:
                return match.group(0).lower()
        return None

    def _extract_year(*values: Optional[str]) -> Optional[int]:
        for value in values:
            if not value or not isinstance(value, str):
                continue
            match = year_pattern.search(value)
            if match:
                try:
                    return int(match.group(0))
                except ValueError:
                    continue
        return None

    fetched = 0
    page = 1
    while fetched < query.max_results and page <= max_pages:
        payload = dict(payload_base)
        if page > 1:
            payload["page"] = page
        try:
            response = requests.post(SERPER_SCHOLAR_ENDPOINT, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            print(f"Serper Scholar request failed for '{query.label}' (page {page}): {exc}")
            break

        data = response.json()
        organic = data.get("organic")
        if not isinstance(organic, list) or not organic:
            if page == 1:
                print(f"Serper Scholar returned no results for '{query.label}'.")
            break

        page_new = 0
        for entry in organic:
            if fetched >= query.max_results:
                break
            if not isinstance(entry, dict):
                continue
            record_id_raw = entry.get("resultId") or entry.get("link") or entry.get("pdf")
            if record_id_raw and record_id_raw in seen_ids:
                continue

            title = entry.get("title") or ""
            link = entry.get("link") or entry.get("pdf") or ""
            snippet = entry.get("snippet") or entry.get("description") or ""
            publication_info = entry.get("publicationInfo") or ""
            inline_links = entry.get("inlineLinks") if isinstance(entry.get("inlineLinks"), dict) else {}
            cited_by_total = inline_links.get("citedBy", {}).get("total") if inline_links else None

            doi = _extract_doi(link, snippet, publication_info)

            year = entry.get("year")
            if isinstance(year, str) and year.isdigit():
                year = int(year)
            elif isinstance(year, int):
                year = year
            else:
                year = _extract_year(publication_info, snippet)

            authors: List[str] = []
            authors_raw = entry.get("authors")
            if isinstance(authors_raw, list):
                for item in authors_raw:
                    if isinstance(item, dict):
                        name = item.get("name")
                    else:
                        name = str(item)
                    if name and name.strip():
                        authors.append(name.strip())
            if not authors and isinstance(publication_info, str):
                author_part = publication_info.split(" - ", 1)[0]
                if author_part and not year_pattern.search(author_part):
                    potential = re.split(r",|·| and ", author_part)
                    authors = [name.strip() for name in potential if name.strip()]

            cited_by = 0
            if cited_by_total is not None:
                try:
                    cited_by = int(str(cited_by_total).replace(",", ""))
                except ValueError:
                    cited_by = 0

            if record_id_raw:
                seen_ids.add(record_id_raw)
            record_id = record_id_raw or f"serper:{title[:80]}:{page}"
            record = _create_record(
                record_id=f"serper:{record_id}",
                doi=doi,
                title=title,
                year=year,
                citations=cited_by,
                landing_page=link,
                authors=authors,
                abstract=snippet,
                source_label=f"serper_scholar:{query.label}",
            )
            results.append(record)
            fetched += 1
            page_new += 1

        if page_new == 0:
            break
        page += 1
        if plan.sleep:
            time.sleep(plan.sleep)

    return results


def _fetch_core(plan: SearchPlan, query: SearchQuery) -> List[Record]:
    api_key = os.environ.get("CORE_API_KEY")
    if not api_key:
        print(f"Skipping CORE for '{query.label}' (CORE_API_KEY not set).")
        return []
    results: List[Record] = []
    page = 1
    fetched = 0
    while fetched < query.max_results:
        page_size = min(query.per_page, query.max_results - fetched, 100)
        url = f"{CORE_SEARCH_ENDPOINT}{quote_plus(query.search)}"
        params = {
            "page": page,
            "pageSize": page_size,
            "apiKey": api_key,
        }
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            print(f"CORE request failed for '{query.label}': {exc}")
            break
        payload = response.json()
        items = payload.get("data") or []
        if not items:
            break
        for item in items:
            title = item.get("title") or ""
            doi = item.get("doi")
            landing = item.get("downloadUrl") or item.get("fulltextIdentifier")
            year_raw = item.get("year") or item.get("datePublished")
            year_int = None
            if year_raw:
                try:
                    year_int = int(str(year_raw)[:4])
                except ValueError:
                    year_int = None
            abstract = item.get("description") or ""
            authors: List[str] = []
            raw_authors = item.get("authors")
            if isinstance(raw_authors, list):
                for author in raw_authors:
                    if isinstance(author, dict):
                        name = author.get("name") or author.get("fullname")
                        if name:
                            authors.append(name)
                    elif isinstance(author, str) and author.strip():
                        authors.append(author.strip())
            elif isinstance(raw_authors, str):
                authors = [name.strip() for name in raw_authors.split(",") if name.strip()]
            citations = int(item.get("citations") or 0)
            record_id = item.get("id") or item.get("coreId") or f"core:{doi or title[:50]}"
            results.append(
                _create_record(
                    record_id=f"core:{record_id}",
                    doi=doi,
                    title=title,
                    year=year_int,
                    citations=citations,
                    landing_page=landing,
                    authors=authors,
                    abstract=abstract,
                    source_label=f"core:{query.label}",
                )
            )
        fetched += len(items)
        if len(items) < page_size:
            break
        page += 1
        if plan.sleep:
            time.sleep(plan.sleep)
    return results


def _fetch_arxiv(plan: SearchPlan, query: SearchQuery) -> List[Record]:
    headers = {"User-Agent": "LitSearchTemplate/1.0"}
    results: List[Record] = []
    start = 0
    while start < query.max_results:
        max_results = min(query.per_page, query.max_results - start, 100)
        params = {
            "search_query": f"all:{query.search}",
            "start": start,
            "max_results": max_results,
        }
        try:
            response = requests.get(ARXIV_API_ENDPOINT, params=params, headers=headers, timeout=30)
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            print(f"arXiv request failed for '{query.label}': {exc}")
            break
        try:
            root = ET.fromstring(response.text)
        except ET.ParseError as exc:
            print(f"Failed to parse arXiv response for '{query.label}': {exc}")
            break
        ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
        entries = root.findall("atom:entry", ns)
        if not entries:
            break
        for entry in entries:
            title = entry.findtext("atom:title", default="", namespaces=ns)
            summary = entry.findtext("atom:summary", default="", namespaces=ns)
            published = entry.findtext("atom:published", default="", namespaces=ns)
            year = None
            if published:
                try:
                    year = int(published[:4])
                except ValueError:
                    year = None
            doi = entry.findtext("arxiv:doi", default="", namespaces=ns) or None
            pdf_url = None
            for link in entry.findall("atom:link", ns):
                if link.get("type") == "application/pdf":
                    pdf_url = link.get("href")
                    break
            authors = [author.findtext("atom:name", default="", namespaces=ns) for author in entry.findall("atom:author", ns)]
            entry_id = entry.findtext("atom:id", default="", namespaces=ns) or f"arxiv:{title[:50]}"
            results.append(
                _create_record(
                    record_id=f"arxiv:{entry_id}",
                    doi=doi,
                    title=title,
                    year=year,
                    citations=0,
                    landing_page=pdf_url or entry_id,
                    authors=authors,
                    abstract=summary,
                    source_label=f"arxiv:{query.label}",
                )
            )
        start += len(entries)
        if len(entries) < max_results:
            break
        if plan.sleep:
            time.sleep(plan.sleep)
    return results


def _fetch_doaj(plan: SearchPlan, query: SearchQuery) -> List[Record]:
    api_key = os.environ.get("DOAJ_API_KEY")
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    results: List[Record] = []
    page = 1
    fetched = 0
    while fetched < query.max_results:
        page_size = min(query.per_page, query.max_results - fetched, 100)
        url = f"{DOAJ_SEARCH_ENDPOINT}{quote_plus(query.search)}"
        params = {"page": page, "pageSize": page_size}
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            print(f"DOAJ request failed for '{query.label}': {exc}")
            break
        payload = response.json()
        items = payload.get("results") or []
        if not items:
            break
        for item in items:
            bibjson = item.get("bibjson", {})
            title = bibjson.get("title") or ""
            abstract = bibjson.get("abstract") or ""
            doi = None
            for identifier in bibjson.get("identifier", []):
                if identifier.get("type") == "doi" and identifier.get("id"):
                    doi = identifier.get("id")
                    break
            year = bibjson.get("year")
            try:
                year_int = int(year) if year else None
            except ValueError:
                year_int = None
            authors = [author.get("name") for author in bibjson.get("author", []) if author.get("name")]
            landing = None
            for link in bibjson.get("link", []):
                if link.get("type") == "fulltext" or not landing:
                    landing = link.get("url")
            record_id = item.get("id") or f"doaj:{doi or title[:50]}"
            results.append(
                _create_record(
                    record_id=f"doaj:{record_id}",
                    doi=doi,
                    title=title,
                    year=year_int,
                    citations=0,
                    landing_page=landing,
                    authors=authors,
                    abstract=abstract,
                    source_label=f"doaj:{query.label}",
                )
            )
        fetched += len(items)
        if len(items) < page_size:
            break
        page += 1
        if plan.sleep:
            time.sleep(plan.sleep)
    return results


PROVIDER_FETCHERS = {
    "openalex": _fetch_openalex,
    "crossref": _fetch_crossref,
    "semantic_scholar": _fetch_semantic_scholar,
    "semanticscholar": _fetch_semantic_scholar,
    "serper_scholar": _fetch_serper_scholar,
    "serper": _fetch_serper_scholar,
    "core": _fetch_core,
    "arxiv": _fetch_arxiv,
    "doaj": _fetch_doaj,
}


def _deduplicate(records: Iterable[Record]) -> List[Record]:
    merged: Dict[str, Record] = {}
    for record in records:
        key = record.doi or record.record_id
        key = key.lower() if isinstance(key, str) else record.record_id
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
        description="Harvest literature results from configured providers and rebuild the dataset CSV.",
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
        for provider in plan.providers:
            provider_key = provider.strip().lower()
            fetcher = PROVIDER_FETCHERS.get(provider_key)
            if not fetcher:
                print(f"Skipping unsupported provider '{provider}'.")
                continue
            print(f"Querying {provider_key} for '{query.label}' ...")
            fetched = fetcher(plan, query)
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
                "record_id": record.record_id,
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
                "record_id": record.record_id,
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
