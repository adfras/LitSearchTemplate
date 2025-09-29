"""Utility helpers for working with the ``scholarly`` package.

This module captures a handful of small, self-contained workflows that
mirror the quick-start patterns described in the Scholarly documentation:

* Set up retries, timeouts, and optional proxy rotation to avoid temporary
  blocks from Google Scholar.
* Search for publications and persist the top N results as a CSV file.
* Retrieve key metrics and publications for an author profile.
* Track new citations for a specific paper while keeping state on disk.
* Export a BibTeX entry for an arbitrary search result.

All functions operate directly on top of the official ``scholarly`` package
and should remain compatible with the latest released version on PyPI.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd
from scholarly import ProxyGenerator, scholarly


@dataclass
class ProxyConfig:
    """Configuration options for ``configure_scholarly``."""

    use_proxy: bool = False
    retries: int = 3
    timeout: int = 10
    proxy_timeout: int = 1
    proxy_wait_time: int = 60


def configure_scholarly(config: ProxyConfig) -> Optional[ProxyGenerator]:
    """Apply common Scholarly runtime settings and optional proxy support.

    Parameters
    ----------
    config: ProxyConfig
        Desired configuration options. When ``config.use_proxy`` is false the
        ``ProxyGenerator`` is skipped.

    Returns
    -------
    Optional[ProxyGenerator]
        The configured proxy generator (only when ``use_proxy`` is true).

    Raises
    ------
    RuntimeError
        If proxy setup was requested but no proxy could be acquired.
    """

    scholarly.set_retries(config.retries)
    scholarly.set_timeout(config.timeout)

    if not config.use_proxy:
        return None

    proxy_generator = ProxyGenerator()
    success = proxy_generator.FreeProxies(
        timeout=config.proxy_timeout, wait_time=config.proxy_wait_time
    )
    if not success:
        raise RuntimeError(
            "ProxyGenerator.FreeProxies() did not return any usable proxies."
        )

    scholarly.use_proxy(proxy_generator)
    return proxy_generator


def _normalize_authors(authors_field) -> str:
    """Return a comma-separated author string from Scholarly's metadata."""

    if not authors_field:
        return ""
    if isinstance(authors_field, str):
        return authors_field
    if isinstance(authors_field, (list, tuple, set)):
        return ", ".join(authors_field)
    return str(authors_field)


def search_publications_to_csv(
    query: str,
    output_path: Path,
    limit: Optional[int] = 100,
    delay_seconds: float = 2.0,
    patents: bool = False,
    citations: bool = True,
    year_low: Optional[int] = None,
    year_high: Optional[int] = None,
    sort_by: Optional[str] = None,
) -> Path:
    """Search Google Scholar and persist the results to ``output_path``.

    The parameters mirror those from ``scholarly.search_pubs``.
    """

    search_kwargs = {
        "query": query,
        "patents": patents,
        "citations": citations,
        "year_low": year_low,
        "year_high": year_high,
        "sort_by": sort_by,
    }

    iterator = scholarly.search_pubs(
        **{key: value for key, value in search_kwargs.items() if value is not None}
    )

    rows: List[dict] = []
    remaining = None if limit is None else max(limit, 0)

    for pub in iterator:
        bib = pub.get("bib", {})
        rows.append(
            {
                "title": bib.get("title"),
                "year": bib.get("pub_year"),
                "venue": bib.get("venue"),
                "authors": _normalize_authors(bib.get("author")),
                "num_citations": pub.get("num_citations"),
                "eprint_url": pub.get("eprint_url"),
                "pub_url": pub.get("pub_url"),
            }
        )

        if remaining is not None:
            remaining -= 1
            if remaining <= 0:
                break

        time.sleep(delay_seconds)

    df = pd.DataFrame(rows)
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def fetch_author_metrics(
    author_query: str,
    sections: Sequence[str] = ("basics", "indices", "counts", "publications"),
) -> dict:
    """Return a dictionary of metrics for the first matching author."""

    author_iter = scholarly.search_author(author_query)
    try:
        author = next(author_iter)
    except StopIteration as exc:  # pragma: no cover - mirrors user workflow
        raise ValueError(f"No author matched query: {author_query!r}") from exc

    author = scholarly.fill(author, sections=sections)
    return {
        "name": author.get("name"),
        "affiliation": author.get("affiliation"),
        "email_domain": author.get("email_domain"),
        "hindex": author.get("hindex"),
        "i10index": author.get("i10index"),
        "cites_per_year": author.get("cites_per_year", {}),
        "interests": author.get("interests", []),
        "publications": [
            {
                "title": pub.get("bib", {}).get("title"),
                "num_citations": pub.get("num_citations"),
                "year": pub.get("bib", {}).get("pub_year"),
            }
            for pub in author.get("publications", [])
        ],
    }


def track_new_citations(
    title: str,
    state_path: Path,
    delay_seconds: float = 2.0,
    limit: Optional[int] = None,
) -> List[dict]:
    """Return new citing publications for the specified paper title.

    Results are persisted to ``state_path`` so subsequent runs only report
    truly new citations.
    """

    state_file = Path(state_path).expanduser()
    if state_file.exists():
        seen_raw = json.loads(state_file.read_text())
        seen: set[Tuple[str, Optional[str]]] = {
            tuple(item) for item in seen_raw if isinstance(item, (list, tuple))
        }
    else:
        seen = set()

    publication = scholarly.search_single_pub(title, filled=False)
    publication = scholarly.fill(publication)

    new_entries: List[dict] = []
    for index, citer in enumerate(scholarly.citedby(publication)):
        bib = citer.get("bib", {})
        key = (bib.get("title"), bib.get("pub_year"))
        if key not in seen:
            new_entries.append(
                {
                    "title": bib.get("title"),
                    "year": bib.get("pub_year"),
                    "venue": bib.get("venue"),
                }
            )
            seen.add(key)

        if limit is not None and index + 1 >= limit:
            break

        time.sleep(delay_seconds)

    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps([list(item) for item in seen]))
    return new_entries


def fetch_bibtex_for_query(query: str) -> str:
    """Return a BibTeX entry for the first publication matching ``query``."""

    publication = next(scholarly.search_pubs(query))
    return scholarly.bibtex(publication)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Utility CLI around the scholarly package workflows.",
    )

    proxy_parent = argparse.ArgumentParser(add_help=False)
    proxy_parent.add_argument(
        "--use-proxy",
        action="store_true",
        help="Enable free proxy rotation via ProxyGenerator.FreeProxies().",
    )
    proxy_parent.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of Scholarly retries (default: 3).",
    )
    proxy_parent.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Scholarly timeout in seconds (default: 10).",
    )
    proxy_parent.add_argument(
        "--proxy-timeout",
        type=int,
        default=1,
        help="Per-proxy response timeout in seconds (default: 1).",
    )
    proxy_parent.add_argument(
        "--proxy-wait",
        type=int,
        default=60,
        help="Seconds to wait between proxy rotations (default: 60).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser(
        "search",
        parents=[proxy_parent],
        help="Search publications and write a CSV file.",
    )
    search_parser.add_argument("query", help="Search query string.")
    search_parser.add_argument(
        "--out",
        type=Path,
        default=Path("scholar_search.csv"),
        help="Output CSV path (default: ./scholar_search.csv).",
    )
    search_parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of publications to export (default: 100).",
    )
    search_parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to sleep between requests (default: 2.0).",
    )
    search_parser.add_argument(
        "--year-low",
        type=int,
        dest="year_low",
        help="Lower bound for publication year filter.",
    )
    search_parser.add_argument(
        "--year-high",
        type=int,
        dest="year_high",
        help="Upper bound for publication year filter.",
    )
    search_parser.add_argument(
        "--sort-by",
        choices=["relevance", "date"],
        help="Sort order for results (default: relevance).",
    )
    search_parser.add_argument(
        "--include-patents",
        action="store_true",
        help="Include patents in the search results (default: exclude).",
    )
    search_parser.add_argument(
        "--exclude-citations",
        action="store_true",
        help="Exclude citations from the search results (default: include).",
    )

    author_parser = subparsers.add_parser(
        "author",
        parents=[proxy_parent],
        help="Fetch metrics for the first author match.",
    )
    author_parser.add_argument("query", help="Author search query.")
    author_parser.add_argument(
        "--sections",
        nargs="+",
        default=["basics", "indices", "counts", "publications"],
        help="Sections passed to scholarly.fill().",
    )

    cited_parser = subparsers.add_parser(
        "citedby",
        parents=[proxy_parent],
        help="Track new citations for a publication title.",
    )
    cited_parser.add_argument("title", help="Exact publication title to search.")
    cited_parser.add_argument(
        "--state",
        type=Path,
        default=Path("seen_citers.json"),
        help="JSON file that stores previously seen citations.",
    )
    cited_parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to sleep between citer requests (default: 2.0).",
    )
    cited_parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of citing publications to inspect.",
    )

    bibtex_parser = subparsers.add_parser(
        "bibtex",
        parents=[proxy_parent],
        help="Fetch a BibTeX entry for the first matching publication.",
    )
    bibtex_parser.add_argument("query", help="Publication search query.")

    return parser


def _parse_config(args: argparse.Namespace) -> ProxyConfig:
    return ProxyConfig(
        use_proxy=args.use_proxy,
        retries=args.retries,
        timeout=args.timeout,
        proxy_timeout=args.proxy_timeout,
        proxy_wait_time=args.proxy_wait,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = _parse_config(args)
    try:
        configure_scholarly(config)
    except RuntimeError as err:
        parser.error(str(err))

    if args.command == "search":
        output = search_publications_to_csv(
            query=args.query,
            output_path=args.out,
            limit=args.limit,
            delay_seconds=args.delay,
            patents=args.include_patents,
            citations=not args.exclude_citations,
            year_low=args.year_low,
            year_high=args.year_high,
            sort_by=args.sort_by,
        )
        print(f"Wrote {output}")
        return 0

    if args.command == "author":
        metrics = fetch_author_metrics(args.query, sections=args.sections)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0

    if args.command == "citedby":
        new_entries = track_new_citations(
            title=args.title,
            state_path=args.state,
            delay_seconds=args.delay,
            limit=args.limit,
        )
        print(f"New citations found: {len(new_entries)}")
        for item in new_entries[:10]:
            print(json.dumps(item, ensure_ascii=False))
        return 0

    if args.command == "bibtex":
        entry = fetch_bibtex_for_query(args.query)
        print(entry)
        return 0

    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
