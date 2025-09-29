# Generalised Scholarly Search Pipeline

Reusable tooling for harvesting, downloading, and citing literature for **any** research topic. The project abstracts the bespoke “Bayesian avatars” workflow into a configuration-driven engine that you can run end-to-end with a single command.

- **Harvester**: Pulls ranked candidates from OpenAlex using configurable query plans.
- **Downloader**: Grabs deterministic open-access PDFs first, then fans out through Serper-driven search + scraping (no credentials stored in the repo).
- **Citations**: Exports BibTeX/RIS for every DOI in the dataset.
- **Orchestrator**: `python -m scripts.run_pipeline` updates the search plan, executes all stages, and prints a diff against the previous run.

Everything ships with ASCII sanitisation, reproducible slugs, and audit-friendly JSON snapshots. The repo contains **no API keys** or harvested outputs—clone it fresh whenever you need a clean slate.

## Requirements

- Python ≥ 3.10
- [Serper](https://serper.dev) API key (set at runtime via `SERPER_API_KEY`)
- Internet access for OpenAlex/Serper/doi.org
- Optional provider keys:
  - `CROSSREF_MAILTO` to identify yourself to Crossref
  - `SEMANTIC_SCHOLAR_API_KEY` for higher Semantic Scholar quotas
  - `CORE_API_KEY` for CORE search access
  - `DOAJ_API_KEY` for authenticated DOAJ queries
  - ArXiv access uses an HTTP User-Agent only (no key required)

Optional:
- `CROSSREF_MAILTO` ENV var to identify yourself to Crossref/doi.org

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # or pip install requests pandas

export SERPER_API_KEY=your-serper-token
python -m scripts.run_pipeline "individual differences for virtual reality social interactions" \
  --profile social-vr-individual-diff --year-min 2018 --year-max 2025
```

The orchestrator will:
1. Regenerate `config/search_plan.json` (unless you point it at a custom profile/plan).
2. Harvest OpenAlex results and write the dataset to `data/processed/library.csv`.
3. Attempt deterministic and Serper-assisted PDF downloads (`data/full_text/`, ignored by git).
4. Export BibTeX + RIS to `data/processed/`.
5. Print a summary highlighting new/removed DOIs and full-text coverage.

## Repository Layout

```
.
├── config/
│   ├── project.json              # central configuration (paths, columns, outputs)
│   └── open_access_sources.json  # optional DOI → {url, note} map for deterministic downloads
├── data/
│   ├── full_text/                # PDFs land here (ignored by git)
│   ├── processed/                # dataset CSV + bibliographies/logs
│   └── raw/                      # place to stash upstream responses if needed
├── scripts/
│   ├── common.py                 # config helpers, path utilities, ASCII slug/normalisation
│   ├── collect_openalex.py       # harvest OpenAlex using the active search plan
│   ├── download_open_pdfs.py     # grab deterministic PDFs from the DOI→URL map
│   ├── serper_download_pdfs.py   # discover missing PDFs through Serper + scraping
│   ├── fetch_references.py       # export BibTeX + RIS using doi.org / Crossref
│   ├── run_pipeline.py           # one-command orchestrator
│   └── scholarly_recipe.py       # optional helpers for the `scholarly` package
├── README.md
└── requirements.txt (optional convenience)
```

`data/full_text/` is ignored by default so large binaries never enter version control.

## Configuring a Search

1. **Choose a profile or edit the plan**
   - Use the built-in `--profile social-vr-individual-diff` or supply your own topic; the orchestrator will auto-generate sensible OpenAlex queries (or you can edit `config/search_plan.json` directly).
- Add `--require TERM` flags to force keywords into the title/abstract filter. When omitted, the script injects "virtual reality" if the topic contains it.
- Use `--require-near TERM_A TERM_B` to keep only records where the two terms appear within `--near-window` characters (default 120) of one another in the title/abstract. Repeat the flag for multiple pairs.
- Required keywords automatically seed focused OpenAlex queries (AND pairs and grouped clauses), so you don’t need to hand-write matching `--extra-query` strings for core concepts.
- Supply `--providers openalex crossref semantic_scholar core arxiv doaj` (default) or a custom subset to control which free indexes contribute results.

2. **Set bounds**
   - `--year-min`, `--year-max`, `--min-citations`, and `--top-n` mirror the JSON fields. CLI arguments win without permanently rewriting your custom plan.

3. **Optional deterministic PDFs**
   - Populate `config/open_access_sources.json` with stable DOI→URL mappings (SLUGs + provenance notes are handled for you).

## Running Steps Manually

Each stage remains usable on its own:

```bash
python -m scripts.collect_openalex --config config/project.json
python -m scripts.download_open_pdfs --config config/project.json
python -m scripts.serper_download_pdfs --config config/project.json
python -m scripts.fetch_references --config config/project.json
```

Use `--dry-run` on the downloader scripts to preview actions without writing files.

## Output Artifacts

- `data/processed/library.csv` – primary dataset (headers only in the repo).
- `data/processed/library.bib` / `.ris` – regenerated per run (not tracked).
- `data/full_text/*.pdf` – downloaded papers (ignored by git).
- `data/processed/*_candidates.json` / `*_dedup.json` – optional audit snapshots (delete if you prefer a clean tree).

## Tips & Troubleshooting

- **Serper 403s** – Ensure `SERPER_API_KEY` is set; you can limit workload with `SERPER_MAX_ROWS` or the orchestrator’s `--serper-rounds`.
- **Manual PDFs** – If some hosts require logins, obtain the PDF manually and drop it into `data/full_text/`. Re-run the Serper script to update the CSV metadata.
- **Profiles** – Add new templates to `PROFILES` in `scripts/run_pipeline.py` for frequently used queries/topics.
- **ASCII cleanliness** – Titles/abstracts are normalised to plain ASCII in `collect_openalex.py` to avoid mojibake in downstream tools.

## Keeping Secrets Safe

- The repo contains no keys. Set runtime secrets via environment variables (`export SERPER_API_KEY=...`).
- Avoid committing generated data; keep `data/processed/library.csv` trimmed to a header when sharing the repo.

Clone, set a topic/question, run the orchestrator, and you have a fresh, comprehensive literature search ready for analysis or import into Zotero/Obsidian/Notion.
