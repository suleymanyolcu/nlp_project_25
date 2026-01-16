from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --- Paths (repo-relative) ---
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
STATS_DIR = DATA_DIR / "stats"

RAW_SCOPES_DIR = RAW_DIR / "scopes"
RAW_ARTICLES_DIR = RAW_DIR / "articles"

# --- Semantic Scholar API ---
S2_BULK_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
S2_SLEEP_SECONDS = 1.1  # strict 1 req/sec + safety margin
S2_MAX_RETRIES = 6

# --- Project dataset spec ---
SEED = 42

# Using an older, long-running journal for broader historical coverage.
VENUES = {
    "tpami": "IEEE Transactions on Pattern Analysis and Machine Intelligence",
}

# For reproducibility, keep this fixed and document it in the report.
# Semantic Scholar requires a query param; use neutral, high-frequency terms to approximate venue-only collection.
QUERIES = [
    "study",
    "analysis",
    "method",
    "approach",
    "result",
    "system",
    "model",
    "data",
]

YEAR_RANGE = "2015-2025"  # Semantic Scholar format

TARGET_WITH_ABSTRACTS_PER_VENUE = 4000
BULK_PAGE_SIZE = 200          # keep moderate; reduces payload + risk
MAX_REQUESTS_PER_QUERY = 10   # safety cap per query per venue


@dataclass(frozen=True)
class CollectionSpec:
    venue_key: str
    venue_name: str
    query_terms: list[str]
    year_range: str
    target_with_abstracts: int
    page_size: int
    max_requests_per_query: int


def default_specs() -> list[CollectionSpec]:
    return [
        CollectionSpec(
            venue_key=k,
            venue_name=v,
            query_terms=list(QUERIES),
            year_range=YEAR_RANGE,
            target_with_abstracts=TARGET_WITH_ABSTRACTS_PER_VENUE,
            page_size=BULK_PAGE_SIZE,
            max_requests_per_query=MAX_REQUESTS_PER_QUERY,
        )
        for k, v in VENUES.items()
    ]
