from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional

from src.config import (
    S2_BULK_URL,
    S2_SLEEP_SECONDS,
    S2_MAX_RETRIES,
    RAW_ARTICLES_DIR,
    default_specs,
)
from src.utils.io import ensure_dir, write_jsonl
from src.utils.s2_client import SemanticScholarClient


FIELDS = "paperId,title,abstract,venue,year,url"


def collect_bulk_for_venue(
    client: SemanticScholarClient,
    venue_name: str,
    year_range: str,
    query_terms: List[str],
    target_with_abstracts: int,
    page_size: int,
    max_requests_per_query: int,
) -> List[Dict[str, Any]]:
    """
    Collect papers with abstracts for a venue using bulk search.
    Uses multiple queries and token pagination (if provided).
    Balances collection across queries to reduce single-query bias, then tops up to target.
    """
    seen: dict[str, Dict[str, Any]] = {}

    if not query_terms:
        return []

    base = target_with_abstracts // len(query_terms)
    extra = target_with_abstracts % len(query_terms)
    query_targets = {q: base + (1 if i < extra else 0) for i, q in enumerate(query_terms)}

    query_tokens: dict[str, Optional[str]] = {q: None for q in query_terms}
    query_requests: dict[str, int] = {q: 0 for q in query_terms}

    def collect_for_query(q: str, max_new: Optional[int]) -> int:
        token = query_tokens[q]
        req = query_requests[q]
        collected = 0

        while req < max_requests_per_query and len(seen) < target_with_abstracts:
            if max_new is not None and collected >= max_new:
                break
            params: Dict[str, Any] = {
                "query": q,
                "venue": venue_name,
                "year": year_range,
                "limit": page_size,
                "fields": FIELDS,
            }
            if token:
                params["token"] = token

            resp = client.get(S2_BULK_URL, params=params)
            data = resp.get("data") or []
            token = resp.get("token")  # may be None if no further pages

            added = 0
            for p in data:
                if len(seen) >= target_with_abstracts:
                    break
                if max_new is not None and collected >= max_new:
                    break
                pid = p.get("paperId")
                if not pid:
                    continue
                if not p.get("abstract"):
                    continue
                if pid not in seen:
                    p["query_used"] = q
                    p["venue_filter"] = venue_name
                    seen[pid] = p
                    added += 1
                    collected += 1

            req += 1
            print(
                f"Venue='{venue_name}' | Query='{q}' | Requests={req:02d} "
                f"| Added={added:3d} | Total(with abs)={len(seen):3d} | token={'YES' if token else 'NO'}"
            )

            if not token:
                break

        query_tokens[q] = token
        query_requests[q] = req
        return collected

    for q in query_terms:
        if len(seen) >= target_with_abstracts:
            break
        target_for_query = query_targets[q]
        if target_for_query <= 0:
            continue
        collect_for_query(q, target_for_query)

    if len(seen) < target_with_abstracts:
        for q in query_terms:
            if len(seen) >= target_with_abstracts:
                break
            collect_for_query(q, None)

    return list(seen.values())


def main():
    parser = argparse.ArgumentParser(description="Collect Semantic Scholar papers (bulk search) for configured venues.")
    parser.add_argument("--api-key", type=str, default=None, help="Semantic Scholar API key (or set S2_API_KEY env var).")
    parser.add_argument("--out-dir", type=str, default=str(RAW_ARTICLES_DIR), help="Output directory for raw JSONL files.")
    args = parser.parse_args()

    out_dir = RAW_ARTICLES_DIR.__class__(args.out_dir)  # Path-like
    ensure_dir(out_dir)

    client = SemanticScholarClient(api_key=args.api_key, sleep_seconds=S2_SLEEP_SECONDS, max_retries=S2_MAX_RETRIES)

    for spec in default_specs():
        out_path = out_dir / f"{spec.venue_key}.jsonl"

        rows = collect_bulk_for_venue(
            client=client,
            venue_name=spec.venue_name,
            year_range=spec.year_range,
            query_terms=spec.query_terms,
            target_with_abstracts=spec.target_with_abstracts,
            page_size=spec.page_size,
            max_requests_per_query=spec.max_requests_per_query,
        )

        # If you overshoot target (rare), keep deterministic first-N
        rows = rows[: spec.target_with_abstracts]

        write_jsonl(out_path, rows)
        print(f"Wrote {len(rows)} rows to: {out_path}")


if __name__ == "__main__":
    main()
