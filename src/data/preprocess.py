from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from src.config import RAW_ARTICLES_DIR, RAW_SCOPES_DIR, PROCESSED_DIR, STATS_DIR, SEED, VENUES
from src.utils.io import ensure_dir, read_jsonl, write_json


def normalize_text(s: str) -> str:
    """Normalize whitespace and trim leading/trailing spaces."""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def load_scopes(scopes_dir: Path, allowed_venues: set[str] | None = None) -> pd.DataFrame:
    """Load scope text files into a DataFrame keyed by venue."""
    rows = []
    for p in sorted(scopes_dir.glob("*.txt")):
        venue_key = p.stem
        if allowed_venues is not None and venue_key not in allowed_venues:
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        rows.append(
            {
                "venue_key": venue_key,
                "scope_text": normalize_text(text),
            }
        )
    return pd.DataFrame(rows)


def load_articles(articles_dir: Path, allowed_venues: set[str] | None = None) -> pd.DataFrame:
    """Load raw JSONL articles into a normalized DataFrame."""
    all_rows = []
    for p in sorted(articles_dir.glob("*.jsonl")):
        venue_key = p.stem
        if allowed_venues is not None and venue_key not in allowed_venues:
            continue
        rows = read_jsonl(p)
        for r in rows:
            all_rows.append(
                {
                    "venue_key": venue_key,
                    "paperId": r.get("paperId"),
                    "title": normalize_text(r.get("title") or ""),
                    "abstract": normalize_text(r.get("abstract") or ""),
                    "venue": r.get("venue") or "",
                    "year": r.get("year"),
                    "url": r.get("url") or "",
                    "query_used": r.get("query_used") or "",
                }
            )
    df = pd.DataFrame(all_rows)
    return df


def main():
    parser = argparse.ArgumentParser(description="Preprocess raw articles + scopes into clean CSV + summary stats.")
    parser.add_argument("--raw-articles-dir", type=str, default=str(RAW_ARTICLES_DIR))
    parser.add_argument("--raw-scopes-dir", type=str, default=str(RAW_SCOPES_DIR))
    parser.add_argument("--processed-dir", type=str, default=str(PROCESSED_DIR))
    parser.add_argument("--stats-dir", type=str, default=str(STATS_DIR))
    args = parser.parse_args()

    raw_articles_dir = Path(args.raw_articles_dir)
    raw_scopes_dir = Path(args.raw_scopes_dir)
    processed_dir = Path(args.processed_dir)
    stats_dir = Path(args.stats_dir)

    ensure_dir(processed_dir)
    ensure_dir(stats_dir)

    allowed_venues = set(VENUES.keys())
    df_articles = load_articles(raw_articles_dir, allowed_venues=allowed_venues)

    # Basic cleaning rules
    df_articles = df_articles[df_articles["paperId"].notna()]
    df_articles = df_articles[df_articles["abstract"].str.len() > 0]
    df_articles = df_articles.drop_duplicates(subset=["paperId"]).reset_index(drop=True)

    # Optional: deterministic shuffle within each venue
    df_articles = (
        df_articles.groupby("venue_key", group_keys=False)
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )

    df_scopes = load_scopes(raw_scopes_dir, allowed_venues=allowed_venues)

    # Save outputs
    articles_out = processed_dir / "articles_clean.csv"
    scopes_out = processed_dir / "scopes_clean.csv"

    df_articles.to_csv(articles_out, index=False)
    df_scopes.to_csv(scopes_out, index=False)

    # Summary stats
    summary: Dict[str, Any] = {
        "num_articles_total": int(len(df_articles)),
        "num_scopes_total": int(len(df_scopes)),
        "articles_per_venue_key": df_articles["venue_key"].value_counts().to_dict(),
        "avg_abstract_len": float(df_articles["abstract"].str.len().mean()) if len(df_articles) else 0.0,
        "min_year": int(df_articles["year"].min()) if df_articles["year"].notna().any() else None,
        "max_year": int(df_articles["year"].max()) if df_articles["year"].notna().any() else None,
    }
    write_json(stats_dir / "dataset_summary.json", summary)

    print(f"Wrote: {articles_out}")
    print(f"Wrote: {scopes_out}")
    print(f"Wrote: {stats_dir / 'dataset_summary.json'}")


if __name__ == "__main__":
    main()
