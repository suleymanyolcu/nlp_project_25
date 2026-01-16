from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from src.config import PROCESSED_DIR
from src.utils.io import ensure_dir
from src.utils.analysis import (
    compute_drift_slope,
    compute_percentile_outliers,
    compute_yearly_means,
    plot_alignment_box,
    plot_alignment_by_year,
    plot_alignment_distribution,
)


SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    """Split text into sentences for sentence-level pooling."""
    text = (text or "").strip()
    if not text:
        return []
    parts = SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def embed_documents(
    docs: Iterable[str],
    model,
    batch_size: int,
    sentence_level: bool,
) -> np.ndarray:
    """Encode documents into embeddings, optionally pooling over sentences."""
    docs = list(docs)
    dim = model.get_sentence_embedding_dimension()
    embeddings = np.zeros((len(docs), dim), dtype=np.float32)

    for i, doc in enumerate(docs):
        sentences = split_sentences(doc) if sentence_level else [doc or ""]
        if not sentences:
            continue
        sent_embs = model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        embeddings[i] = sent_embs.mean(axis=0)

    return embeddings


def compute_embedding_alignment(
    df_articles: pd.DataFrame,
    df_scopes: pd.DataFrame,
    model,
    batch_size: int,
    abstract_strategy: str,
    scope_strategy: str,
) -> pd.DataFrame:
    """Compute cosine similarity between abstract and scope embeddings."""
    df_articles = df_articles.copy()
    df_scopes = df_scopes.copy()

    df_articles["abstract"] = df_articles["abstract"].fillna("")
    df_scopes["scope_text"] = df_scopes["scope_text"].fillna("")

    scope_sentence_level = scope_strategy == "sentence"
    abstract_sentence_level = abstract_strategy == "sentence"

    scope_embeddings = embed_documents(
        df_scopes["scope_text"].tolist(),
        model,
        batch_size=batch_size,
        sentence_level=scope_sentence_level,
    )

    abstract_embeddings = embed_documents(
        df_articles["abstract"].tolist(),
        model,
        batch_size=batch_size,
        sentence_level=abstract_sentence_level,
    )

    scope_embeddings = normalize(scope_embeddings)
    abstract_embeddings = normalize(abstract_embeddings)

    venue_to_idx = {k: i for i, k in enumerate(df_scopes["venue_key"])}
    scope_indices = df_articles["venue_key"].map(venue_to_idx)
    if scope_indices.isna().any():
        missing = df_articles.loc[scope_indices.isna(), "venue_key"].unique().tolist()
        raise ValueError(f"Missing scope text for venue_key: {missing}")
    scope_indices = scope_indices.astype(int).to_numpy()

    scope_for_articles = scope_embeddings[scope_indices]
    sims = np.einsum("ij,ij->i", abstract_embeddings, scope_for_articles)
    df_articles["embedding_alignment"] = sims
    return df_articles


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic alignment with Sentence-BERT embeddings.")
    parser.add_argument("--articles", type=str, default=str(PROCESSED_DIR / "articles_clean.csv"))
    parser.add_argument("--scopes", type=str, default=str(PROCESSED_DIR / "scopes_clean.csv"))
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "results" / "semantic_alignment"),
    )
    parser.add_argument(
        "--outlier-percentile",
        type=float,
        default=5.0,
        help="Percentile per tail for outlier detection (e.g., 5 for 5%%).",
    )
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--abstract-strategy",
        type=str,
        choices=["sentence", "full"],
        default="sentence",
        help="Use mean pooling over sentences or a single embedding for each abstract.",
    )
    parser.add_argument(
        "--scope-strategy",
        type=str,
        choices=["full", "sentence"],
        default="sentence",
        help="Represent scopes as a single vector or mean pooled over sentences.",
    )
    args = parser.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise RuntimeError("sentence-transformers not installed. Run: pip install sentence-transformers") from exc

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    df_articles = pd.read_csv(args.articles)
    df_scopes = pd.read_csv(args.scopes)

    model = SentenceTransformer(args.model)

    df_embeddings = compute_embedding_alignment(
        df_articles,
        df_scopes,
        model=model,
        batch_size=args.batch_size,
        abstract_strategy=args.abstract_strategy,
        scope_strategy=args.scope_strategy,
    )

    embedding_scores_out = out_dir / "semantic_alignment_scores.csv"
    df_embeddings[
        ["paperId", "venue_key", "title", "year", "url", "embedding_alignment"]
    ].to_csv(embedding_scores_out, index=False)

    dist_out = out_dir / "semantic_alignment_distribution.png"
    plot_alignment_distribution(
        df_embeddings,
        "embedding_alignment",
        "Semantic alignment distribution by journal",
        dist_out,
    )

    trend_out = out_dir / "semantic_alignment_trend_by_year.png"
    plot_alignment_by_year(
        df_embeddings,
        "embedding_alignment",
        "Semantic alignment trend by year",
        trend_out,
    )

    box_out = out_dir / "semantic_alignment_boxplot_by_venue.png"
    plot_alignment_box(
        df_embeddings,
        "embedding_alignment",
        "Semantic alignment spread by journal",
        box_out,
    )

    yearly_means = compute_yearly_means(df_embeddings, "embedding_alignment")
    drift_df = compute_drift_slope(yearly_means, "embedding_alignment")
    drift_out = out_dir / "drift_slope_by_year.csv"
    drift_df.to_csv(drift_out, index=False)

    outliers_df = compute_percentile_outliers(df_embeddings, "embedding_alignment", args.outlier_percentile)
    outliers_out = out_dir / "outliers_percentile.csv"
    outliers_df.to_csv(outliers_out, index=False)

    print(f"Wrote: {embedding_scores_out}")
    print(f"Wrote: {dist_out}")
    print(f"Wrote: {trend_out}")
    print(f"Wrote: {box_out}")
    print(f"Wrote: {drift_out}")
    print(f"Wrote: {outliers_out}")


if __name__ == "__main__":
    main()
