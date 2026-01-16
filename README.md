# Thematic Alignment of TPAMI Publications with Journal Scope (2015–2025)

This project quantifies how closely IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) articles align with the journal’s stated “Aims & Scope”. It collects abstracts, embeds scope + abstracts using Sentence‑BERT, and computes an alignment score per paper (cosine similarity), enabling outlier detection and drift analysis over time.

## What’s implemented

- **Data collection** from the Semantic Scholar Graph API (bulk search) for a target venue and year range
- **Preprocessing** into clean CSVs (`articles_clean.csv`, `scopes_clean.csv`)
- **Semantic alignment** (Sentence‑BERT embeddings + cosine similarity)
- **Analysis outputs**: score distribution, trend by year, boxplot, percentile outliers, and drift slope
- **Notebook** for EDA + qualitative inspection support

## Repository layout

- `src/config.py` — dataset spec (venue, year range, target size, neutral queries for API)
- `src/data/collect_articles.py` — collects raw JSONL into `data/raw/articles/`
- `src/data/preprocess.py` — creates `data/processed/*.csv` + `data/stats/dataset_summary.json`
- `src/semantic_alignment.py` — computes alignment + writes outputs into `results/semantic_alignment/`
- `src/utils/` — I/O, API client, shared analysis/plotting helpers
- `data/raw/scopes/` — scope text files (e.g., `tpami.txt`)
- `experiments/01_eda_and_pipeline_showcase.ipynb` — EDA + top/bottom papers for qualitative review
- `report/report.md` — written report (intro/method/results/conclusion)

## Setup

1) Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2) Set your Semantic Scholar API key (required for collection):

```bash
export S2_API_KEY="YOUR_KEY"
```

## Run the pipeline

```bash
# 1) collect raw articles (writes to data/raw/articles/tpami.jsonl by default)
PYTHONPATH=. python src/data/collect_articles.py

# 2) preprocess into clean CSVs + summary stats
PYTHONPATH=. python src/data/preprocess.py

# 3) compute semantic alignment + plots/outliers/drift
PYTHONPATH=. python src/semantic_alignment.py --out-dir results/semantic_alignment
```

## Outputs

`results/semantic_alignment/` contains:

- `semantic_alignment_scores.csv` — per-paper alignment scores
- `semantic_alignment_distribution.png` — alignment histogram
- `semantic_alignment_trend_by_year.png` — yearly mean trend
- `semantic_alignment_boxplot_by_venue.png` — score spread (single venue by default)
- `outliers_percentile.csv` — low/high tail papers for qualitative inspection
- `drift_slope_by_year.csv` — linear drift slope fitted on yearly means

## Notes

- Semantic Scholar bulk search requires a `query` parameter. `src/config.py` uses **neutral, high-frequency terms** to approximate venue-only collection while remaining reproducible. You can adjust `QUERIES`, `YEAR_RANGE`, and `TARGET_WITH_ABSTRACTS_PER_VENUE` in `src/config.py`.
