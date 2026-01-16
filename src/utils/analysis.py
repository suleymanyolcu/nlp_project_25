from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def normalize_percentile(value: float) -> float:
    """Normalize a percentile argument to [0, 1] for quantile computations."""
    if value < 0:
        raise ValueError("outlier percentile must be non-negative")
    return value / 100.0 if value > 1 else value


def compute_percentile_outliers(
    df: pd.DataFrame,
    score_col: str,
    percentile: float,
) -> pd.DataFrame:
    """Return rows in the lower/upper percentile tails for each venue."""
    p = normalize_percentile(percentile)
    columns = [
        "paperId",
        "venue_key",
        "title",
        "year",
        "url",
        score_col,
        "outlier_side",
        "low_threshold",
        "high_threshold",
    ]
    if p <= 0 or df.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for venue_key, group in df.groupby("venue_key"):
        scores = group[score_col]
        if scores.empty:
            continue
        low = scores.quantile(p)
        high = scores.quantile(1 - p)
        mask = (scores <= low) | (scores >= high)
        if not mask.any():
            continue
        subset = group.loc[mask, ["paperId", "venue_key", "title", "year", "url", score_col]].copy()
        subset["outlier_side"] = np.where(subset[score_col] <= low, "low", "high")
        subset["low_threshold"] = low
        subset["high_threshold"] = high
        rows.append(subset)

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.concat(rows, ignore_index=True)


def compute_yearly_means(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Compute mean alignment per year and venue."""
    data = df.copy()
    if "year" not in data.columns:
        return pd.DataFrame(columns=["venue_key", "year", score_col])
    data["year"] = pd.to_numeric(data["year"], errors="coerce")
    data = data.dropna(subset=["year"])
    if data.empty:
        return pd.DataFrame(columns=["venue_key", "year", score_col])
    data["year"] = data["year"].astype(int)
    return (
        data.groupby(["venue_key", "year"], as_index=False)[score_col]
        .mean()
        .sort_values("year")
    )


def compute_drift_slope(yearly_df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Fit a simple linear trend (slope) to yearly mean alignment."""
    columns = ["venue_key", "n_years", "year_min", "year_max", "slope"]
    if yearly_df.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for venue_key, group in yearly_df.groupby("venue_key"):
        g = group.dropna(subset=["year", score_col]).sort_values("year")
        if len(g) >= 2:
            slope = np.polyfit(g["year"], g[score_col], 1)[0]
        else:
            slope = np.nan
        rows.append(
            {
                "venue_key": venue_key,
                "n_years": int(len(g)),
                "year_min": int(g["year"].min()) if len(g) else None,
                "year_max": int(g["year"].max()) if len(g) else None,
                "slope": float(slope) if not np.isnan(slope) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_alignment_distribution(
    df: pd.DataFrame,
    score_col: str,
    title: str,
    out_path: Path,
) -> None:
    """Plot alignment score histograms by venue."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    for venue_key, group in df.groupby("venue_key"):
        plt.hist(group[score_col], bins=30, alpha=0.5, label=venue_key)
    plt.xlabel("Cosine similarity (scope vs abstract)")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_alignment_by_year(
    df: pd.DataFrame,
    score_col: str,
    title: str,
    out_path: Path,
) -> None:
    """Plot yearly mean alignment trends by venue."""
    import matplotlib.pyplot as plt

    data = compute_yearly_means(df, score_col)
    if data.empty:
        return
    plt.figure(figsize=(8, 5))
    for venue_key, group in data.groupby("venue_key"):
        plt.plot(group["year"], group[score_col], marker="o", label=venue_key)
    plt.xlabel("Year")
    plt.ylabel("Mean alignment")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_alignment_box(
    df: pd.DataFrame,
    score_col: str,
    title: str,
    out_path: Path,
) -> None:
    """Plot alignment score boxplots by venue."""
    import matplotlib.pyplot as plt

    grouped = [g[score_col].to_numpy() for _, g in df.groupby("venue_key")]
    labels = [k for k, _ in df.groupby("venue_key")]
    if not grouped:
        return
    plt.figure(figsize=(6, 4))
    plt.boxplot(grouped, tick_labels=labels, showfliers=False)
    plt.ylabel("Alignment")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
