# src/functions.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================
# I/O helpers
# =========================

def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV with UTF-8 and inferred dtypes."""
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    """Save a DataFrame to CSV (creates parent folders if needed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


# =========================
# General cleaning helpers
# =========================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to snake_case:
    - strip spaces
    - lowercase
    - replace whitespace with underscore
    - remove non [a-z0-9_]
    """
    out = df.copy()
    out.columns = (
        out.columns
          .str.strip()
          .str.lower()
          .str.replace(r"\s+", "_", regex=True)
          .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    return out


def drop_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Drop columns if they exist."""
    out = df.copy()
    existing = [c for c in cols if c in out.columns]
    return out.drop(columns=existing)


def drop_dupes_and_nulls(df: pd.DataFrame, subset: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Drop duplicate rows and rows that are null in any of the subset columns.
    """
    out = df.drop_duplicates().copy()
    if subset:
        out = out.dropna(subset=list(subset))
    return out


def check_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary table of missing values: count and percentage per column.
    """
    nulls = df.isna().sum()
    pct = (df.isna().mean() * 100).round(2)
    return pd.DataFrame({"missing": nulls, "missing_pct": pct}).sort_values("missing", ascending=False)


def clean_strings(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """
    Trim, collapse internal whitespace and standardize to simple spaces on given string columns.
    """
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = (out[c]
                      .astype(str)
                      .str.strip()
                      .str.replace(r"\s+", " ", regex=True))
    return out


# =========================
# Runners dataset cleaning
# (cleaned_marathon.csv)
# =========================

def standardize_gender(df: pd.DataFrame, col: str = "gender") -> pd.DataFrame:
    """
    Map gender variants to {'male','female','unknown'}.
    """
    out = df.copy()
    if col not in out.columns:
        return outimport sys
sys.path.append("../")  # ajusta según tu estructura

from src.functions import (
    load_csv, save_csv, normalize_columns, clean_runners_pipeline,
    winners_wide_to_long, top_countries_by_wins, winners_by_decade_and_country,
    plot_top_countries_bar, plot_decade_country_stacked
)

# Cargar datasets LIMPIOS (ya congelados)
runners = load_csv("../data/clean/cleaned_marathon.csv")
winners = load_csv("../data/clean/cleaned_marathon_winners.csv")

# (Si partieras del raw, ejemplo de pipeline)
# runners_raw = load_csv("../data/raw/berlin_marathon_runners.csv")
# runners_clean = clean_runners_pipeline(runners_raw)
# save_csv(runners_clean, "../data/clean/cleaned_marathon.csv")

# Country dominance
top10 = top_countries_by_wins(winners, n=10)
ax1 = plot_top_countries_bar(top10, save_path="../figures/top_countries_wins.png")

by_decade = winners_by_decade_and_country(winners, top_k_per_decade=5)
ax2 = plot_decade_country_stacked(by_decade, save_path="../figures/country_by_decade_stacked.png")

    mapping = {
        "m": "male", "male": "male",
        "f": "female", "female": "female",
    }
    out[col] = (out[col].astype(str).str.lower().str.strip()
                .map(mapping).fillna("unknown"))
    return out


def time_to_seconds(df: pd.DataFrame, time_col: str = "time",
                    out_time_col: str = "finish_time",
                    out_seconds_col: str = "finish_seconds") -> pd.DataFrame:
    """
    Parse a time-like column to timedelta and seconds.
    """
    out = df.copy()
    out[out_time_col] = pd.to_timedelta(out[time_col], errors="coerce")
    out[out_seconds_col] = out[out_time_col].dt.total_seconds()
    return out


def clean_runners_pipeline(
    df: pd.DataFrame,
    drop_cols: Iterable[str] = ("country", "age"),
) -> pd.DataFrame:
    """
    Full cleaning used on the participants dataset:
    - normalize columns
    - drop duplicates
    - standardize gender
    - convert time to timedelta + seconds
    - drop irrelevant columns (e.g., country, age if missing-heavy)
    """
    out = normalize_columns(df)
    out = drop_dupes_and_nulls(out)
    out = standardize_gender(out, "gender")
    out = time_to_seconds(out, "time", "finish_time", "finish_seconds")
    out = drop_columns(out, drop_cols)
    return out


# =========================
# Winners dataset cleaning
# (cleaned_marathon_winners.csv)
# =========================

def winners_wide_to_long(
    df: pd.DataFrame,
    year_col: str = "year",
    men_cols: Tuple[str, str, str] = ("MEN_WINNER", "MEN_COUNTRY", "MEN_TIME"),
    women_cols: Tuple[str, str, str] = ("WOMEN_WINNER", "WOMEN_COUNTRY", "WOMEN_TIME"),
) -> pd.DataFrame:
    """
    Convert winners dataset from wide (men*, women*) to long with columns:
    ['year', 'winner', 'country', 'time', 'gender'].
    Assumes MEN_* and WOMEN_* columns exist (case-insensitive handled by normalize_columns).
    """
    temp = normalize_columns(df)

    m_winner, m_country, m_time = [c.lower() for c in men_cols]
    f_winner, f_country, f_time = [c.lower() for c in women_cols]

    men = temp[[year_col, m_winner, m_country, m_time]].copy()
    men.columns = [year_col, "winner", "country", "time"]
    men["gender"] = "male"

    women = temp[[year_col, f_winner, f_country, f_time]].copy()
    women.columns = [year_col, "winner", "country", "time"]
    women["gender"] = "female"

    long_df = pd.concat([men, women], ignore_index=True)
    long_df = clean_strings(long_df, ["winner", "country"])
    long_df = time_to_seconds(long_df, "time", "finish_time", "finish_seconds")
    return long_df


# =========================
# Country dominance analysis
# =========================

def top_countries_by_wins(winners_long: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Return Top-N countries by total wins (based on winners_long with 'country').
    """
    counts = (winners_long["country"]
              .fillna("Unknown")
              .value_counts()
              .rename_axis("country")
              .reset_index(name="wins"))
    return counts.head(n)


def add_decade(df: pd.DataFrame, year_col: str = "year") -> pd.DataFrame:
    """
    Add a 'decade' column as e.g. 1980s, 1990s.
    """
    out = df.copy()
    out["decade"] = (out[year_col] // 10 * 10).astype(int).astype(str) + "s"
    return out


def winners_by_decade_and_country(winners_long: pd.DataFrame,
                                  top_k_per_decade: int = 5) -> pd.DataFrame:
    """
    Compute winners per decade and country, keeping top-K countries within each decade.
    """
    temp = add_decade(winners_long, "year")
    grp = (temp.groupby(["decade", "country"])
                .size()
                .reset_index(name="wins"))
    # rank within decade
    grp["rank_in_decade"] = grp.groupby("decade")["wins"].rank(method="first", ascending=False)
    return grp.loc[grp["rank_in_decade"] <= top_k_per_decade].sort_values(["decade", "wins"], ascending=[True, False])


# =========================
# Plotting helpers
# =========================

def plot_top_countries_bar(df_top: pd.DataFrame,
                           country_col: str = "country",
                           wins_col: str = "wins",
                           title: str = "Top Countries by Wins",
                           figsize: Tuple[int, int] = (8, 5),
                           save_path: Optional[str | Path] = None) -> plt.Axes:
    """
    Simple horizontal bar chart for top countries by wins.
    """
    fig, ax = plt.subplots(figsize=figsize)
    df_plot = df_top.sort_values(wins_col)
    ax.barh(df_plot[country_col], df_plot[wins_col])
    ax.set_xlabel("Wins")
    ax.set_ylabel("Country")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    return ax


def plot_decade_country_stacked(df_decade: pd.DataFrame,
                                decade_col: str = "decade",
                                country_col: str = "country",
                                wins_col: str = "wins",
                                title: str = "Country Dominance by Decade (Top 5 per decade)",
                                figsize: Tuple[int, int] = (9, 5),
                                save_path: Optional[str | Path] = None) -> plt.Axes:
    """
    Stacked bar chart: rows=decades, columns=countries, values=wins (only top-K already filtered).
    """
    pivot = df_decade.pivot_table(index=decade_col, columns=country_col, values=wins_col, aggfunc="sum", fill_value=0)
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=figsize)
    bottom = np.zeros(len(pivot))
    for col in pivot.columns:
        ax.bar(pivot.index, pivot[col].values, bottom=bottom, label=col)
        bottom += pivot[col].values

    ax.set_xlabel("Decade")
    ax.set_ylabel("Wins")
    ax.set_title(title)
    ax.legend(loc="best", ncol=2, fontsize=8)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    return ax
