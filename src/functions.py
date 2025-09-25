from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import yaml  # make sure this import is at the top with the others

# ----------------------------
# CONFIG HELPERS (YAML)
# ----------------------------
def load_config(path: str | Path = "../config.yaml") -> dict:
    """
    Load YAML configuration file.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def read_csv_from_config(config: dict, key: str,
                         low_memory: bool = False,
                         **kwargs) -> pd.DataFrame:
    """
    Read a CSV file using path from config["input_data"][key].
    """
    if "input_data" not in config or key not in config["input_data"]:
        raise KeyError(f"Key '{key}' not found in config['input_data']")
    csv_path = Path(config["input_data"][key])
    return pd.read_csv(csv_path, low_memory=low_memory, **kwargs)

# ----------------------------
# 1) GENERAL UTILITIES
# ----------------------------
def print_shape(df: pd.DataFrame, name: str = "df") -> None:
    """Print DataFrame shape with label."""
    print(f"{name} shape: {df.shape}")


# ----------------------------
# 2) CLEANING HELPERS
# ----------------------------
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, strip, and replace spaces with underscores in column names."""
    df = df.copy()
    df.columns = (df.columns.str.strip()
                              .str.lower()
                              .str.replace(" ", "_")
                              .str.replace("-", "_"))
    return df

def drop_empty_columns(df: pd.DataFrame, thresh: float = 0.5) -> pd.DataFrame:
    """
    Drop columns with more than (1-thresh) fraction of missing values.
    Example: thresh=0.5 drops columns with >50% NaNs.
    """
    df = df.copy()
    limit = len(df) * thresh
    return df.dropna(axis=1, thresh=limit)

def fill_missing_with_unknown(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Fill missing values in specific columns with 'Unknown'."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
    return df

def standardize_text_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Standardize text column: strip, lowercase, replace multiple spaces with single."""
    if column not in df.columns:
        return df
    df = df.copy()
    df[column] = (df[column].astype(str)
                              .str.strip()
                              .str.lower()
                              .str.replace(r"\s+", " ", regex=True))
    return df


# ----------------------------
# 3) WINNERS TRANSFORMATIONS
# ----------------------------
def winners_wide_to_long(winners_df: pd.DataFrame) -> pd.DataFrame:
    """Reshape winners dataset from wide to long format with [year, winner, country, time, gender]."""
    men = winners_df[["year", "men_winner", "men_country", "men_time"]].copy()
    men = men.rename(columns={
        "men_winner": "winner",
        "men_country": "country",
        "men_time": "time"
    })
    men["gender"] = "male"

    women = winners_df[["year", "women_winner", "women_country", "women_time"]].copy()
    women = women.rename(columns={
        "women_winner": "winner",
        "women_country": "country",
        "women_time": "time"
    })
    women["gender"] = "female"

    for df in (men, women):
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    return pd.concat([men, women], ignore_index=True)

def top_countries_by_wins(winners_long: pd.DataFrame) -> pd.DataFrame:
    """Count total wins by country. Returns [country, wins]."""
    return (winners_long["country"]
            .value_counts()
            .reset_index()
            .rename(columns={"index": "country", "country": "wins"}))

def winners_by_decade_and_country(winners_long: pd.DataFrame) -> pd.DataFrame:
    """Aggregate number of wins by decade and country. Returns [decade, country, wins]."""
    tmp = winners_long.copy()
    tmp["year"] = pd.to_numeric(tmp["year"], errors="coerce")
    tmp["decade"] = (tmp["year"] // 10) * 10
    return (tmp.groupby(["decade", "country"])
            .size()
            .reset_index(name="wins")
            .sort_values(["decade", "wins"], ascending=[True, False]))


# ----------------------------
# 4) PLOTTING HELPERS
# ----------------------------
def plot_top_countries_bar(df_top: pd.DataFrame,
                           country_col: str = "country",
                           wins_col: str = "wins",
                           title: str = "Top Countries by Wins",
                           figsize: Tuple[int, int] = (8, 5),
                           save_path: Optional[str | Path] = None) -> plt.Axes:
    """Plot horizontal bar chart of countries by wins."""
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
    """Plot stacked bar chart of wins per decade and country."""
    pivot = (df_decade
             .pivot_table(index=decade_col, columns=country_col,
                          values=wins_col, aggfunc="sum", fill_value=0)
             .sort_index())

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
# ----------------------------
# 5) COMPATIBILITY HELPERS
# ----------------------------

def load_csv(path: str, **kwargs) -> pd.DataFrame:
    """
    Wrapper for pd.read_csv so notebooks using load_csv keep working.
    """
    return pd.read_csv(path, **kwargs)

def save_csv(df: pd.DataFrame, path: str, index: bool = False, **kwargs) -> None:
    """
    Wrapper for df.to_csv so notebooks using save_csv keep working.
    Creates parent directories if needed.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, **kwargs)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Alias for clean_column_names to keep notebooks compatible.
    """
    return clean_column_names(df)

def check_missing(df: pd.DataFrame, name: str = "df") -> None:
    """
    Simple missing values check.
    Prints number of missing values per column.
    """
    missing = df.isnull().sum()
    print(f"Missing values in {name}:")
    print(missing[missing > 0])

def clean_runners_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder cleaning pipeline for runners dataset.
    Currently just returns the DataFrame unchanged.
    Added for compatibility with notebooks that import it.
    """
    # If you want, you can later add real cleaning steps here.
    return df

