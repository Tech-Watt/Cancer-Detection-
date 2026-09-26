"""Shared helpers for cancer-risk EDA and training.

Open `eda.ipynb` for the interactive analysis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "cancer data.csv"
FIGURES_DIR = ROOT / "figures"

ID_COLUMNS = ["index", "Patient Id"]
TARGET = "Level"
LEVEL_ORDER = ["Low", "Medium", "High"]
LEVEL_COLORS = {"Low": "#4C9F70", "Medium": "#E0A100", "High": "#C0392B"}

KEY_FEATURES = [
    "Smoking",
    "Passive Smoker",
    "Alcohol use",
    "Obesity",
    "Genetic Risk",
    "Air Pollution",
    "Coughing of Blood",
    "Chest Pain",
    "chronic Lung Disease",
    "OccuPational Hazards",
    "Fatigue",
    "Shortness of Breath",
]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ID_COLUMNS + [TARGET]]


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    cols = feature_columns(df)
    return df[cols].copy(), df[TARGET].astype(str), cols


def save_fig(name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / name, dpi=160, bbox_inches="tight")
