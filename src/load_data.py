from pathlib import Path
import pandas as pd


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw"


# Define NASA dataset columns
COLUMNS = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_data(filename: str) -> pd.DataFrame:
    """Load a NASA C-MAPSS dataset file."""

    file_path = DATA_DIR / filename

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=COLUMNS
    )

    return df


def calculate_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate RUL for complete run-to-failure training data."""

    df = df.copy()

    max_cycles = (
        df.groupby("engine_id")["cycle"].transform("max")
    )

    df["rul"] = max_cycles - df["cycle"]

    return df


def get_last_observations(df: pd.DataFrame) -> pd.DataFrame:
    """Get the latest available observation for each engine."""

    df = df.sort_values(
        ["engine_id", "cycle"]
    )

    last_observations = (
        df.groupby("engine_id")
        .tail(1)
        .reset_index(drop=True)
    )

    return last_observations


if __name__ == "__main__":

    train_df = load_data("train_FD001.txt")

    train_df = calculate_rul(train_df)

    print("Training data:", train_df.shape)

    print("Total engines:", train_df["engine_id"].nunique())

    print(train_df[["engine_id", "cycle", "rul"]].head())