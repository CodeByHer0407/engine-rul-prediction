import pandas as pd
from pathlib import Path

# Step 1: Define the dataset location
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train_FD001.txt"


# Step 2: Define column names
columns = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


# Step 3: Load the training dataset
df = pd.read_csv(
    DATA_PATH,
    sep=r"\s+",
    header=None,
    names=columns
)


# Step 4: Explore the dataset
print("Dataset shape:", df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nNumber of engines:")
print(df["engine_id"].nunique())

print("\nMissing values:")
print(df.isnull().sum())


# Step 5: Calculate Remaining Useful Life
max_cycles = df.groupby("engine_id")["cycle"].transform("max")

df["rul"] = max_cycles - df["cycle"]


# Step 6: Verify RUL calculation
print("\nRUL calculation:")
print(df[["engine_id", "cycle", "rul"]].head(10))