import pandas as pd
from pathlib import Path

file_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "earthquakes.csv"
df = pd.read_csv(file_path, parse_dates=["time"])

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN TYPES")
print(df.dtypes)

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE EVENT IDs")
print(df["event_id"].duplicated().sum())

print("\nMAGNITUDE SUMMARY")
print(df["magnitude"].describe())

print("\nDATE RANGE")
print(df["time"].min())
print(df["time"].max())

print("\nFIRST 5 RECORDS")
print(df.head())