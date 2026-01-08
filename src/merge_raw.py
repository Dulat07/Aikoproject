import pandas as pd
import glob
import os

RAW_DIR = "collector/saved/"
OUT_PATH = "data/raw/participant1.csv"

os.makedirs("data/raw", exist_ok=True)

files = glob.glob(f"{RAW_DIR}/participant_*.csv")

print(f"Found {len(files)} raw CSV files")

dfs = []
for f in files:
    try:
        df = pd.read_csv(f)
        if set(["key","hold","time"]).issubset(df.columns):
            dfs.append(df)
        else:
            print("Skipping invalid file:", f)
    except:
        print("Failed to read:", f)

merged = pd.concat(dfs, ignore_index=True)
merged.to_csv(OUT_PATH, index=False)

print("Saved merged CSV to", OUT_PATH)
print("Rows:", len(merged))