import pandas as pd
import glob

FILES = glob.glob("collector/saved/*.csv")
print(f"Found files: {len(FILES)}\n")

for f in FILES:
    df = pd.read_csv(f)
    shape = df.shape
    
    print(f"{f} -> shape={shape}")

    if shape[1] != 5:
        print(f"❌ WRONG COLUMNS: {f} has {shape[1]} columns (expected 5)\n")
    else:
        print(f"✅ OK (5 features)\n")