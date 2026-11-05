import sys
import pandas as pd

def main(csv_path):
    try:
        df = pd.read_csv(csv_path, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        print("[INFO] UTF-8 failed, retrying as UTF-16...\n")
        df = pd.read_csv(csv_path, low_memory=False, encoding="utf-16")

    print(f"Total rows: {len(df)}")
    print(f"\nAll columns:\n{list(df.columns)}\n")

    candidates = [c for c in df.columns if "diagnosis" in c.lower()]
    print(f"Diagnosis-related columns found: {candidates}\n")

    for col in candidates:
        print(f"--- Unique values in '{col}' ---")
        print(df[col].value_counts(dropna=False).head(30))
        print()

if __name__ == "__main__":
    main(sys.argv[1])
