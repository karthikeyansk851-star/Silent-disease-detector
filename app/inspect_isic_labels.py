"""
inspect_isic_labels.py

Run this AFTER `isic metadata download` completes, and BEFORE downloading
any images. It finds the exact column name and label spelling ISIC uses
for diagnosis, so the filtered image download in the next step actually
works (ISIC has changed this field's name/format across tool versions).

Usage:
    python inspect_isic_labels.py path\to\metadata.csv
"""
import sys
import pandas as pd

def main(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)

    print(f"Total rows: {len(df)}")
    print(f"\nAll columns:\n{list(df.columns)}\n")

    # Common candidate column names across different isic-cli versions
    candidates = [c for c in df.columns if "diagnosis" in c.lower()]
    print(f"Diagnosis-related columns found: {candidates}\n")

    for col in candidates:
        print(f"--- Unique values in '{col}' ---")
        print(df[col].value_counts(dropna=False).head(30))
        print()

    print(
        "Look through the lists above for the values that correspond to "
        "melanoma and actinic keratosis. Use that EXACT column name and "
        "EXACT string in the next download step, e.g.:\n"
        '  isic image download --search \'<column>:"<exact label>"\' images/mel\n'
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_isic_labels.py path\\to\\metadata.csv")
        sys.exit(1)
    main(sys.argv[1])