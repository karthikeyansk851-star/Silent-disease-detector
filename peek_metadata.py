import pandas as pd
df = pd.read_csv("data/raw/combined_metadata.csv")
print("Columns:", list(df.columns))
print(df.head(3))
