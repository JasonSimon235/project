import pandas as pd

# Configuration
PARQUET_PATH = "val-00002-of-00007-51c2c861421b1198.parquet"
OUTPUT_CSV = "dataset.csv"
MAX_IMAGES = 250   # Set desired number of rows here

print("Loading dataset parquet file...")
df = pd.read_parquet(PARQUET_PATH)

print("Original dataset size:", len(df))

# Select first N rows
df = df.head(MAX_IMAGES)

# Create output CSV structure
df_out = pd.DataFrame({
    "image_id": df["image_id"].astype(int),
    "caption": df["caption"]
})

# Save to CSV
df_out.to_csv(OUTPUT_CSV, index=False)

print("--------------------------------------------------")
print(f"dataset.csv successfully created with {len(df_out)} rows")
print(f"Saved at: {OUTPUT_CSV}")
print("Columns:", df_out.columns.tolist())
print("--------------------------------------------------")
