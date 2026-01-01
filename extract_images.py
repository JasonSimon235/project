import pandas as pd
import io
import os

PARQUET_SOURCE = "val-00002-of-00007-51c2c861421b1198.parquet"
TARGET_FOLDER = "benchmark_images"
LIMIT = 250


# -------------------------------------------------------------
# Ensure output directory exists
# -------------------------------------------------------------
def prepare_output_folder(folder: str):
    """
    Creates the target directory if it does not already exist.
    """
    if not os.path.isdir(folder):
        os.makedirs(folder)


# -------------------------------------------------------------
# Load parquet into DataFrame
# -------------------------------------------------------------
def load_dataset(path: str) -> pd.DataFrame:
    """
    Loads the parquet dataset and returns it.
    """
    return pd.read_parquet(path)


# -------------------------------------------------------------
# Save a single HF image (bytes) to disk
# -------------------------------------------------------------
def write_image_file(byte_data: bytes, filename: str, destination: str):
    """
    Saves raw image bytes into the specified output directory.
    """
    image_stream = io.BytesIO(byte_data)
    output_path = os.path.join(destination, filename)

    with open(output_path, "wb") as outfile:
        outfile.write(image_stream.getbuffer())

    return output_path


# -------------------------------------------------------------
# Main Extraction Routine
# -------------------------------------------------------------
def extract_sample_images(parquet_path: str, out_dir: str, count: int):
    """
    Reads the parquet file, pulls out the first N images,
    and writes them to the output directory.
    """
    print("Loading dataset...")
    df = load_dataset(parquet_path)

    print(f"Preparing to export {count} images into '{out_dir}'...")
    prepare_output_folder(out_dir)

    for idx in range(count):
        entry = df.iloc[idx]

        img_bytes = entry["image"]["bytes"]
        img_name = entry["image_id"]

        saved_path = write_image_file(img_bytes, img_name, out_dir)
        print(f"[{idx}] → Saved: {saved_path}")

    print("\n✔ Extraction complete. Images are ready for evaluation.")


# -------------------------------------------------------------
# Script Entry Point
# -------------------------------------------------------------
if __name__ == "__main__":
    extract_sample_images(PARQUET_SOURCE, TARGET_FOLDER, LIMIT)
