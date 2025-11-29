import pandas as pd
import pyarrow.parquet as pq
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import io

# ------------------------------------------------------------
# 1. Load Parquet File
# ------------------------------------------------------------
def load_parquet(path):
    df = pd.read_parquet(path)
    return df


# ------------------------------------------------------------
# 2. Load BLIP Caption Model
# ------------------------------------------------------------
def load_caption_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model


# ------------------------------------------------------------
# 3. Convert HF image bytes to PIL
# ------------------------------------------------------------
def load_hf_image(image_obj):
    # HuggingFace parquet image format: {'bytes': b'...'}
    if isinstance(image_obj, dict) and "bytes" in image_obj:
        img_bytes = image_obj["bytes"]
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    raise ValueError("Image format not recognized. Expected dict with 'bytes'.")


# ------------------------------------------------------------
# 4. Generate Caption
# ------------------------------------------------------------
def generate_caption(image, processor, model):
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=40)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


# ------------------------------------------------------------
# 5. Main Pipeline
# ------------------------------------------------------------
def run_caption_pipeline(parquet_file):
    df = load_parquet(parquet_file)
    processor, caption_model = load_caption_model()

    generated_data = []

    for i, row in df.iterrows():
        # Extract image
        image = load_hf_image(row["image"])

        # Generate caption
        generated_caption = generate_caption(image, processor, caption_model)

        generated_data.append({
            "generated_caption": generated_caption
        })

        print(f"[{i}] Caption: {generated_caption}")

    # Save to CSV for Flask backend
    output_df = pd.DataFrame(generated_data)
    output_df.to_csv("generated_captions.csv", index=False)

    print("\n✔ Captions saved to generated_captions.csv")


# ------------------------------------------------------------
# 6. Run Script
# ------------------------------------------------------------
if __name__ == "__main__":
    parquet_file = "val-00002-of-00007-51c2c861421b1198.parquet"
    run_caption_pipeline(parquet_file)
