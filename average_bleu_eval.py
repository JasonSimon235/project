import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import os

# ----------------------------
# Load BLIP model
# ----------------------------
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
model.eval()

# ----------------------------
# Load dataset
# ----------------------------
data = pd.read_csv("dataset.csv")

IMAGE_FOLDER = "benchmark_images"

smoothie = SmoothingFunction().method1
results = []

# Get all image files once
image_files = os.listdir(IMAGE_FOLDER)

# ----------------------------
# Generate captions + BLEU
# ----------------------------
for idx, row in data.iterrows():
    image_id = str(row["image_id"])
    reference_caption = row["caption"]

    # Find matching file (any extension or no extension)
    matched_file = None
    for fname in image_files:
        if os.path.splitext(fname)[0] == image_id:
            matched_file = fname
            break

    if matched_file is None:
        continue

    image_path = os.path.join(IMAGE_FOLDER, matched_file)
    image = Image.open(image_path).convert("RGB")

    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=25)

    generated_caption = processor.decode(
        output_ids[0], skip_special_tokens=True
    )

    bleu = sentence_bleu(
        [reference_caption.split()],
        generated_caption.split(),
        weights=(1, 0, 0, 0),
        smoothing_function=smoothie
    )

    results.append({
        "image_id": image_id,
        "bleu_score": float(bleu)
    })

    print(f"[{idx}] BLEU-1: {bleu:.4f}")

# ----------------------------
# Save results + average
# ----------------------------
bleu_df = pd.DataFrame(results)

if bleu_df.empty:
    print(" No BLEU scores were generated.")
else:
    bleu_df.to_csv("bleu_scores.csv", index=False)
    print("\n Images evaluated:", len(bleu_df))
    print(" Average BLEU-1:", round(bleu_df["bleu_score"].mean(), 6))
    print(" Saved to bleu_scores.csv")
