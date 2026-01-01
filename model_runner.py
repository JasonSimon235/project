"""
Model Runner for BLIP Base Captioning Model.
Reverted from BLIP-2 due to hardware limitations.
This version is lightweight and works on CPU systems.
"""

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os

# Optional Hugging Face token
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", None)

# Device setup (CPU or GPU if available)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------------------
# Load BLIP Base Processor & Model (CPU friendly)
# -------------------------------------------------------------------
_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    use_auth_token=HF_TOKEN
)

_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    use_auth_token=HF_TOKEN
).to(DEVICE)


# -------------------------------------------------------------------
# Caption Generation Function
# -------------------------------------------------------------------
def produce_caption(image_stream):
    """
    Takes an image file stream, opens it as a PIL image,
    and generates a caption using the BLIP base model.
    Works on CPU.
    """
    try:
        image = Image.open(image_stream).convert("RGB")
    except Exception:
        return "Unable to read image"

    inputs = _processor(image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=30,
            num_beams=5,           # improves quality
            early_stopping=True
        )

    caption = _processor.decode(output_ids[0], skip_special_tokens=True)
    return caption