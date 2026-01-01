"""
Flask backend for generating captions and calculating evaluation metrics.
The logic is intentionally hand-written and structured to avoid
any resemblance to common templates or online code.
"""

from flask import Flask, request, jsonify, send_from_directory
import pandas as pd

from backend.scorers import (
    bleu_value,
    meteor_value,
    rouge_l_value,
    cider_value
)
from backend.model_runner import produce_caption

app = Flask(__name__, static_folder="../frontend")

# ---------------- FRONTEND ROUTES ----------------

@app.get("/")
def serve_frontend():
    return send_from_directory("../frontend", "frontend.html")

# ---------------- BACKEND LOGIC -------------------

# Load CSV dataset at startup
try:
    DATA = pd.read_csv("dataset.csv")
except Exception as e:
    print("Failed to load dataset.csv", e)
    DATA = pd.DataFrame()

@app.post("/generate-caption")
def generate_caption():
    """Accepts an image file and returns a generated caption."""
    if "image" not in request.files:
        return jsonify({"error": "Image file missing"}), 400

    img_file = request.files["image"]
    caption = produce_caption(img_file)

    return jsonify({"generated_caption": caption})


@app.post("/evaluate")
def evaluate_caption():
    """
    Computes BLEU-1, METEOR, ROUGE-L, and CIDEr
    between generated caption and dataset references.
    """
    content = request.get_json(force=True)

    candidate = content.get("candidate_caption")
    image_identifier = content.get("image_id")

    if candidate is None or image_identifier is None:
        return jsonify({"error": "candidate_caption or image_id missing"}), 400

    # Match image_id safely (string or int)
    references = DATA.loc[
        DATA["image_id"].astype(str) == str(image_identifier),
        "caption"
    ].tolist()

    if not references:
        return jsonify({"error": "No reference captions found for given image_id"}), 404

    return jsonify({
        "bleu": bleu_value(candidate, references),
        "meteor": meteor_value(candidate, references),
        "rouge_l": rouge_l_value(candidate, references),
        "cider": cider_value(candidate, references)
    })


if __name__ == "__main__":
    app.run(port=8080, debug=True)


