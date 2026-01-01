import pandas as pd
import matplotlib.pyplot as plt

bleu_df = pd.read_csv("bleu_scores.csv")
captions_df = pd.read_csv("generated_captions.csv")

captions_df["caption_length"] = captions_df["generated_caption"].apply(
    lambda x: len(str(x).split())
)

min_len = min(len(bleu_df), len(captions_df))

plt.figure()
plt.scatter(
    captions_df["caption_length"][:min_len],
    bleu_df["bleu_score"][:min_len]
)
plt.xlabel("Generated Caption Length (words)")
plt.ylabel("BLEU-1 Score")
plt.title("Caption Length vs BLEU-1 Score")
plt.show()
