import pandas as pd
import matplotlib.pyplot as plt

bleu_df = pd.read_csv("bleu_scores.csv")

zero_bleu = (bleu_df["bleu_score"] == 0).sum()
non_zero_bleu = (bleu_df["bleu_score"] > 0).sum()

plt.figure()
plt.bar(["BLEU = 0", "BLEU > 0"], [zero_bleu, non_zero_bleu])
plt.ylabel("Number of Images")
plt.title("Zero vs Non-Zero BLEU Scores")
plt.show()
