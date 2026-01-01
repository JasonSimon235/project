import pandas as pd
import matplotlib.pyplot as plt

bleu_df = pd.read_csv("bleu_scores.csv")

plt.figure()
plt.plot(range(len(bleu_df)), bleu_df["bleu_score"])
plt.xlabel("Image Index")
plt.ylabel("BLEU-1 Score")
plt.title("BLEU-1 Score Across Images")
plt.show()
