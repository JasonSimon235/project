import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("bleu_scores.csv")

avg_bleu = df["bleu_score"].mean()

plt.figure()
plt.bar(["Average BLEU-1"], [avg_bleu])
plt.ylabel("BLEU-1 Score")
plt.title("Average BLEU-1 Score Across Dataset")
plt.show()

print("Average BLEU-1:", avg_bleu)
