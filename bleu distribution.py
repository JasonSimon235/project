import pandas as pd
import matplotlib.pyplot as plt

# Load BLEU scores
df = pd.read_csv("bleu_scores.csv")

# Plot histogram
plt.figure()
plt.hist(df["bleu_score"], bins=30)
plt.xlabel("BLEU-1 Score")
plt.ylabel("Number of Images")
plt.title("Distribution of BLEU-1 Scores Across 250 Images")
plt.show()
