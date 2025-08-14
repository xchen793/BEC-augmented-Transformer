import pandas as pd
import matplotlib.pyplot as plt

# Load CSV file
csv_path = "results.csv"  # <-- Change this path if your file is named differently or in another directory
df = pd.read_csv(csv_path)

# Plot setup
fig, ax1 = plt.subplots(figsize=(9, 6.5))

# Left Y-axis: Validation Accuracy
color_acc = "tab:blue"
color_acc_awgn = "tab:cyan"
ax1.set_xlabel("p (dropout rate)", fontsize=18)
ax1.set_ylabel("Validation Accuracy", color=color_acc, fontsize=18)
acc_line, = ax1.plot(df["p"], df["val_acc"], marker="o", color=color_acc, label="Validation Accuracy")
acc_awgn_line, = ax1.plot(df["p"], df["val_acc_awgn"], marker="^", linestyle="--", color=color_acc_awgn, label="Validation Accuracy (AWGN)")
ax1.tick_params(axis='y', labelcolor=color_acc, labelsize=14)
ax1.tick_params(axis='x', labelsize=14)


# Right Y-axis: BLEU Score
ax2 = ax1.twinx()
color_bleu = "tab:red"
color_bleu_awgn = "tab:pink"
ax2.set_ylabel("Average BLEU", color=color_bleu, fontsize=18)
bleu_line, = ax2.plot(df["p"], df["bleu"], marker="s", linestyle=":", color=color_bleu, label="BLEU")
bleu_awgn_line, = ax2.plot(df["p"], df["bleu_awgn"], marker="x", linestyle="--", color=color_bleu_awgn, label="BLEU (AWGN)")
ax2.tick_params(axis='y', labelcolor=color_bleu, labelsize=14)

# Combine legends
lines = [acc_line, acc_awgn_line, bleu_line, bleu_awgn_line]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc="upper right")

plt.tight_layout()
plt.savefig("p_acc_bleu_plot.png", dpi=500)
