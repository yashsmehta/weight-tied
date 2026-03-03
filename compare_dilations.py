"""
Overlay test accuracy curves for the four dilation schedule ablations.

Usage:
    python compare_dilations.py

Run from the directory containing the history_*.json files.
"""
import json
import matplotlib.pyplot as plt

CONFIGS = [
    ("1-1-1-1-1-1", "No dilation [1,1,1,1,1,1]",     "steelblue",  "--"),
    ("1-2-3-1-2-3", "Cycling [1,2,3,1,2,3]",          "darkorange", "--"),
    ("3-2-1-2-1-1", "Reversed [3,2,1,2,1,1]",         "green",      "--"),
    ("1-1-2-1-2-3", "Original [1,1,2,1,2,3] (ours)",  "crimson",    "-"),
]

fig, ax = plt.subplots(figsize=(9, 5))

for dil_str, label, color, linestyle in CONFIGS:
    path = f"history_depth6_dil{dil_str}.json"
    try:
        with open(path) as f:
            history = json.load(f)
    except FileNotFoundError:
        print(f"Missing: {path} — skipping")
        continue

    epochs = range(1, len(history["test_acc"]) + 1)
    ax.plot(epochs, history["test_acc"],
            label=f"{label}  (best: {max(history['test_acc']):.1f}%)",
            color=color, linestyle=linestyle,
            linewidth=2.5 if linestyle == "-" else 1.5)

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Test Accuracy (%)", fontsize=12)
ax.set_title("Dilation Schedule Ablation — ECTiedNet on CIFAR-10", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()

plt.savefig("compare_dilations.png", dpi=150, bbox_inches="tight")
print("Saved compare_dilations.png")
plt.show()
