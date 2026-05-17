import matplotlib.pyplot as plt

# Create figure
fig, ax = plt.subplots(figsize=(14, 5))
ax.axis("off")

# -------------------------------------------------
# Block definitions (text, x, y, width, height)
# -------------------------------------------------
blocks = [
    ("Input\n12 × 1000 ECG",               0.05, 0.5, 0.18, 0.22),

    ("Conv Block 1\n1D Conv + BN + ReLU",   0.30, 0.5, 0.22, 0.22),
    ("Conv Block 2\nDeeper Temporal CNN",  0.55, 0.5, 0.24, 0.22),

    ("Attention\nTemporal + Lead-wise",    0.85, 0.5, 0.26, 0.22),

    ("Global Pooling",                     1.18, 0.5, 0.20, 0.18),

    ("FC Classifier\n5 Classes",           1.45, 0.5, 0.22, 0.22),
]

# -------------------------------------------------
# Draw blocks
# -------------------------------------------------
for text, x, y, w, h in blocks:
    ax.text(
        x, y, text,
        ha="center", va="center",
        fontsize=11,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#e6e6e6",
            edgecolor="black",
            linewidth=1.5
        )
    )

# -------------------------------------------------
# Draw arrows
# -------------------------------------------------
for i in range(len(blocks) - 1):
    x1 = blocks[i][1] + blocks[i][3] / 2
    x2 = blocks[i + 1][1] - blocks[i + 1][3] / 2

    ax.annotate(
        "",
        xy=(x2, 0.5),
        xytext=(x1, 0.5),
        arrowprops=dict(arrowstyle="->", lw=2)
    )

# -------------------------------------------------
# Group labels (subsystem annotation)
# -------------------------------------------------
ax.text(0.43, 0.75, "Feature Extraction (CNN)",
        ha="center", va="center", fontsize=12, fontweight="bold")

ax.text(0.85, 0.75, "Attention Mechanism",
        ha="center", va="center", fontsize=12, fontweight="bold")

ax.text(1.42, 0.75, "Classification Head",
        ha="center", va="center", fontsize=12, fontweight="bold")

# -------------------------------------------------
# Save
# -------------------------------------------------
plt.tight_layout()
plt.savefig("visuals/architecture_placeholder.png", dpi=300, bbox_inches="tight")
plt.close()
