"""Single 4-bar chart: partial hackathon score per (timeframe × indicator) combo."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

METRICS_DIR = Path("results/metrics")
OUT_DIR     = Path("results/plots")

with open(METRICS_DIR / "results.json") as f:
    data = json.load(f)

df = pd.DataFrame(data)
df = df[df["model"] != "RandomWalk"]
bull = df[df["period"] == "bull_pre_covid"]

# Partial score per (timeframe, indicator): (T + max(0, rho)) / 2
combos = [
    ("daily",  "macd",        "Daily\nMACD"),
    ("weekly", "macd",        "Weekly\nMACD"),
    ("daily",  "stochastic",  "Daily\nStoch"),
    ("weekly", "stochastic",  "Weekly\nStoch"),
]

scores, trend_vals, pearson_vals = [], [], []
for tf, ind, _ in combos:
    sub = bull[(bull["timeframe"] == tf) & (bull["indicator"] == ind)]
    t  = sub["trend_accuracy"].mean()
    rh = max(0, sub["pearson_correlation"].mean())
    scores.append((t + rh) / 2)
    trend_vals.append(t)
    pearson_vals.append(rh)

labels = [c[2] for c in combos]

# Colors: orange family for daily, blue-green for weekly
colors = ["#E07B39", "#2A9D8F", "#F4A261", "#264653"]
edge   = ["#b35a1a", "#1a7a6b", "#c97a30", "#142f38"]

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

x = np.arange(len(labels))
bars = ax.bar(x, scores, width=0.55, color=colors, edgecolor=edge, linewidth=1.2, zorder=3)

# Threshold line at 0.30 (random walk benchmark)
ax.axhline(0.263 / 2, color="#999999", linewidth=1.0, linestyle=":", zorder=2,
           label="Random Walk")

# Value labels on bars
for bar, s, t, r in zip(bars, scores, trend_vals, pearson_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f"{s:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#222222")
    # Sub-labels: T and ρ
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + 0.008,
            f"T={t:.3f}\nρ={r:.3f}", ha="center", va="bottom",
            fontsize=7.5, color="white", fontweight="bold", linespacing=1.4)

# Y-axis: tight range to amplify visual differences
y_min, y_max = 0.22, 0.40
ax.set_ylim(y_min, y_max)
ax.yaxis.set_major_locator(plt.MultipleLocator(0.02))
ax.set_yticks(np.arange(y_min, y_max + 0.001, 0.02))

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("Score partiel  $(T + \\max(0,\\rho))\\,/\\,2$", fontsize=10)
ax.set_title(
    "Comparaison Daily vs Weekly par indicateur\n"
    "(Bull Market pré-COVID — tous modèles, tous actifs)",
    fontsize=12, fontweight="bold", pad=12,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

# Legend patch for colors
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#E07B39", edgecolor="#b35a1a", label="Daily"),
    Patch(facecolor="#2A9D8F", edgecolor="#1a7a6b", label="Weekly"),
]
ax.legend(handles=legend_elements, fontsize=9, loc="upper right", framealpha=0.8)

fig.tight_layout()
out = OUT_DIR / "daily_weekly_4bar.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved {out}")
