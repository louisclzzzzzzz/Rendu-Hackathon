"""Generate regime-comparison heatmaps: hackathon score by model × period."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

METRICS_DIR = Path("results/metrics")
OUT_DIR     = Path("results/plots")

PERIOD_LABELS = {
    "bull_pre_covid": "Bull\n2017–20",
    "covid_crash":    "COVID\n2020–21",
    "bear_inflation": "Bear\n2022",
    "ai_rebound":     "AI Rebound\n2023–24",
    "full_10y":       "Full 10Y\n2015–24",
}
PERIOD_ORDER = list(PERIOD_LABELS.keys())

MODEL_ORDER = [
    "RandomWalk", "ARIMA", "SARIMA", "HoltWinters",
    "RandomForest", "SVR", "LSTM", "GRU", "Transformer",
]

MODEL_LABELS = {
    "RandomWalk":   "Random Walk",
    "ARIMA":        "ARIMA",
    "SARIMA":       "SARIMA",
    "HoltWinters":  "Holt-Winters",
    "RandomForest": "Random Forest",
    "SVR":          "SVR",
    "LSTM":         "LSTM",
    "GRU":          "GRU",
    "Transformer":  "Transformer",
}

# ── Load data ────────────────────────────────────────────────────────────────
with open(METRICS_DIR / "results.json") as f:
    raw = json.load(f)

df = pd.DataFrame(raw)

# Compute hackathon score per record pair (macd + stochastic on same combo)
pivot = df.pivot_table(
    index=["ticker", "timeframe", "period", "model"],
    columns="indicator",
    values=["trend_accuracy", "pearson_correlation"],
    aggfunc="first",
).reset_index()
pivot.columns = [
    f"{c[0]}_{c[1]}" if c[1] else c[0] for c in pivot.columns
]

pivot["hackathon_score"] = (
    pivot["trend_accuracy_macd"].fillna(0)
    + np.maximum(0, pivot["pearson_correlation_macd"].fillna(0))
    + pivot["trend_accuracy_stochastic"].fillna(0)
    + np.maximum(0, pivot["pearson_correlation_stochastic"].fillna(0))
) / 4


def make_heatmap(data, title, filename, vmin=0.20, vmax=0.41, threshold=0.30):
    """data: DataFrame with model × period score matrix."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    cmap = plt.cm.RdYlGn
    im = ax.imshow(data.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    # Axes labels
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(
        [PERIOD_LABELS[p] for p in data.columns],
        fontsize=10, color="black", ha="center",
    )
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(
        [MODEL_LABELS[m] for m in data.index],
        fontsize=10, color="black",
    )
    ax.tick_params(colors="black", length=0)

    # Cell annotations
    for i in range(len(data.index)):
        for j in range(len(data.columns)):
            val = data.values[i, j]
            if np.isnan(val):
                txt, col = "—", "#888888"
            else:
                txt = f"{val:.3f}"
                # dark text on light cells, white on dark cells
                norm_val = (val - vmin) / (vmax - vmin)
                col = "white" if norm_val > 0.65 or norm_val < 0.25 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=9, color=col, fontweight="bold")

    # Separator line after RandomWalk row
    ax.axhline(0.5, color="#555555", linewidth=1.2, linestyle="--", alpha=0.7)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="black", labelsize=9)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="black")
    cbar.set_label("Score hackathon", color="black", fontsize=10)

    # Threshold annotation on colorbar
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cbar.ax.axhline(norm(threshold), color="black", linewidth=1.5, linestyle="--")
    cbar.ax.text(
        1.6, norm(threshold), f" seuil {threshold}",
        va="center", ha="left", color="black", fontsize=8,
        transform=cbar.ax.transAxes,
    )

    ax.set_title(title, color="black", fontsize=13, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"Saved {filename}")


# ── Heatmap 1: all timeframes combined ───────────────────────────────────────
agg = (
    pivot
    .groupby(["model", "period"])["hackathon_score"]
    .mean()
    .reset_index()
)
matrix_all = agg.pivot(index="model", columns="period", values="hackathon_score")
matrix_all = matrix_all.reindex(index=MODEL_ORDER, columns=PERIOD_ORDER)

make_heatmap(
    matrix_all,
    "Score hackathon moyen par modèle et régime de marché\n(tous actifs, daily + weekly)",
    "regime_heatmap_all.png",
)

# ── Heatmap 2: weekly only ────────────────────────────────────────────────────
agg_w = (
    pivot[pivot["timeframe"] == "weekly"]
    .groupby(["model", "period"])["hackathon_score"]
    .mean()
    .reset_index()
)
matrix_w = agg_w.pivot(index="model", columns="period", values="hackathon_score")
matrix_w = matrix_w.reindex(index=MODEL_ORDER, columns=PERIOD_ORDER)

make_heatmap(
    matrix_w,
    "Score hackathon moyen par modèle et régime de marché\n(granularité Weekly — tous actifs)",
    "regime_heatmap_weekly.png",
)
